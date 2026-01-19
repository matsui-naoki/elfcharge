"""
Spatial partitioning module for BadELF

Implements:
- Voronoi-like partitioning for atomic regions (at ELF minima planes)
- Bader-like zero-flux surface partitioning for electride regions
- Combined partitioning scheme
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import watershed_ift, label, distance_transform_edt
from dataclasses import dataclass

from .io import GridData
from .analysis import BondPair, ElectrideSite, AtomRadii


@dataclass
class PartitionResult:
    """Result of spatial partitioning"""
    labels: np.ndarray           # (NGX, NGY, NGZ) region labels
    n_atoms: int                 # Number of atoms
    n_electride_sites: int       # Number of electride sites
    atom_labels: List[int]       # Labels corresponding to atoms (1 to n_atoms)
    electride_labels: List[int]  # Labels corresponding to electride sites


class VoronoiPartitioner:
    """
    Voronoi-like partitioner for atomic regions

    Uses ELF minima positions to define partitioning planes between atoms
    """

    def __init__(self, grid_data: GridData):
        """
        Initialize partitioner

        Parameters
        ----------
        grid_data : GridData
            Reference grid data for structure info
        """
        self.grid_data = grid_data
        self.lattice = grid_data.lattice
        self.frac_coords = grid_data.frac_coords
        self.ngrid = grid_data.ngrid

    def simple_voronoi(self) -> np.ndarray:
        """
        Simple Voronoi partitioning based on nearest atom distance

        Returns
        -------
        np.ndarray
            (NGX, NGY, NGZ) array with atom index labels (0 to n_atoms-1)
        """
        ngx, ngy, ngz = self.ngrid
        n_atoms = self.grid_data.n_atoms
        lattice = self.lattice

        # Create grid of fractional coordinates
        fx = np.linspace(0, 1, ngx, endpoint=False)
        fy = np.linspace(0, 1, ngy, endpoint=False)
        fz = np.linspace(0, 1, ngz, endpoint=False)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        grid_frac = np.stack([FX, FY, FZ], axis=-1)  # (ngx, ngy, ngz, 3)

        # Calculate distance to each atom with minimum image convention
        min_dist_sq = np.full((ngx, ngy, ngz), np.inf)
        labels = np.zeros((ngx, ngy, ngz), dtype=np.int32)

        for i in range(n_atoms):
            atom_frac = self.frac_coords[i]

            # Difference in fractional coordinates
            diff = grid_frac - atom_frac
            # Minimum image convention
            diff = diff - np.round(diff)
            # Convert to Cartesian
            diff_cart = np.einsum('...j,jk->...k', diff, lattice)
            # Distance squared
            dist_sq = np.sum(diff_cart ** 2, axis=-1)

            # Update nearest atom
            closer = dist_sq < min_dist_sq
            labels[closer] = i
            min_dist_sq[closer] = dist_sq[closer]

        return labels

    def partition_with_elf_planes(self, bonds: List[BondPair]) -> np.ndarray:
        """
        Partitioning based on ELF minimum positions along bonds

        For each bond, the dividing plane passes through the ELF minimum
        position and is perpendicular to the bond vector. This implements
        the BadELF algorithm's atomic boundary definition.

        Algorithm:
        1. Start with simple Voronoi (nearest atom)
        2. For each unique atom pair, find the "best" ELF minimum
           (lowest ELF value = clearest boundary)
        3. Adjust boundary using the ELF minimum position

        The key insight is that the ELF minimum position determines the
        splitting ratio between atoms i and j. If t_min = dist_i / (dist_i + dist_j),
        then points closer than t_min * bond_length to atom i belong to i.

        Parameters
        ----------
        bonds : List[BondPair]
            List of analyzed bonds with ELF minimum positions

        Returns
        -------
        np.ndarray
            (NGX, NGY, NGZ) array with atom index labels (0 to n_atoms-1)
        """
        # Start with simple Voronoi as base
        labels = self.simple_voronoi()

        ngx, ngy, ngz = self.ngrid
        lattice = self.lattice

        # Create grid coordinates (fractional)
        fx = np.linspace(0, 1, ngx, endpoint=False)
        fy = np.linspace(0, 1, ngy, endpoint=False)
        fz = np.linspace(0, 1, ngz, endpoint=False)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        grid_frac = np.stack([FX, FY, FZ], axis=-1)

        # Group bonds by atom pair and find the best ELF minimum for each pair
        # "Best" = lowest ELF value (clearest boundary between atoms)
        pair_bonds = {}
        for bond in bonds:
            if bond.elf_minimum_frac is None:
                continue

            i, j = bond.atom_i, bond.atom_j
            pair = (min(i, j), max(i, j))

            if pair not in pair_bonds:
                pair_bonds[pair] = bond
            elif bond.elf_minimum_value < pair_bonds[pair].elf_minimum_value:
                pair_bonds[pair] = bond

        # Process each unique atom pair
        for pair, bond in pair_bonds.items():
            i, j = bond.atom_i, bond.atom_j

            # Only adjust points currently assigned to i or j
            mask_i = labels == i
            mask_j = labels == j
            mask_bond = mask_i | mask_j

            if not np.any(mask_bond):
                continue

            # Get atom positions (base cell)
            frac_i = self.frac_coords[i]
            frac_j = self.frac_coords[j]

            # ELF minimum defines the splitting ratio
            # radius_i = distance from atom i to ELF minimum
            # radius_j = distance from atom j to ELF minimum
            radius_i = bond.distance_i
            radius_j = bond.distance_j

            # For each grid point in the bond region, compute distances to both atoms
            # using minimum image convention, then compare with ELF-defined radii
            grid_points = grid_frac[mask_bond]

            # Distance to atom i (minimum image)
            diff_i = grid_points - frac_i
            diff_i = diff_i - np.round(diff_i)
            diff_i_cart = np.einsum('...j,jk->...k', diff_i, lattice)
            dist_i = np.linalg.norm(diff_i_cart, axis=-1)

            # Distance to atom j (minimum image)
            diff_j = grid_points - frac_j
            diff_j = diff_j - np.round(diff_j)
            diff_j_cart = np.einsum('...j,jk->...k', diff_j, lattice)
            dist_j = np.linalg.norm(diff_j_cart, axis=-1)

            # Normalized distances: dist / radius
            # If dist_i / radius_i < dist_j / radius_j, point belongs to i
            # This is equivalent to checking which "scaled Voronoi" region the point is in
            # Using radii from ELF minima rather than equal radii
            with np.errstate(divide='ignore', invalid='ignore'):
                scaled_dist_i = np.where(radius_i > 0, dist_i / radius_i, np.inf)
                scaled_dist_j = np.where(radius_j > 0, dist_j / radius_j, np.inf)

            # Points closer to i (in scaled distance)
            closer_to_i = scaled_dist_i < scaled_dist_j
            closer_to_j = scaled_dist_j < scaled_dist_i

            # Create index arrays for masked positions
            mask_bond_indices = np.where(mask_bond)
            current_labels = labels[mask_bond]

            # Reassign based on scaled distances
            reassign_to_i = (current_labels == j) & closer_to_i
            reassign_to_j = (current_labels == i) & closer_to_j

            # Apply reassignments
            if np.any(reassign_to_j):
                idx = tuple(arr[reassign_to_j] for arr in mask_bond_indices)
                labels[idx] = j

            if np.any(reassign_to_i):
                idx = tuple(arr[reassign_to_i] for arr in mask_bond_indices)
                labels[idx] = i

        return labels


class BaderPartitioner:
    """
    Bader-like zero-flux surface partitioner based on ELF

    Uses watershed segmentation on inverted ELF to find basins
    """

    def __init__(self, elf_data: GridData):
        """
        Initialize Bader partitioner

        Parameters
        ----------
        elf_data : GridData
            ELF grid data
        """
        self.elf_data = elf_data
        self.elf_grid = elf_data.grid
        self.ngrid = elf_data.ngrid

    def steepest_ascent_partition(self) -> np.ndarray:
        """
        Partition space by following steepest ascent paths to ELF maxima

        This is the accurate but slower method.

        Returns
        -------
        np.ndarray
            (NGX, NGY, NGZ) array with basin labels
        """
        ngx, ngy, ngz = self.ngrid
        grid = self.elf_grid

        # Initialize labels
        labels = np.zeros((ngx, ngy, ngz), dtype=np.int32)
        visited = np.zeros((ngx, ngy, ngz), dtype=bool)

        # Find all maxima first
        maxima_labels = {}
        current_label = 1

        def get_neighbors(ix, iy, iz):
            """Get 26-connected neighbors with periodic boundaries"""
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nx = (ix + dx) % ngx
                        ny = (iy + dy) % ngy
                        nz = (iz + dz) % ngz
                        neighbors.append((nx, ny, nz))
            return neighbors

        def is_maximum(ix, iy, iz):
            """Check if point is a local maximum"""
            val = grid[ix, iy, iz]
            for nx, ny, nz in get_neighbors(ix, iy, iz):
                if grid[nx, ny, nz] > val:
                    return False
            return True

        def steepest_neighbor(ix, iy, iz):
            """Find neighbor with highest ELF value"""
            best = (ix, iy, iz)
            best_val = grid[ix, iy, iz]
            for nx, ny, nz in get_neighbors(ix, iy, iz):
                if grid[nx, ny, nz] > best_val:
                    best = (nx, ny, nz)
                    best_val = grid[nx, ny, nz]
            return best

        # Process each grid point
        for ix in range(ngx):
            for iy in range(ngy):
                for iz in range(ngz):
                    if visited[ix, iy, iz]:
                        continue

                    # Follow steepest ascent path
                    path = [(ix, iy, iz)]
                    cx, cy, cz = ix, iy, iz

                    while True:
                        visited[cx, cy, cz] = True
                        nx, ny, nz = steepest_neighbor(cx, cy, cz)

                        if (nx, ny, nz) == (cx, cy, cz):
                            # Reached maximum
                            break

                        if visited[nx, ny, nz] and labels[nx, ny, nz] > 0:
                            # Reached already labeled region
                            break

                        path.append((nx, ny, nz))
                        cx, cy, cz = nx, ny, nz

                    # Assign label
                    if labels[cx, cy, cz] > 0:
                        label_val = labels[cx, cy, cz]
                    elif (cx, cy, cz) in maxima_labels:
                        label_val = maxima_labels[(cx, cy, cz)]
                    else:
                        label_val = current_label
                        maxima_labels[(cx, cy, cz)] = current_label
                        current_label += 1

                    # Label all points on path
                    for px, py, pz in path:
                        labels[px, py, pz] = label_val

        return labels

    def watershed_partition(self, markers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Partition space using watershed segmentation on inverted ELF

        This is faster than steepest ascent but may be less accurate.

        Parameters
        ----------
        markers : np.ndarray, optional
            Pre-defined markers for watershed. If None, use detected maxima.

        Returns
        -------
        np.ndarray
            (NGX, NGY, NGZ) array with basin labels
        """
        grid = self.elf_grid.copy()

        # Convert to integer for watershed_ift
        # Invert so that maxima become minima (watershed finds basins)
        # watershed_ift requires uint8 or uint16
        grid_int = ((1.0 - grid) * 65535).astype(np.uint16)

        # Pad for periodic boundaries
        pad_size = 1
        grid_padded = np.pad(grid_int, pad_size, mode='wrap')

        if markers is None:
            # Detect maxima as markers
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(self.elf_grid, size=3, mode='wrap')
            is_max = (self.elf_grid == local_max) & (self.elf_grid > 0.3)

            # Label connected components of maxima
            markers, n_features = label(is_max)
            markers = markers.astype(np.int32)

        # Pad markers
        markers_padded = np.pad(markers, pad_size, mode='wrap')

        # Run watershed
        labels_padded = watershed_ift(grid_padded, markers_padded)

        # Remove padding
        labels = labels_padded[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

        return labels


class BadELFPartitioner:
    """
    Combined partitioner implementing the BadELF algorithm

    - Voronoi-like partitioning at ELF minima for atom-atom boundaries
    - Bader-like zero-flux partitioning for electride boundaries
    """

    def __init__(self, elf_data: GridData, chg_data: Optional[GridData] = None):
        """
        Initialize BadELF partitioner

        Parameters
        ----------
        elf_data : GridData
            ELF grid data
        chg_data : GridData, optional
            Charge density data (same structure as ELF)
        """
        self.elf_data = elf_data
        self.chg_data = chg_data
        self.voronoi = VoronoiPartitioner(elf_data)
        self.bader = BaderPartitioner(elf_data)

    def partition(
        self,
        bonds: List[BondPair],
        electride_sites: List[ElectrideSite],
        method: str = 'watershed',
        use_elf_planes: bool = False
    ) -> PartitionResult:
        """
        Perform BadELF partitioning

        Strategy (following BadELF paper):
        1. Atom-atom boundaries: Voronoi with ELF minima planes
        2. Atom-electride boundaries: Zero-flux surfaces in ELF gradient
           (implemented via watershed on inverted ELF)

        The key insight is that we use zero-flux (watershed) to determine
        which regions belong to electride sites, but we preserve atomic
        regions from Voronoi partitioning to maintain chemically reasonable
        atomic volumes.

        Parameters
        ----------
        bonds : List[BondPair]
            Analyzed bonds with ELF minima
        electride_sites : List[ElectrideSite]
            Detected electride sites
        method : str
            'watershed' or 'steepest_ascent' for zero-flux partitioning

        Returns
        -------
        PartitionResult
            Partitioning result with labels
        """
        n_atoms = self.elf_data.n_atoms
        n_electride = len(electride_sites)
        ngrid = self.elf_data.ngrid

        # Step 1: Voronoi partition for atoms (with optional ELF plane adjustment)
        if use_elf_planes:
            atom_labels = self.voronoi.partition_with_elf_planes(bonds)
        else:
            atom_labels = self.voronoi.simple_voronoi()

        # If no electride sites, just return atom partition
        if n_electride == 0:
            return PartitionResult(
                labels=atom_labels + 1,  # Convert to 1-indexed
                n_atoms=n_atoms,
                n_electride_sites=0,
                atom_labels=list(range(1, n_atoms + 1)),
                electride_labels=[]
            )

        # Step 2: Extended Voronoi including electride sites
        # This determines atom-electride boundaries
        lattice = self.elf_data.lattice

        # All site coordinates (atoms + electrides)
        all_frac_coords = np.vstack([
            self.elf_data.frac_coords,
            np.array([site.frac_coord for site in electride_sites])
        ])

        # Create fractional grid
        fx = np.linspace(0, 1, ngrid[0], endpoint=False)
        fy = np.linspace(0, 1, ngrid[1], endpoint=False)
        fz = np.linspace(0, 1, ngrid[2], endpoint=False)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        grid_frac = np.stack([FX, FY, FZ], axis=-1)

        # Find nearest site for each grid point (Voronoi for all sites)
        min_dist_sq = np.full(ngrid, np.inf)
        nearest_site = np.zeros(ngrid, dtype=np.int32)

        for i in range(n_atoms + n_electride):
            site_frac = all_frac_coords[i]
            diff = grid_frac - site_frac
            diff = diff - np.round(diff)  # Minimum image
            diff_cart = np.einsum('...j,jk->...k', diff, lattice)
            dist_sq = np.sum(diff_cart ** 2, axis=-1)

            closer = dist_sq < min_dist_sq
            nearest_site[closer] = i
            min_dist_sq[closer] = dist_sq[closer]

        # Step 3: Create markers for watershed (for electride-electride boundaries)
        markers = np.zeros(ngrid, dtype=np.int32)

        # Set electride markers at ELF maxima positions
        for i, site in enumerate(electride_sites):
            markers[site.grid_index] = n_atoms + i + 1

        # Step 4: Run zero-flux partitioning on ELF for electride regions only
        # This handles electride-electride boundaries correctly
        if method == 'watershed' and n_electride > 1:
            # Only run watershed on the electride regions (where nearest_site >= n_atoms)
            electride_region_mask = nearest_site >= n_atoms

            # Create markers only for electride sites
            electride_markers = np.zeros(ngrid, dtype=np.int32)
            for i, site in enumerate(electride_sites):
                electride_markers[site.grid_index] = i + 1  # 1-indexed for watershed

            # Run watershed
            zeroflux_labels = self.bader.watershed_partition(electride_markers)
        else:
            zeroflux_labels = None

        # Step 5: Combine partitioning results
        # Start with atom Voronoi labels (1-indexed)
        final_labels = atom_labels + 1

        # Assign electride regions
        for i in range(n_electride):
            # Use Voronoi result for atom-electride boundary
            voronoi_electride_mask = nearest_site == (n_atoms + i)

            if zeroflux_labels is not None and n_electride > 1:
                # For electride-electride boundaries, use watershed result
                # Only apply within the Voronoi electride region
                electride_label = n_atoms + i + 1

                # Watershed label for this electride (1-indexed in watershed)
                watershed_mask = (zeroflux_labels == (i + 1))

                # Combine: must be in overall electride region (any electride's Voronoi)
                # AND assigned to this electride by watershed
                overall_electride_region = nearest_site >= n_atoms
                combined_mask = overall_electride_region & watershed_mask

                final_labels[combined_mask] = electride_label
            else:
                # Single electride or no watershed: use Voronoi directly
                final_labels[voronoi_electride_mask] = n_atoms + i + 1

        return PartitionResult(
            labels=final_labels,
            n_atoms=n_atoms,
            n_electride_sites=n_electride,
            atom_labels=list(range(1, n_atoms + 1)),
            electride_labels=list(range(n_atoms + 1, n_atoms + n_electride + 1))
        )

    def partition_atoms_only(self, bonds: List[BondPair], use_elf_planes: bool = False) -> np.ndarray:
        """
        Partition only atomic regions (no electride sites)

        Parameters
        ----------
        bonds : List[BondPair]
            Analyzed bonds with ELF minima
        use_elf_planes : bool
            If True, adjust boundaries based on ELF minima planes.
            Default False uses simple Voronoi which gives stable results.

        Returns
        -------
        np.ndarray
            Labels array with atoms labeled 1 to n_atoms
        """
        if use_elf_planes:
            atom_labels = self.voronoi.partition_with_elf_planes(bonds)
        else:
            atom_labels = self.voronoi.simple_voronoi()
        return atom_labels + 1  # Convert from 0-indexed to 1-indexed

    def partition_with_bader(
        self,
        electride_sites: List[ElectrideSite],
        atom_labels: np.ndarray
    ) -> np.ndarray:
        """
        Add electride sites to existing atomic partition using Bader method

        Parameters
        ----------
        electride_sites : List[ElectrideSite]
            Electride sites to add
        atom_labels : np.ndarray
            Existing atomic labels (1 to n_atoms)

        Returns
        -------
        np.ndarray
            Updated labels including electride sites
        """
        n_atoms = self.elf_data.n_atoms
        ngrid = self.elf_data.ngrid

        # Create markers
        markers = np.zeros(ngrid, dtype=np.int32)

        # Set atom markers from current labels
        for i in range(1, n_atoms + 1):
            mask = atom_labels == i
            # Find centroid of each atomic region
            indices = np.argwhere(mask)
            if len(indices) > 0:
                centroid = indices.mean(axis=0).astype(int)
                centroid = tuple(centroid % ngrid)
                markers[centroid] = i

        # Set electride markers
        for i, site in enumerate(electride_sites):
            markers[site.grid_index] = n_atoms + i + 1

        # Run watershed
        labels = self.bader.watershed_partition(markers)

        return labels
