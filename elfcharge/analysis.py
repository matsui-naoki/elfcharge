"""
ELF analysis module for BadELF algorithm

This module provides functions for:
- Finding neighbor pairs using CrystalNN or distance-based methods
- Analyzing ELF along bond paths
- Finding ELF minima and radii
- Detecting electride sites
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import argrelmin, argrelmax
from scipy.ndimage import maximum_filter

from .io import GridData


@dataclass
class BondPair:
    """Information about a bond between two atoms"""
    atom_i: int                          # First atom index
    atom_j: int                          # Second atom index
    jimage: Tuple[int, int, int] = (0, 0, 0)  # Periodic image of atom j

    # Computed properties
    elf_minimum_frac: np.ndarray = None  # Fractional coords of ELF minimum
    elf_minimum_value: float = None      # ELF value at minimum
    distance_i: float = None             # Distance from atom i to minimum [Å]
    distance_j: float = None             # Distance from atom j to minimum [Å]
    bond_length: float = None            # Total bond length [Å]
    is_covalent: bool = False            # Whether bond is covalent


@dataclass
class ELFRadii:
    """ELF-derived radii for an atom (computed from bond analysis)"""
    atom_index: int = None
    species: str = None
    distances: List[float] = field(default_factory=list)
    elf_radius: float = None      # Minimum distance (Shannon-like)
    boundary_radius: float = None  # Maximum distance (for electride detection)


# Backwards compatibility alias (deprecated, will be removed in v0.4.0)
AtomRadii = ELFRadii


@dataclass
class ElectrideSite:
    """Information about a detected electride site"""
    frac_coord: np.ndarray    # Fractional coordinates
    cart_coord: np.ndarray    # Cartesian coordinates
    elf_value: float          # ELF value at this site
    grid_index: Tuple[int, int, int]  # Grid indices


class ELFAnalyzer:
    """
    Analyzer for ELF data to find bond minima and electride sites
    """

    def __init__(self, elf_data: GridData, structure_data: Optional[GridData] = None):
        """
        Initialize ELF analyzer

        Parameters
        ----------
        elf_data : GridData
            ELF grid data
        structure_data : GridData, optional
            Structure data (if different from elf_data)
        """
        self.elf_data = elf_data
        self.structure = structure_data if structure_data else elf_data

        # Create interpolator for ELF with periodic boundary conditions
        self._setup_interpolator()

    def _setup_interpolator(self):
        """Setup trilinear interpolator with periodic boundary handling"""
        # Pad the grid for periodic boundaries
        ngx, ngy, ngz = self.elf_data.ngrid
        grid = self.elf_data.grid

        # Create extended grid for periodic interpolation
        # We'll handle periodicity manually in interpolation
        x = np.linspace(0, 1, ngx, endpoint=False)
        y = np.linspace(0, 1, ngy, endpoint=False)
        z = np.linspace(0, 1, ngz, endpoint=False)

        self._interp_coords = (x, y, z)
        self._grid_for_interp = grid

    def interpolate_elf(self, frac_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate ELF at given fractional coordinates with periodic boundaries

        Parameters
        ----------
        frac_coords : np.ndarray
            Fractional coordinates, shape (..., 3)

        Returns
        -------
        np.ndarray
            Interpolated ELF values
        """
        # Wrap to [0, 1)
        frac_wrapped = frac_coords % 1.0

        # Trilinear interpolation with periodic boundaries
        ngx, ngy, ngz = self.elf_data.ngrid
        grid = self._grid_for_interp

        # Convert to grid coordinates
        gx = frac_wrapped[..., 0] * ngx
        gy = frac_wrapped[..., 1] * ngy
        gz = frac_wrapped[..., 2] * ngz

        # Get integer indices
        x0 = np.floor(gx).astype(int) % ngx
        y0 = np.floor(gy).astype(int) % ngy
        z0 = np.floor(gz).astype(int) % ngz
        x1 = (x0 + 1) % ngx
        y1 = (y0 + 1) % ngy
        z1 = (z0 + 1) % ngz

        # Get fractional parts
        xd = gx - np.floor(gx)
        yd = gy - np.floor(gy)
        zd = gz - np.floor(gz)

        # Trilinear interpolation
        c000 = grid[x0, y0, z0]
        c001 = grid[x0, y0, z1]
        c010 = grid[x0, y1, z0]
        c011 = grid[x0, y1, z1]
        c100 = grid[x1, y0, z0]
        c101 = grid[x1, y0, z1]
        c110 = grid[x1, y1, z0]
        c111 = grid[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd

    def get_neighbor_pairs(self, cutoff: float = 4.0) -> List[BondPair]:
        """
        Get neighbor pairs based on distance cutoff

        Parameters
        ----------
        cutoff : float
            Distance cutoff in Å

        Returns
        -------
        List[BondPair]
            List of bond pairs
        """
        lattice = self.structure.lattice
        frac_coords = self.structure.frac_coords
        n_atoms = self.structure.n_atoms

        pairs = []

        # Check periodic images
        images = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    images.append((i, j, k))

        for i in range(n_atoms):
            for j in range(i, n_atoms):
                for jimage in images:
                    if i == j and jimage == (0, 0, 0):
                        continue

                    # Calculate distance
                    frac_j_shifted = frac_coords[j] + np.array(jimage)
                    diff_frac = frac_j_shifted - frac_coords[i]
                    diff_cart = diff_frac @ lattice
                    dist = np.linalg.norm(diff_cart)

                    if dist < cutoff:
                        pairs.append(BondPair(
                            atom_i=i,
                            atom_j=j,
                            jimage=jimage,
                            bond_length=dist
                        ))

        return pairs

    def get_neighbor_pairs_pymatgen(
        self,
        oxidation_states: Optional[Dict[str, float]] = None
    ) -> List[BondPair]:
        """
        Get neighbor pairs using pymatgen's CrystalNN

        Parameters
        ----------
        oxidation_states : Dict[str, float], optional
            Oxidation states for each species (e.g., {'La': 3, 'Mg': 2, 'H': -1}).
            Following the BadELF paper, providing oxidation states improves
            the accuracy of CrystalNN neighbor detection.

        Returns
        -------
        List[BondPair]
            List of bond pairs
        """
        try:
            from pymatgen.core import Structure, Lattice, Species
            from pymatgen.analysis.local_env import CrystalNN
        except ImportError:
            print("pymatgen not available, falling back to distance-based neighbors")
            return self.get_neighbor_pairs()

        # Create pymatgen Structure
        lattice = Lattice(self.structure.lattice)
        species_list = self.structure.get_species_list()
        frac_coords = self.structure.frac_coords

        # If oxidation states provided, use Species objects with oxidation states
        # This improves CrystalNN accuracy as per the BadELF paper SI
        if oxidation_states is not None:
            species_with_oxi = []
            for sp in species_list:
                if sp in oxidation_states:
                    species_with_oxi.append(Species(sp, oxidation_states[sp]))
                else:
                    species_with_oxi.append(sp)
            struct = Structure(lattice, species_with_oxi, frac_coords)
        else:
            struct = Structure(lattice, species_list, frac_coords)

        # Use CrystalNN to find neighbors
        cnn = CrystalNN()
        pairs = []
        seen = set()

        for i in range(len(struct)):
            nn_info = cnn.get_nn_info(struct, i)
            for nn in nn_info:
                j = nn['site_index']
                jimage = tuple(nn['image'])

                # Avoid duplicates
                key = (min(i, j), max(i, j), jimage if i < j else tuple(-x for x in jimage))
                if key in seen:
                    continue
                seen.add(key)

                # Calculate distance
                site_i = struct[i]
                site_j = struct[j]
                jimage_arr = np.array(jimage)
                frac_j_shifted = site_j.frac_coords + jimage_arr
                diff = frac_j_shifted - site_i.frac_coords
                cart_diff = diff @ self.structure.lattice
                dist = np.linalg.norm(cart_diff)

                pairs.append(BondPair(
                    atom_i=i,
                    atom_j=j,
                    jimage=jimage,
                    bond_length=dist
                ))

        return pairs

    # Default core radii to exclude when searching for ELF bond minima
    # These values are based on typical atomic core sizes where ELF is low
    DEFAULT_CORE_RADII = {
        'La': 0.8,   # La has ELF max around 0.6 Å, exclude inner core
        'Mg': 0.5,   # Mg core
        'H': 0.0,    # H has no core
    }

    def analyze_bond_elf(
        self,
        bond: BondPair,
        n_points: int = 100,
        core_radii: Optional[Dict[str, float]] = None
    ) -> BondPair:
        """
        Analyze ELF along a bond and find the minimum

        Parameters
        ----------
        bond : BondPair
            Bond pair to analyze
        n_points : int
            Number of points along the bond path
        core_radii : dict, optional
            Species-specific core radii to exclude from minimum search.
            Points within this distance from atom centers are ignored.
            Default uses DEFAULT_CORE_RADII.

        Returns
        -------
        BondPair
            Updated bond pair with ELF minimum information
        """
        if core_radii is None:
            core_radii = self.DEFAULT_CORE_RADII

        frac_coords = self.structure.frac_coords
        lattice = self.structure.lattice
        species_list = self.structure.get_species_list()

        # Get positions
        pos_i = frac_coords[bond.atom_i]
        pos_j = frac_coords[bond.atom_j] + np.array(bond.jimage)

        sp_i = species_list[bond.atom_i]
        sp_j = species_list[bond.atom_j]

        # Get core radii for these species
        core_i = core_radii.get(sp_i, 0.5)  # Default 0.5 Å
        core_j = core_radii.get(sp_j, 0.5)

        # Generate path
        t = np.linspace(0, 1, n_points)
        path_frac = pos_i[None, :] + t[:, None] * (pos_j - pos_i)[None, :]

        # Interpolate ELF along path
        elf_values = self.interpolate_elf(path_frac)

        # Calculate distances along path from both atoms
        path_cart = path_frac @ lattice
        cart_i = pos_i @ lattice
        cart_j = pos_j @ lattice

        distances_from_i = np.linalg.norm(path_cart - cart_i, axis=1)
        distances_from_j = np.linalg.norm(path_cart - cart_j, axis=1)

        # Create mask for valid region (outside both atomic cores)
        valid_mask = (distances_from_i > core_i) & (distances_from_j > core_j)

        # Also apply the relative 20% cutoff as fallback
        start_idx = int(0.2 * n_points)
        end_idx = int(0.8 * n_points)
        relative_mask = np.zeros(n_points, dtype=bool)
        relative_mask[start_idx:end_idx] = True

        # Combine: must be outside cores AND within relative bounds
        search_mask = valid_mask & relative_mask

        if not np.any(search_mask):
            # Fallback: use relative bounds only if no valid region
            search_mask = relative_mask

        # Find local minima in the valid region
        valid_indices = np.where(search_mask)[0]
        if len(valid_indices) < 3:
            # Not enough points to find minimum
            bond.elf_minimum_frac = None
            return bond

        local_minima = []
        for i, idx in enumerate(valid_indices[1:-1], start=1):
            prev_idx = valid_indices[i-1]
            next_idx = valid_indices[i+1]
            if elf_values[idx] < elf_values[prev_idx] and elf_values[idx] < elf_values[next_idx]:
                local_minima.append(idx)

        # If no local minimum found, use the global minimum in valid region
        if len(local_minima) == 0:
            inner_idx = valid_indices[np.argmin(elf_values[valid_indices])]
        else:
            # Use the deepest local minimum
            inner_idx = local_minima[np.argmin(elf_values[local_minima])]

        # Refine minimum location using parabolic interpolation
        if 1 < inner_idx < len(elf_values) - 2:
            idx = inner_idx
            # Parabolic fit around minimum
            x = np.array([-1, 0, 1])
            y = elf_values[idx-1:idx+2]
            # Parabolic minimum: x_min = -b/(2a) for y = ax² + bx + c
            a = (y[0] - 2*y[1] + y[2]) / 2
            b = (y[2] - y[0]) / 2
            if abs(a) > 1e-10:
                x_min = -b / (2 * a)
                x_min = np.clip(x_min, -0.5, 0.5)
                t_refined = (idx + x_min) / (n_points - 1)
            else:
                t_refined = idx / (n_points - 1)
        else:
            t_refined = inner_idx / (n_points - 1)

        # Get minimum position
        min_frac = pos_i + t_refined * (pos_j - pos_i)
        min_elf = self.interpolate_elf(min_frac)

        # Calculate distances from atoms to minimum
        min_cart = min_frac @ lattice

        dist_i = np.linalg.norm(min_cart - cart_i)
        dist_j = np.linalg.norm(min_cart - cart_j)

        # Update bond
        bond.elf_minimum_frac = min_frac
        bond.elf_minimum_value = float(min_elf)
        bond.distance_i = float(dist_i)
        bond.distance_j = float(dist_j)

        # Check covalency
        bond.is_covalent = self._check_covalent(elf_values, bond)

        return bond

    def _check_covalent(self, elf_values: np.ndarray, bond: BondPair) -> bool:
        """
        Check if a bond is covalent based on ELF profile

        Criteria from paper SI:
        1. Same element bonds are covalent
        2. ELF maximum closer to bond center than minimum
        3. ELF minimum < 0.5 × nearby maximum
        """
        species_list = self.structure.get_species_list()
        sp_i = species_list[bond.atom_i]
        sp_j = species_list[bond.atom_j]

        # Same element = covalent
        if sp_i == sp_j:
            return True

        # Find local extrema
        minima = argrelmin(elf_values)[0]
        maxima = argrelmax(elf_values)[0]

        if len(minima) == 0 or len(maxima) == 0:
            return False

        bond_center = len(elf_values) // 2
        min_idx = minima[np.argmin(elf_values[minima])]

        # Check if any maximum is closer to center than minimum
        for max_idx in maxima:
            if abs(max_idx - bond_center) < abs(min_idx - bond_center):
                return True

        # Check if minimum < 0.5 × maximum
        min_elf = elf_values[min_idx]
        max_elf = max(elf_values[maxima])
        if min_elf < 0.5 * max_elf:
            return True

        return False

    def compute_elf_radii(self, bonds: List[BondPair]) -> Dict[int, ELFRadii]:
        """
        Compute ELF radii for each atom from bond analysis.

        For each atom, the ELF radius is the minimum distance to an ELF minimum
        (Shannon-like radius), and the boundary radius is the maximum distance
        (used for electride detection).

        Parameters
        ----------
        bonds : List[BondPair]
            List of analyzed bonds

        Returns
        -------
        Dict[int, ELFRadii]
            Dictionary mapping atom index to ELF radii information
        """
        species_list = self.structure.get_species_list()
        radii = {i: ELFRadii(atom_index=i, species=species_list[i])
                 for i in range(self.structure.n_atoms)}

        for bond in bonds:
            if bond.distance_i is not None:
                radii[bond.atom_i].distances.append(bond.distance_i)
            if bond.distance_j is not None:
                radii[bond.atom_j].distances.append(bond.distance_j)

        for i, r in radii.items():
            if r.distances:
                r.elf_radius = min(r.distances)
                r.boundary_radius = max(r.distances)

        return radii

    # Backwards compatibility alias
    def compute_atom_radii(self, bonds: List[BondPair]) -> Dict[int, ELFRadii]:
        """Deprecated: Use compute_elf_radii() instead."""
        return self.compute_elf_radii(bonds)

    def find_electride_sites(
        self,
        atom_boundaries: Dict[int, float],
        elf_threshold: float = 0.5,
        neighborhood_size: int = 3
    ) -> List[ElectrideSite]:
        """
        Find potential electride sites (ELF maxima outside atomic boundaries)

        Parameters
        ----------
        atom_boundaries : Dict[int, float]
            Boundary radius for each atom
        elf_threshold : float
            Minimum ELF value for electride sites
        neighborhood_size : int
            Size of neighborhood for local maximum detection

        Returns
        -------
        List[ElectrideSite]
            List of detected electride sites
        """
        grid = self.elf_data.grid
        lattice = self.structure.lattice
        frac_coords = self.structure.frac_coords

        # Find local maxima with periodic boundary conditions
        # Pad grid for periodic maximum filter
        padded = np.pad(grid, neighborhood_size, mode='wrap')
        local_max_padded = maximum_filter(padded, size=2*neighborhood_size+1)
        local_max = local_max_padded[
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size
        ]

        is_local_max = (grid == local_max) & (grid >= elf_threshold)
        max_indices = np.argwhere(is_local_max)

        electride_sites = []
        ngrid = np.array(self.elf_data.ngrid)

        for idx in max_indices:
            # Convert to fractional coordinates
            frac_coord = idx / ngrid
            cart_coord = frac_coord @ lattice
            elf_value = grid[tuple(idx)]

            # Check if outside all atomic boundaries
            is_inside_any = False
            for i in range(self.structure.n_atoms):
                atom_frac = frac_coords[i]

                # Check minimum image distance
                diff = frac_coord - atom_frac
                diff = diff - np.round(diff)  # Minimum image
                diff_cart = diff @ lattice
                dist = np.linalg.norm(diff_cart)

                boundary = atom_boundaries.get(i, 2.0)  # Default 2 Å
                if dist < boundary:
                    is_inside_any = True
                    break

            if not is_inside_any:
                electride_sites.append(ElectrideSite(
                    frac_coord=frac_coord,
                    cart_coord=cart_coord,
                    elf_value=elf_value,
                    grid_index=tuple(idx)
                ))

        return electride_sites

    def find_all_elf_maxima(
        self,
        elf_threshold: float = 0.3,
        neighborhood_size: int = 3
    ) -> List[ElectrideSite]:
        """
        Find all ELF maxima above threshold

        Parameters
        ----------
        elf_threshold : float
            Minimum ELF value
        neighborhood_size : int
            Size of neighborhood for local maximum detection

        Returns
        -------
        List[ElectrideSite]
            List of all ELF maxima
        """
        grid = self.elf_data.grid
        lattice = self.structure.lattice
        ngrid = np.array(self.elf_data.ngrid)

        # Find local maxima with periodic boundary conditions
        padded = np.pad(grid, neighborhood_size, mode='wrap')
        local_max_padded = maximum_filter(padded, size=2*neighborhood_size+1)
        local_max = local_max_padded[
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size
        ]

        is_local_max = (grid == local_max) & (grid >= elf_threshold)
        max_indices = np.argwhere(is_local_max)

        maxima = []
        for idx in max_indices:
            frac_coord = idx / ngrid
            cart_coord = frac_coord @ lattice
            elf_value = grid[tuple(idx)]

            maxima.append(ElectrideSite(
                frac_coord=frac_coord,
                cart_coord=cart_coord,
                elf_value=elf_value,
                grid_index=tuple(idx)
            ))

        return maxima

    def find_interstitial_electrides(
        self,
        elf_threshold: float = 0.5,
        elf_radii: Optional[Dict[int, ELFRadii]] = None,
        use_boundary_radius: bool = True,
        default_cutoff: float = 1.5,
        neighborhood_size: int = 3,
        atom_radii: Optional[Dict[int, ELFRadii]] = None  # Deprecated alias
    ) -> List[ElectrideSite]:
        """
        Find electride sites as ELF maxima at interstitial (non-atomic) positions

        This method identifies electride sites by:
        1. Finding all local maxima in the ELF
        2. Excluding maxima near atomic nuclei (these are core/valence electrons)
        3. Returning only interstitial maxima (electride electrons)

        The atomic boundary for electride detection uses the "boundary radius"
        (largest ELF minimum distance from bond analysis).

        Parameters
        ----------
        elf_threshold : float
            Minimum ELF value for electride sites (default: 0.5)
        elf_radii : Dict[int, ELFRadii], optional
            Per-atom radii computed from bond analysis (compute_elf_radii).
            Uses boundary_radius (max ELF minimum distance) for detection.
        use_boundary_radius : bool
            If True and elf_radii provided, use boundary_radius.
        default_cutoff : float
            Fallback cutoff when elf_radii not available (default: 1.5 Å)
        neighborhood_size : int
            Size of neighborhood for local maximum detection
        atom_radii : Dict[int, ELFRadii], optional
            Deprecated: Use elf_radii instead.

        Returns
        -------
        List[ElectrideSite]
            List of interstitial electride sites
        """
        # Handle deprecated parameter
        if atom_radii is not None and elf_radii is None:
            elf_radii = atom_radii
        grid = self.elf_data.grid
        lattice = self.structure.lattice
        frac_coords = self.structure.frac_coords
        ngrid = np.array(self.elf_data.ngrid)

        # Find all local maxima with periodic boundary conditions
        padded = np.pad(grid, neighborhood_size, mode='wrap')
        local_max_padded = maximum_filter(padded, size=2*neighborhood_size+1)
        local_max = local_max_padded[
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size,
            neighborhood_size:-neighborhood_size
        ]

        is_local_max = (grid == local_max) & (grid >= elf_threshold)
        max_indices = np.argwhere(is_local_max)

        electride_sites = []

        for idx in max_indices:
            frac_coord = idx / ngrid
            cart_coord = frac_coord @ lattice
            elf_value = grid[tuple(idx)]

            # Check if this maximum is too close to any atom
            is_near_atom = False
            for i in range(self.structure.n_atoms):
                atom_frac = frac_coords[i]
                diff = frac_coord - atom_frac
                diff = diff - np.round(diff)  # Minimum image
                diff_cart = diff @ lattice
                dist = np.linalg.norm(diff_cart)

                # Determine cutoff for this atom from elf_radii
                if use_boundary_radius and elf_radii is not None and i in elf_radii:
                    cutoff = elf_radii[i].boundary_radius
                    if cutoff is None:
                        cutoff = default_cutoff
                else:
                    cutoff = default_cutoff

                if dist < cutoff:
                    is_near_atom = True
                    break

            # Only keep interstitial maxima (far from all atoms)
            if not is_near_atom:
                electride_sites.append(ElectrideSite(
                    frac_coord=frac_coord,
                    cart_coord=cart_coord,
                    elf_value=elf_value,
                    grid_index=tuple(idx)
                ))

        return electride_sites
