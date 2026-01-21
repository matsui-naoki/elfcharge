"""
I/O module for reading VASP output files (ELFCAR, CHGCAR)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class GridData:
    """Container for 3D grid data from VASP files"""
    lattice: np.ndarray          # (3, 3) lattice vectors [Å]
    species: List[str]           # element symbols
    num_atoms: List[int]         # number of atoms per species
    frac_coords: np.ndarray      # (N_atoms, 3) fractional coordinates
    grid: np.ndarray             # (NGX, NGY, NGZ) grid data
    ngrid: Tuple[int, int, int]  # grid dimensions

    @property
    def volume(self) -> float:
        """Unit cell volume in Å³"""
        return np.abs(np.linalg.det(self.lattice))

    @property
    def n_atoms(self) -> int:
        """Total number of atoms"""
        return sum(self.num_atoms)

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        """Convert fractional to Cartesian coordinates"""
        return frac @ self.lattice

    def cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian to fractional coordinates"""
        return cart @ np.linalg.inv(self.lattice)

    def get_cart_coords(self) -> np.ndarray:
        """Get Cartesian coordinates of all atoms"""
        return self.frac_to_cart(self.frac_coords)

    def get_species_list(self) -> List[str]:
        """Get full list of species for each atom"""
        species_list = []
        for sp, n in zip(self.species, self.num_atoms):
            species_list.extend([sp] * n)
        return species_list


def read_vasp_grid(filepath: str) -> GridData:
    """
    Read VASP grid file (ELFCAR, CHGCAR, LOCPOT, etc.)

    Parameters
    ----------
    filepath : str
        Path to the VASP grid file

    Returns
    -------
    GridData
        Parsed grid data including structure and volumetric data
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Line 1: System name (ignored)
    # Line 2: Scale factor
    scale = float(lines[1].strip())

    # Lines 3-5: Lattice vectors
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    # Line 6: Species names (VASP 5 format)
    species_line = lines[5].split()
    if species_line[0].isalpha():
        species = species_line
        num_atoms_line = 6
    else:
        # VASP 4 format - no species names
        species = [f"Type{i+1}" for i in range(len(species_line))]
        num_atoms_line = 5

    # Line 7: Number of atoms per species
    num_atoms = [int(x) for x in lines[num_atoms_line].split()]
    total_atoms = sum(num_atoms)

    # Line 8: Selective dynamics or Direct/Cartesian
    coord_line = num_atoms_line + 1
    if lines[coord_line].strip().lower().startswith('s'):
        coord_line += 1

    # Check if Direct or Cartesian
    is_direct = lines[coord_line].strip().lower().startswith('d')
    coord_line += 1

    # Read atomic positions
    frac_coords = np.zeros((total_atoms, 3))
    for i in range(total_atoms):
        parts = lines[coord_line + i].split()
        frac_coords[i] = [float(parts[j]) for j in range(3)]

    # Convert Cartesian to fractional if needed
    if not is_direct:
        inv_lattice = np.linalg.inv(lattice)
        frac_coords = frac_coords @ inv_lattice

    # Find grid dimensions line (after blank line)
    grid_line = coord_line + total_atoms
    while grid_line < len(lines) and lines[grid_line].strip() == '':
        grid_line += 1

    # Read grid dimensions
    ngrid = tuple(int(x) for x in lines[grid_line].split())
    grid_line += 1

    # Read grid data (stop at "augmentation" line if present)
    data_lines = []
    for line in lines[grid_line:]:
        if 'augmentation' in line.lower():
            break
        data_lines.append(line)
    data_text = ' '.join(data_lines)

    # Handle Fortran-style scientific notation (e.g., 0.123E-01 or 0.123-01)
    data_text = data_text.replace('E', 'e').replace('D', 'e')
    # Handle cases like "0.123-01" (missing E)
    import re
    data_text = re.sub(r'(\d)(-\d{2})(\s|$)', r'\1e\2\3', data_text)
    data_text = re.sub(r'(\d)(\+\d{2})(\s|$)', r'\1e\2\3', data_text)

    data = np.array([float(x) for x in data_text.split()])

    # Check if we have the right amount of data
    expected_size = ngrid[0] * ngrid[1] * ngrid[2]
    if len(data) < expected_size:
        raise ValueError(f"Not enough grid data. Expected {expected_size}, got {len(data)}")

    # Reshape grid data
    # VASP stores data in Fortran order: x varies fastest, then y, then z
    grid = data[:expected_size].reshape(ngrid[2], ngrid[1], ngrid[0])
    grid = np.transpose(grid, (2, 1, 0))  # Convert to (NGX, NGY, NGZ) with C order

    return GridData(
        lattice=lattice,
        species=species,
        num_atoms=num_atoms,
        frac_coords=frac_coords,
        grid=grid,
        ngrid=ngrid
    )


def check_grid_resolution(data: GridData, min_voxels_per_angstrom: float = 16.0) -> dict:
    """
    Check grid resolution against recommended values from BadELF paper.

    The paper recommends 40 voxels/Å for converged results, but 16 voxels/Å
    gives < 0.2% error which is acceptable for most purposes.

    Parameters
    ----------
    data : GridData
        Grid data to check
    min_voxels_per_angstrom : float
        Minimum recommended linear voxel density. Default: 16.0 (< 0.2% error)
        For publication-quality results, use 40.0.

    Returns
    -------
    dict
        Resolution info with keys: 'voxels_per_angstrom', 'is_sufficient', 'message'
    """
    # Calculate linear voxel densities along each axis
    lattice_lengths = np.linalg.norm(data.lattice, axis=1)
    voxels_per_angstrom = np.array(data.ngrid) / lattice_lengths

    avg_resolution = np.mean(voxels_per_angstrom)
    min_resolution = np.min(voxels_per_angstrom)

    is_sufficient = min_resolution >= min_voxels_per_angstrom

    if is_sufficient:
        message = f"Grid resolution OK: {avg_resolution:.1f} voxels/Å (min: {min_resolution:.1f})"
    else:
        message = (f"Warning: Grid resolution may be too low: {avg_resolution:.1f} voxels/Å "
                   f"(min: {min_resolution:.1f}). Recommended: >= {min_voxels_per_angstrom} voxels/Å. "
                   f"Consider increasing NGX/NGY/NGZ in VASP calculation.")

    return {
        'voxels_per_angstrom': voxels_per_angstrom,
        'average': avg_resolution,
        'minimum': min_resolution,
        'is_sufficient': is_sufficient,
        'message': message
    }


def read_elfcar(
    filepath: str,
    smooth: bool = False,
    smooth_size: int = 3,
    check_resolution: bool = True
) -> GridData:
    """
    Read ELFCAR file from VASP

    Parameters
    ----------
    filepath : str
        Path to ELFCAR file
    smooth : bool
        If True, apply uniform filter smoothing to the ELF data.
        This helps reduce noise and creates smoother partition boundaries
        when visualizing. Default: False
    smooth_size : int
        Size of the uniform filter kernel. Larger values = more smoothing.
        Default: 3 (3x3x3 kernel)
    check_resolution : bool
        If True, check and print grid resolution info. Default: True

    Returns
    -------
    GridData
        ELF data with values in [0, 1]
    """
    data = read_vasp_grid(filepath)

    # Check grid resolution
    if check_resolution:
        res_info = check_grid_resolution(data)
        if not res_info['is_sufficient']:
            print(res_info['message'])

    # Apply smoothing if requested
    if smooth:
        from scipy.ndimage import uniform_filter
        # Use wrap mode for periodic boundaries
        data.grid = uniform_filter(data.grid, size=smooth_size, mode='wrap')

    # Verify ELF values are in expected range
    if data.grid.min() < -0.01 or data.grid.max() > 1.01:
        print(f"Warning: ELF values outside [0,1] range: [{data.grid.min():.3f}, {data.grid.max():.3f}]")

    return data


def read_chgcar(filepath: str) -> GridData:
    """
    Read CHGCAR file from VASP

    Parameters
    ----------
    filepath : str
        Path to CHGCAR file

    Returns
    -------
    GridData
        Charge density data (values are ρ × V_cell)

    Note
    ----
    CHGCAR stores electron density × cell volume.
    To get electrons/Å³, divide by volume.
    Total electrons = sum(grid) / n_grid_points
    """
    return read_vasp_grid(filepath)


def interpolate_grid(
    data: GridData,
    target_ngrid: Tuple[int, int, int],
    method: str = 'linear',
    clip_range: Optional[Tuple[float, float]] = None
) -> GridData:
    """
    Interpolate grid data to a different resolution.

    This is useful for:
    - Matching grid sizes between ELFCAR and CHGCAR
    - Upsampling for higher resolution analysis
    - Downsampling for faster computation

    Parameters
    ----------
    data : GridData
        Input grid data
    target_ngrid : Tuple[int, int, int]
        Target grid dimensions (NGX, NGY, NGZ)
    method : str
        Interpolation method. Options:
        - 'linear': Trilinear interpolation (default, fast)
        - 'cubic': Tricubic spline interpolation (smoother)
        - 'nearest': Nearest neighbor (fastest, no smoothing)
    clip_range : Tuple[float, float], optional
        If provided, clip interpolated values to this range.
        Useful for ELF data: clip_range=(0.0, 1.0)

    Returns
    -------
    GridData
        New GridData with interpolated grid at target resolution

    Examples
    --------
    >>> elf = read_elfcar("ELFCAR")
    >>> chg = read_chgcar("CHGCAR")
    >>> # Match CHGCAR grid to ELFCAR grid
    >>> chg_interp = interpolate_grid(chg, elf.ngrid)
    """
    from scipy.interpolate import RegularGridInterpolator

    # Original grid coordinates (fractional: 0 to 1)
    x_orig = np.linspace(0, 1, data.ngrid[0], endpoint=False)
    y_orig = np.linspace(0, 1, data.ngrid[1], endpoint=False)
    z_orig = np.linspace(0, 1, data.ngrid[2], endpoint=False)

    # Pad for periodic boundary conditions
    # Add one extra point at the end that wraps around
    x_padded = np.append(x_orig, 1.0)
    y_padded = np.append(y_orig, 1.0)
    z_padded = np.append(z_orig, 1.0)

    # Pad grid data for periodicity
    grid_padded = np.zeros((data.ngrid[0] + 1, data.ngrid[1] + 1, data.ngrid[2] + 1))
    grid_padded[:-1, :-1, :-1] = data.grid
    grid_padded[-1, :-1, :-1] = data.grid[0, :, :]  # x periodicity
    grid_padded[:-1, -1, :-1] = data.grid[:, 0, :]  # y periodicity
    grid_padded[:-1, :-1, -1] = data.grid[:, :, 0]  # z periodicity
    grid_padded[-1, -1, :-1] = data.grid[0, 0, :]   # xy edge
    grid_padded[-1, :-1, -1] = data.grid[0, :, 0]   # xz edge
    grid_padded[:-1, -1, -1] = data.grid[:, 0, 0]   # yz edge
    grid_padded[-1, -1, -1] = data.grid[0, 0, 0]    # corner

    # Create interpolator
    if method == 'nearest':
        interp_method = 'nearest'
    elif method == 'cubic':
        interp_method = 'cubic'
    else:
        interp_method = 'linear'

    interpolator = RegularGridInterpolator(
        (x_padded, y_padded, z_padded),
        grid_padded,
        method=interp_method,
        bounds_error=False,
        fill_value=None
    )

    # Target grid coordinates
    x_new = np.linspace(0, 1, target_ngrid[0], endpoint=False)
    y_new = np.linspace(0, 1, target_ngrid[1], endpoint=False)
    z_new = np.linspace(0, 1, target_ngrid[2], endpoint=False)

    # Create meshgrid for interpolation
    xx, yy, zz = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Interpolate
    new_grid = interpolator(points).reshape(target_ngrid)

    # Clip values if requested
    if clip_range is not None:
        new_grid = np.clip(new_grid, clip_range[0], clip_range[1])

    return GridData(
        lattice=data.lattice.copy(),
        species=data.species.copy(),
        num_atoms=data.num_atoms.copy(),
        frac_coords=data.frac_coords.copy(),
        grid=new_grid,
        ngrid=target_ngrid
    )


def write_chgcar(filepath: str, data: GridData, comment: str = "BadELF output"):
    """
    Write data in CHGCAR format

    Parameters
    ----------
    filepath : str
        Output file path
    data : GridData
        Grid data to write
    comment : str
        Comment line at the beginning of file
    """
    filepath = Path(filepath)

    with open(filepath, 'w') as f:
        # Comment line
        f.write(f"{comment}\n")

        # Scale factor
        f.write("   1.00000000000000\n")

        # Lattice vectors
        for i in range(3):
            f.write(f"  {data.lattice[i, 0]:12.6f}  {data.lattice[i, 1]:12.6f}  {data.lattice[i, 2]:12.6f}\n")

        # Species names
        f.write("   " + "   ".join(data.species) + "\n")

        # Number of atoms
        f.write("   " + "   ".join(str(n) for n in data.num_atoms) + "\n")

        # Direct coordinates
        f.write("Direct\n")
        for i in range(data.n_atoms):
            f.write(f"  {data.frac_coords[i, 0]:10.6f}  {data.frac_coords[i, 1]:10.6f}  {data.frac_coords[i, 2]:10.6f}\n")

        # Blank line
        f.write("\n")

        # Grid dimensions
        f.write(f"  {data.ngrid[0]}  {data.ngrid[1]}  {data.ngrid[2]}\n")

        # Grid data (5 values per line, Fortran order)
        grid_fortran = np.transpose(data.grid, (2, 1, 0)).flatten()
        for i in range(0, len(grid_fortran), 5):
            chunk = grid_fortran[i:i+5]
            f.write(" " + " ".join(f"{v:17.11E}" for v in chunk) + "\n")


if __name__ == "__main__":
    # Test reading
    import sys
    if len(sys.argv) > 1:
        data = read_vasp_grid(sys.argv[1])
        print(f"Lattice:\n{data.lattice}")
        print(f"Species: {data.species}")
        print(f"Num atoms: {data.num_atoms}")
        print(f"Grid shape: {data.ngrid}")
        print(f"Grid min/max: {data.grid.min():.4f} / {data.grid.max():.4f}")
        print(f"Volume: {data.volume:.4f} Å³")
