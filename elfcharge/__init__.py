"""
elfcharge - ELF-based electron counting for electrides

This package implements the BadELF algorithm for counting electrons
in electrides and other interstitial electron systems.

Reference:
    Weaver et al., "Counting Electrons in Electrides"
    J. Am. Chem. Soc. 2023, 145, 26472-26476

Main features:
- Voronoi-like partitioning of atoms based on ELF minima
- Detection of electride sites from interstitial ELF maxima
- Bader-like charge integration for atoms and electride sites
- Export structures with electride sites to CIF/POSCAR

Example usage:
    from elfcharge import read_elfcar, read_chgcar, ELFAnalyzer
    from elfcharge import create_structure_with_electrides

    # Load VASP data
    elf_data = read_elfcar("ELFCAR")
    chg_data = read_chgcar("CHGCAR")

    # Analyze ELF to find electride sites
    analyzer = ELFAnalyzer(elf_data)
    electride_sites = analyzer.find_interstitial_electrides(
        elf_threshold=0.5,
        atom_cutoffs={'La': 1.0, 'Mg': 1.0, 'H': 0.5}
    )

    # Export structure with electride sites as dummy atom "X"
    structure = create_structure_with_electrides(elf_data, electride_sites)
    structure.to(filename="structure_with_electrides.cif")
"""

from importlib.metadata import version, PackageNotFoundError

from .io import read_elfcar, read_chgcar, GridData, check_grid_resolution
from .analysis import ELFAnalyzer, ElectrideSite, BondPair, AtomRadii
from .partition import VoronoiPartitioner
from .integrate import ATOMIC_NUMBERS
from .structure import (
    create_structure_with_electrides,
    write_structure_cif,
    write_structure_poscar,
    BadELFResult
)
from .core import run_badelf_analysis

# Visualization (optional - requires matplotlib)
try:
    from .visualize import (
        plot_elf_along_bond,
        plot_elf_profiles_by_type,
        plot_radial_elf,
        plot_electron_distribution,
        plot_electride_elf_histogram,
        plot_elf_slice,
        plot_elf_slice_with_partition
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Get version from package metadata (set in pyproject.toml)
try:
    __version__ = version("elfcharge")
except PackageNotFoundError:
    __version__ = "0.2.0"  # Fallback for development

__author__ = "Naoki Matsui"

__all__ = [
    # Version
    "__version__",
    # I/O
    "read_elfcar",
    "read_chgcar",
    "GridData",
    # Analysis
    "ELFAnalyzer",
    "ElectrideSite",
    "BondPair",
    # Partitioning
    "VoronoiPartitioner",
    # Integration
    "ATOMIC_NUMBERS",
    # Structure output
    "create_structure_with_electrides",
    "write_structure_cif",
    "write_structure_poscar",
    "BadELFResult",
    # ZVAL
    "ZVAL",
]

# ZVAL (valence electrons) - common POTCAR values
ZVAL = {
    'H': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 9, 'Ca': 10, 'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 12, 'Mn': 13,
    'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
    'Ga': 13, 'Ge': 14, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
    'Rb': 9, 'Sr': 10, 'Y': 11, 'Zr': 12, 'Nb': 13, 'Mo': 14,
    'La': 11, 'Ce': 12, 'Pr': 13, 'Nd': 14,
    # Mg_pv uses 8 valence electrons (2s²2p⁶3s²)
    'Mg_pv': 8,
}
