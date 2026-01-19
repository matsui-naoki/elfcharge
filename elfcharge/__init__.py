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
    from elfcharge import run_badelf_analysis

    # Run complete analysis (recommended)
    result = run_badelf_analysis(
        elfcar_path="ELFCAR",
        chgcar_path="CHGCAR",
        oxidation_states={'Y': 3, 'C': -4},
        core_radii={'Y': 0.8, 'C': 0.3},
        elf_threshold=0.5
    )

    # Access results
    print(result.oxidation.species_avg_oxidation)
"""

from importlib.metadata import version, PackageNotFoundError

from .io import read_elfcar, read_chgcar, GridData, check_grid_resolution
from .analysis import ELFAnalyzer, ElectrideSite, BondPair, ELFRadii, AtomRadii  # AtomRadii is deprecated alias
from .partition import VoronoiPartitioner
from .integrate import ATOMIC_NUMBERS, DEFAULT_ZVAL
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
    __version__ = "0.3.0"  # Fallback for development

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
    "ELFRadii",
    "AtomRadii",  # Deprecated alias for ELFRadii
    # Partitioning
    "VoronoiPartitioner",
    # Integration
    "ATOMIC_NUMBERS",
    "DEFAULT_ZVAL",
    # Structure output
    "create_structure_with_electrides",
    "write_structure_cif",
    "write_structure_poscar",
    "BadELFResult",
    # High-level API
    "run_badelf_analysis",
]
