# elfcharge

**ELF-based electron counting for electrides**

A Python package for analyzing electron localization function (ELF) from VASP calculations to count electrons in electrides and other interstitial electron systems.

## Reference

This implementation is based on the BadELF algorithm:

> Weaver, J. R., et al. "Counting Electrons in Electrides"
> *J. Am. Chem. Soc.* **2023**, 145, 26472-26476.
> DOI: [10.1021/jacs.3c11019](https://doi.org/10.1021/jacs.3c11019)

## Features

- **ELF-based radius calculation**: Compute ELF radii from minima along bonds
- **Electride detection**: Find interstitial electron sites from ELF maxima
- **Voronoi partitioning**: Simple Voronoi partitioning of electron density
- **Charge integration**: Calculate electron counts and oxidation states
- **Structure export**: Export structures with electride sites to CIF/POSCAR

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/elfcharge.git
cd elfcharge

# Install with pip
pip install -e .
```

## Quick Start

```python
from elfcharge import run_badelf_analysis

# Run complete analysis
# Note: Use CHGCAR (valence electrons only), not CHGCAR_sum
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",  # Valence electrons only!
    oxidation_states={'Y': 3, 'C': -4},
    core_radii={'Y': 0.8, 'C': 0.3},
    elf_threshold=0.5,
    save_dir="./badelf_results"
)

# Access results
print(result.oxidation.species_avg_oxidation)
```

## Algorithm Overview

```
1. Load ELFCAR, CHGCAR
2. Find neighbor pairs (CrystalNN)
3. Analyze ELF along bonds → find ELF minima → compute elf_radii
4. Detect electride sites (ELF maxima outside atomic boundaries)
5. Partition space (Simple Voronoi)
6. Integrate charge in each region
7. Calculate oxidation states (ZVAL - electrons)
```

### Partitioning Strategy

| Boundary | Method |
|----------|--------|
| Atom-Atom | Simple Voronoi |
| Atom-Electride | Simple Voronoi |
| Electride regions | Merged into single region |

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `elfcar_path` | str | Path to ELFCAR file |
| `chgcar_path` | str | Path to CHGCAR file (valence electrons only) |
| `zval` | Dict[str, float] | Valence electrons per species (optional, uses DEFAULT_ZVAL) |
| `oxidation_states` | Dict[str, float] | Expected oxidation states for CrystalNN accuracy |
| `core_radii` | Dict[str, float] | Core radii to exclude when finding ELF minima |
| `elf_threshold` | float | Minimum ELF for electride sites (default: 0.5) |
| `apply_smooth` | bool | Apply smoothing to ELF (default: False) |
| `save_dir` | str | Output directory for plots and CIF |

### Parameter Guide

- **core_radii**: Used when finding ELF minima along bonds. Excludes atomic core region where ELF is low. Important for heavy elements (e.g., Y, La).
- **elf_threshold**: Minimum ELF value for electride site detection (default: 0.5)

## ZVAL Configuration

The package includes **DEFAULT_ZVAL** based on [Materials Project recommended POTCARs](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/pseudopotentials).

```python
from elfcharge import DEFAULT_ZVAL

# Check default value
print(DEFAULT_ZVAL['Y'])   # 11 (Y_sv)
print(DEFAULT_ZVAL['Mg'])  # 8 (Mg_pv)

# Override for specific POTCARs
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",
    zval={'Li': 1},  # Li instead of Li_sv
)
```

## Module Structure

```
elfcharge/
├── __init__.py      # Public exports
├── io.py            # read_elfcar, read_chgcar
├── analysis.py      # ELFAnalyzer, ELFRadii, BondPair, ElectrideSite
├── partition.py     # VoronoiPartitioner, BadELFPartitioner
├── integrate.py     # ChargeIntegrator, OxidationCalculator, DEFAULT_ZVAL
├── structure.py     # create_structure_with_electrides, BadELFResult
├── visualize.py     # Plotting functions (optional)
└── core.py          # run_badelf_analysis, BadELFAnalyzer
```

## Requirements

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.7
- pymatgen >= 2022.0.0
- matplotlib >= 3.5 (optional, for visualization)

## Limitations

1. **Grid size mismatch**: When ELFCAR and CHGCAR have different grid sizes, nearest-neighbor resampling is applied automatically.

2. **Electride region merging**: All detected electride sites are merged into a single region for charge integration.

## License

MIT License

## Author

Naoki Matsui

## Development

This repository was implemented with [Claude Opus 4.5](https://www.anthropic.com/claude) (Anthropic).

## Citation

```bibtex
@article{weaver2023counting,
  title={Counting Electrons in Electrides},
  author={Weaver, James R. and others},
  journal={J. Am. Chem. Soc.},
  volume={145},
  pages={26472--26476},
  year={2023},
  doi={10.1021/jacs.3c11019}
}
```
