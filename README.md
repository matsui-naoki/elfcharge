# elfcharge

**ELF-based electron counting for electrides**

A Python package for analyzing electron localization function (ELF) from VASP calculations to count electrons in electrides and other interstitial electron systems.

## Reference

This implementation is based on the BadELF algorithm:

> Weaver, J. R., et al. "Counting Electrons in Electrides"
> *J. Am. Chem. Soc.* **2023**, 145, 26472-26476.
> DOI: [10.1021/jacs.3c11019](https://doi.org/10.1021/jacs.3c11019)

## Features

- **ELF-based spatial partitioning**: Partition space using ELF minima along bonds
- **Electride detection**: Find interstitial electron sites from ELF maxima
- **Charge integration**: Calculate electron counts and oxidation states
- **Structure export**: Export structures with electride sites to CIF/POSCAR
- **Visualization**: Plot ELF profiles, slices, and electron distributions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/elfcharge.git
cd elfcharge

# Install with pip
pip install -e .

# With visualization support
pip install -e ".[visualization]"
```

## Quick Start

### Easy-Use Wrapper (Recommended)

```python
from elfcharge import run_badelf_analysis

# Run complete analysis with one function call
result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR_sum",
    zval={'Y': 11, 'C': 4, 'F': 7},
    oxidation_states={'Y': 3, 'C': -4, 'F': -1},
    core_radii={'Y': 0.8},
    elf_threshold=0.5,
    apply_smooth=True,
    save_dir="./badelf_results",
    save_plots=True,
    save_cif=True
)

# Access results
print(result.oxidation.species_avg_oxidation)  # Average oxidation by species
print(result.charges.atom_electrons)           # Electrons per atom
print(result.electride_sites)                  # Detected electride sites
```

### Step-by-Step API

```python
from elfcharge import read_elfcar, read_chgcar, ELFAnalyzer
from elfcharge import create_structure_with_electrides

# Load VASP output (with optional smoothing)
elf_data = read_elfcar("ELFCAR", smooth=True, smooth_size=3)
chg_data = read_chgcar("CHGCAR_sum")

# Initialize analyzer
analyzer = ELFAnalyzer(elf_data)

# Find neighbor pairs (with oxidation states for better accuracy)
bonds = analyzer.get_neighbor_pairs_pymatgen(
    oxidation_states={'Y': 3, 'C': -4, 'F': -1}
)

# Analyze ELF along bonds
for i, bond in enumerate(bonds):
    bonds[i] = analyzer.analyze_bond_elf(bond, core_radii={'Y': 0.8})

# Compute atom radii from bond analysis
atom_radii = analyzer.compute_atom_radii(bonds)

# Find electride sites (uses boundary_radius from atom_radii)
electride_sites = analyzer.find_interstitial_electrides(
    elf_threshold=0.5,
    atom_radii=atom_radii,
    use_boundary_radius=True
)

# Export structure with electride sites
structure = create_structure_with_electrides(elf_data, electride_sites)
structure.to(filename="structure_with_electrides.cif")
```

## Example

See `examples/example_elfcharge.ipynb` for a complete analysis workflow including:
- Loading VASP output files
- Finding electride sites
- Spatial partitioning
- Charge integration
- Oxidation state calculation
- Visualization

## Algorithm Overview

### Workflow

```
Input: ELFCAR, CHGCAR
           ↓
┌─────────────────────────────────┐
│ 1. Load Data                    │
│    - Lattice vectors            │
│    - Atomic positions           │
│    - ELF/charge density grids   │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 2. Find Neighbor Pairs          │
│    - CrystalNN for bonds        │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 3. Analyze Bond ELF             │
│    - Find ELF minima on bonds   │
│    - Compute ELF radii          │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 4. Detect Electride Sites       │
│    - Find ELF maxima outside    │
│      atomic boundaries          │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 5. Spatial Partitioning         │
│    - Atom-atom: Voronoi with    │
│      ELF minimum planes         │
│    - Atom-electride: Voronoi    │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ 6. Charge Integration           │
│    - Sum CHGCAR in each region  │
│    - Calculate oxidation states │
└─────────────────────────────────┘
           ↓
Output: Electron counts, oxidation states
```

### Partitioning Strategy

| Boundary Type | Method | Reason |
|--------------|--------|--------|
| Atom-Atom | Voronoi with ELF minimum planes | Maintains convex atomic regions |
| Atom-Electride | Simple Voronoi | Allows non-spherical shapes |

### Key Equations

**Charge Integration:**
```
N_electrons = Σ(CHGCAR[region]) / N_grid_total
```

**Oxidation State:**
```
Oxidation = ZVAL - N_electrons
```

**Minimum Image Distance:**
```python
diff_frac = r_frac - atom_frac
diff_frac = diff_frac - np.round(diff_frac)  # wrap to [-0.5, 0.5]
diff_cart = diff_frac @ lattice
distance = np.linalg.norm(diff_cart)
```

## ZVAL Configuration

For oxidation state calculations, specify valence electron counts from your POTCAR:

```python
ZVAL = {
    'Y': 11,   # PAW_PBE Y_sv
    'C': 4,    # PAW_PBE C
    'F': 7,    # PAW_PBE F
}
```

## Key Parameters

### run_badelf_analysis() Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `elfcar_path` | str | Path to ELFCAR file |
| `chgcar_path` | str | Path to CHGCAR file (CHGCAR_sum recommended) |
| `zval` | Dict[str, float] | Valence electrons per species (from POTCAR) |
| `oxidation_states` | Dict[str, float] | Expected oxidation states for CrystalNN |
| `core_radii` | Dict[str, float] | Core radii to exclude in ELF minimum search |
| `atom_cutoffs` | Dict[str, float] | Fallback cutoffs for electride detection |
| `elf_threshold` | float | Minimum ELF for electride sites (default: 0.5) |
| `apply_smooth` | bool | Apply smoothing to ELF (default: False) |
| `smooth_size` | int | Smoothing kernel size (default: 3) |
| `save_dir` | str | Output directory for plots and CIF |
| `save_plots` | bool | Generate visualization plots (default: True) |
| `save_cif` | bool | Export structure with electrides (default: True) |

### Grid Resolution

The package checks grid resolution against BadELF paper recommendations:
- **16 voxels/Å**: < 0.2% error (acceptable for most analyses)
- **40 voxels/Å**: Fully converged results

### Electride Detection
- `elf_threshold`: Minimum ELF value for electride sites (default: 0.5)
- `atom_radii`: Uses `boundary_radius` (max distance to ELF minimum) for interstitial classification
- `atom_cutoffs`: Fallback species-specific distance cutoffs

### Bond Analysis
- `core_radii`: Atomic core radii to exclude when finding ELF minima
  - Important for heavy elements (e.g., La) where ELF is low in the core
- `oxidation_states`: Improves CrystalNN neighbor detection accuracy

## Visualization

```python
from elfcharge.visualize import plot_elf_slice_with_partition

# Plot ELF slice with partition boundaries
fig = plot_elf_slice_with_partition(
    elf_data, labels,
    plane='xy', position=0.5,
    electride_sites=electride_sites,
    interpolate=True,           # High resolution
    interpolate_factor=3,
    coord_system='cartesian',   # or 'fractional'
)
fig.savefig('elf_partition.png', dpi=150)
```

## Module Structure

```
elfcharge/
├── __init__.py      # Package initialization
├── io.py            # VASP file I/O (ELFCAR, CHGCAR)
├── analysis.py      # ELF analysis (bonds, minima, radii)
├── partition.py     # Spatial partitioning (Voronoi)
├── integrate.py     # Charge integration
├── structure.py     # Structure output with electrides
├── visualize.py     # Visualization tools
└── core.py          # High-level API
```

## API Reference

### Data Classes

```python
@dataclass
class GridData:
    """Container for 3D grid data"""
    lattice: np.ndarray          # (3, 3) lattice vectors [Å]
    species: List[str]           # Element symbols
    num_atoms: List[int]         # Atom counts per species
    frac_coords: np.ndarray      # (N_atoms, 3) fractional coordinates
    grid: np.ndarray             # (NGX, NGY, NGZ) grid data
    ngrid: Tuple[int, int, int]  # Grid dimensions

@dataclass
class BondPair:
    """Bond information between atoms"""
    atom_i: int
    atom_j: int
    jimage: Tuple[int, int, int]  # Periodic image
    elf_minimum_frac: np.ndarray  # ELF minimum position
    elf_minimum_value: float
    distance_i: float  # Distance from atom i to minimum
    distance_j: float  # Distance from atom j to minimum

@dataclass
class ElectrideSite:
    """Electride site information"""
    frac_coord: np.ndarray
    cart_coord: np.ndarray
    elf_value: float
    grid_index: Tuple[int, int, int]
```

### Main Functions

```python
# High-level API (recommended)
run_badelf_analysis(
    elfcar_path, chgcar_path,
    zval=None, oxidation_states=None, core_radii=None,
    elf_threshold=0.5, apply_smooth=False, smooth_size=3,
    save_dir=None, save_plots=True, save_cif=True
) -> BadELFResult

# I/O
read_elfcar(filepath, smooth=False, smooth_size=3) -> GridData
read_chgcar(filepath) -> GridData
check_grid_resolution(data, min_voxels_per_angstrom=16.0) -> dict

# Analysis
ELFAnalyzer(elf_data)
  .get_neighbor_pairs_pymatgen(oxidation_states=None) -> List[BondPair]
  .analyze_bond_elf(bond, core_radii=None) -> BondPair
  .compute_atom_radii(bonds) -> Dict[int, AtomRadii]
  .find_interstitial_electrides(
      elf_threshold, atom_cutoffs=None,
      atom_radii=None, use_boundary_radius=True
  ) -> List[ElectrideSite]

# Structure output
create_structure_with_electrides(elf_data, electride_sites) -> Structure

# Visualization
plot_elf_slice_with_partition(elf_data, labels, plane, position, ...)
plot_elf_along_bond(elf_data, bond)
plot_radial_elf(elf_data, atom_index)
plot_electron_distribution(atom_electrons, species_list)
```

## Requirements

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.7
- pymatgen >= 2022.0.0
- matplotlib >= 3.5 (optional, for visualization)

## Known Limitations

1. **ELF plane adjustment**: The ELF minimum plane partitioning may be unstable for some systems. Simple Voronoi partitioning is used as fallback.

2. **Grid size mismatch**: When ELFCAR and CHGCAR have different grid sizes, nearest-neighbor resampling is applied automatically.

3. **Covalent materials**: Materials with strong covalent bonds may show less accurate electron distribution.

## License

MIT License

## Author

Naoki Matsui

## Development

This repository was implemented with [Claude Opus 4.5](https://www.anthropic.com/claude) (Anthropic).

## Citation

If you use this package, please cite the original BadELF paper:

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
