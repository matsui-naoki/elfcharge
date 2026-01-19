# SPEC.md - elfcharge Package Specification

**Version**: 0.3.0
**Last Updated**: 2026-01-20

This document is the **Single Source of Truth** for the elfcharge package.
All code, documentation, and examples must conform to this specification.

---

## 1. Package Overview

**Purpose**: Analyze ELF (Electron Localization Function) from VASP to count electrons in electrides.

**Reference**: Weaver et al., "Counting Electrons in Electrides", J. Am. Chem. Soc. 2023, 145, 26472-26476

---

## 2. Core Concepts

### 2.1 Input Files
| File | Description | Required |
|------|-------------|----------|
| ELFCAR | ELF data from VASP | Yes |
| CHGCAR | Charge density (valence electrons only) | Yes |

**Important**: Use CHGCAR (valence only), NOT CHGCAR_sum (valence + core).

### 2.2 Key Parameters

| Parameter | Type | Description | User-specified |
|-----------|------|-------------|----------------|
| `core_radii` | Dict[str, float] | Atomic core radii to exclude when finding ELF minima along bonds. Important for heavy elements (Y, La, etc.) where ELF is low in core region. | Yes |
| `elf_threshold` | float | Minimum ELF value for electride site detection. Default: 0.5 | Yes |
| `oxidation_states` | Dict[str, float] | Expected oxidation states for CrystalNN accuracy | Yes (optional) |

### 2.3 Internal Data Structures (not user-specified)

| Name | Type | Description |
|------|------|-------------|
| `elf_radii` | Dict[int, ELFRadii] | Per-atom ELF radii computed from bond analysis. Contains `elf_radius` (min distance to ELF minimum) and `boundary_radius` (max distance). |
| `bonds` | List[BondPair] | Analyzed bonds with ELF minima positions |
| `electride_sites` | List[ElectrideSite] | Detected electride sites |

### 2.4 Algorithm Flow

```
1. Load ELFCAR, CHGCAR
2. Find neighbor pairs (CrystalNN)
3. Analyze ELF along bonds → find ELF minima → compute elf_radii
4. Detect electride sites (ELF maxima outside atomic boundaries)
5. Partition space (Simple Voronoi)
6. Integrate charge in each region
7. Calculate oxidation states (ZVAL - electrons)
```

### 2.5 Partitioning Strategy

| Boundary | Method |
|----------|--------|
| Atom-Atom | Simple Voronoi |
| Atom-Electride | Simple Voronoi |
| Electride regions | Merged into single region |

---

## 3. Public API

### 3.1 Main Entry Point

```python
from elfcharge import run_badelf_analysis

result = run_badelf_analysis(
    elfcar_path: str,           # Path to ELFCAR
    chgcar_path: str,           # Path to CHGCAR (valence only!)
    zval: Dict[str, float] = None,           # Override DEFAULT_ZVAL
    oxidation_states: Dict[str, float] = None,  # For CrystalNN
    core_radii: Dict[str, float] = None,     # Core exclusion radii
    elf_threshold: float = 0.5,              # Electride detection threshold
    apply_smooth: bool = False,              # Smooth ELF data
    smooth_size: int = 3,                    # Smoothing kernel size
    save_dir: str = None,                    # Output directory
    save_plots: bool = True,                 # Generate plots
    save_cif: bool = True,                   # Export CIF with electrides
    verbose: bool = True                     # Print progress
) -> BadELFResult
```

### 3.2 Return Type

```python
@dataclass
class BadELFResult:
    elf_data: GridData
    chg_data: GridData
    bonds: List[BondPair]
    elf_radii: Dict[int, ELFRadii]  # Renamed from atom_radii
    electride_sites: List[ElectrideSite]
    partition: PartitionResult
    charges: ChargeResult
    oxidation: OxidationResult
    total_electrons: float
    species_list: List[str]
```

### 3.3 Data Classes

```python
@dataclass
class GridData:
    lattice: np.ndarray          # (3, 3) lattice vectors [Å]
    species: List[str]           # Element symbols
    num_atoms: List[int]         # Atom counts per species
    frac_coords: np.ndarray      # (N, 3) fractional coordinates
    grid: np.ndarray             # (NGX, NGY, NGZ) grid data
    ngrid: Tuple[int, int, int]  # Grid dimensions

@dataclass
class ELFRadii:  # Renamed from AtomRadii
    atom_index: int
    species: str
    elf_radius: float       # Min distance to ELF minimum (Shannon-like)
    boundary_radius: float  # Max distance to ELF minimum

@dataclass
class BondPair:
    atom_i: int
    atom_j: int
    jimage: Tuple[int, int, int]
    distance: float
    elf_minimum_frac: np.ndarray
    elf_minimum_value: float
    distance_i: float  # Distance from atom_i to ELF minimum
    distance_j: float  # Distance from atom_j to ELF minimum

@dataclass
class ElectrideSite:
    frac_coord: np.ndarray
    cart_coord: np.ndarray
    elf_value: float
    grid_index: Tuple[int, int, int]
```

### 3.4 DEFAULT_ZVAL

Built-in valence electron counts based on Materials Project recommended POTCARs.
Covers ~90 elements. Examples:

```python
DEFAULT_ZVAL = {
    'H': 1,    # H
    'C': 4,    # C
    'Y': 11,   # Y_sv
    'La': 11,  # La
    'Mg': 8,   # Mg_pv
    ...
}
```

---

## 4. Module Structure

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

---

## 5. Removed/Deprecated (DO NOT USE)

| Name | Removed in | Reason |
|------|------------|--------|
| `atom_cutoffs` | v0.2.0 | Replaced by elf_radii from bond analysis |
| `use_elf_planes` | v0.3.0 | Unstable, simple Voronoi is sufficient |
| `AtomRadii` | v0.3.0 | Renamed to ELFRadii |
| `atom_radii` | v0.3.0 | Renamed to elf_radii |

---

## 6. Example Usage

### Minimal Example

```python
from elfcharge import run_badelf_analysis

result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",
    core_radii={'Y': 0.8, 'C': 0.3},
    elf_threshold=0.5
)

print(result.oxidation.species_avg_oxidation)
```

### Full Example

```python
from elfcharge import run_badelf_analysis

result = run_badelf_analysis(
    elfcar_path="ELFCAR",
    chgcar_path="CHGCAR",
    oxidation_states={'Y': 3, 'C': -4},
    core_radii={'Y': 0.8, 'C': 0.3},
    elf_threshold=0.5,
    apply_smooth=True,
    save_dir="./results",
    save_plots=True,
    save_cif=True
)

# Results
print(f"Y oxidation: {result.oxidation.species_avg_oxidation['Y']:+.2f}")
print(f"C oxidation: {result.oxidation.species_avg_oxidation['C']:+.2f}")
print(f"Electride electrons: {sum(result.charges.electride_electrons):.2f}")
```

---

## 7. Development Rules

1. **SPEC.md is the single source of truth** - Update SPEC.md first before code changes
2. **No backward compatibility hacks** - Remove deprecated code completely
3. **Keep it simple** - Prefer deletion over maintenance of unused features
4. **Sync all docs** - README, README_JP, examples must match SPEC.md
5. **Test after changes** - Every code change must pass tests
