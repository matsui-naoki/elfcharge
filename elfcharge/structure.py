"""
Structure utilities for BadELF

Functions for creating pymatgen Structure objects with electride sites
and exporting to various formats (CIF, POSCAR, etc.)
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .io import GridData
from .analysis import ElectrideSite


def create_structure_with_electrides(
    grid_data: GridData,
    electride_sites: List[ElectrideSite],
    electride_symbol: str = "X",
    electride_electrons: Optional[List[float]] = None
) -> "Structure":
    """
    Create a pymatgen Structure object with electride sites as dummy atoms.

    Parameters
    ----------
    grid_data : GridData
        Original structure data from ELFCAR/CHGCAR
    electride_sites : List[ElectrideSite]
        Detected electride sites
    electride_symbol : str
        Symbol to use for electride sites (default: "X" for dummy atom)
        Other options: "He" (inert, ~0 electrons), "Og" (for visibility)
    electride_electrons : List[float], optional
        Electron counts for each electride site (for site properties)

    Returns
    -------
    Structure
        pymatgen Structure object with electride sites included

    Notes
    -----
    The electride sites are added as dummy atoms with the specified symbol.
    Site properties include:
    - 'elf_value': ELF value at the site
    - 'electrons': Number of electrons (if provided)
    - 'is_electride': True for electride sites, False for real atoms
    """
    try:
        from pymatgen.core import Structure, Lattice, DummySpecies, Element
    except ImportError:
        raise ImportError("pymatgen is required for structure output. "
                          "Install with: pip install pymatgen")

    # Create lattice
    lattice = Lattice(grid_data.lattice)

    # Build species and coordinates lists
    species_list = grid_data.get_species_list()
    all_species = []
    all_coords = []
    site_properties = {
        'elf_value': [],
        'is_electride': [],
        'electrons': []
    }

    # Add original atoms
    for i, sp in enumerate(species_list):
        all_species.append(Element(sp))
        all_coords.append(grid_data.frac_coords[i])
        site_properties['elf_value'].append(None)
        site_properties['is_electride'].append(False)
        site_properties['electrons'].append(None)

    # Add electride sites
    for i, site in enumerate(electride_sites):
        # Use DummySpecies for unknown/dummy atoms
        if electride_symbol == "X":
            all_species.append(DummySpecies("X"))
        else:
            try:
                all_species.append(Element(electride_symbol))
            except:
                all_species.append(DummySpecies(electride_symbol))

        all_coords.append(site.frac_coord)
        site_properties['elf_value'].append(site.elf_value)
        site_properties['is_electride'].append(True)
        if electride_electrons is not None and i < len(electride_electrons):
            site_properties['electrons'].append(electride_electrons[i])
        else:
            site_properties['electrons'].append(None)

    # Create structure
    structure = Structure(
        lattice,
        all_species,
        all_coords,
        site_properties=site_properties
    )

    return structure


def write_structure_cif(
    structure: "Structure",
    filepath: str,
    comment: str = "Structure with electride sites from BadELF analysis"
):
    """
    Write structure to CIF format.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object
    filepath : str
        Output file path
    comment : str
        Comment to include in the CIF file
    """
    structure.to(filename=filepath, fmt="cif")
    print(f"CIF file written to: {filepath}")


def write_structure_poscar(
    structure: "Structure",
    filepath: str,
    comment: str = "Structure with electride sites"
):
    """
    Write structure to POSCAR format.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object
    filepath : str
        Output file path
    comment : str
        Comment for the POSCAR file
    """
    structure.to(filename=filepath, fmt="poscar")
    print(f"POSCAR file written to: {filepath}")


def get_electride_summary(structure: "Structure") -> Dict[str, Any]:
    """
    Get summary of electride sites in the structure.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object with electride sites

    Returns
    -------
    dict
        Summary including counts, positions, and properties
    """
    summary = {
        'n_atoms': 0,
        'n_electrides': 0,
        'electride_positions': [],
        'electride_elf_values': [],
        'electride_electrons': [],
        'species_counts': {}
    }

    for site in structure:
        sp_str = str(site.specie)
        if site.properties.get('is_electride', False):
            summary['n_electrides'] += 1
            summary['electride_positions'].append(site.frac_coords.tolist())
            summary['electride_elf_values'].append(site.properties.get('elf_value'))
            summary['electride_electrons'].append(site.properties.get('electrons'))
        else:
            summary['n_atoms'] += 1

        if sp_str not in summary['species_counts']:
            summary['species_counts'][sp_str] = 0
        summary['species_counts'][sp_str] += 1

    return summary


@dataclass
class BadELFResult:
    """
    Container for BadELF analysis results.

    Attributes
    ----------
    structure : Structure
        pymatgen Structure with electride sites
    atom_electrons : np.ndarray
        Electron counts for each atom
    atom_oxidation : np.ndarray
        Oxidation states for each atom
    electride_electrons : np.ndarray
        Electron counts for each electride site
    electride_sites : List[ElectrideSite]
        List of detected electride sites
    total_electrons : float
        Total electron count
    species_stats : dict
        Statistics by species (mean, std of electrons and oxidation)
    """
    structure: Any  # pymatgen Structure
    atom_electrons: np.ndarray
    atom_oxidation: np.ndarray
    electride_electrons: np.ndarray
    electride_sites: List[ElectrideSite]
    total_electrons: float
    species_stats: Dict[str, Dict[str, float]]

    def to_cif(self, filepath: str):
        """Export structure with electride sites to CIF file."""
        write_structure_cif(self.structure, filepath)

    def to_poscar(self, filepath: str):
        """Export structure with electride sites to POSCAR file."""
        write_structure_poscar(self.structure, filepath)

    def get_charge_neutrality(self) -> Dict[str, float]:
        """
        Check charge neutrality.

        Returns
        -------
        dict
            Contains total_charge, contributions by species, and electride contribution
        """
        result = {
            'contributions': {},
            'electride_contribution': -np.sum(self.electride_electrons),
            'total_charge': 0.0
        }

        for sp, stats in self.species_stats.items():
            contrib = stats['count'] * stats['mean_oxidation']
            result['contributions'][sp] = contrib
            result['total_charge'] += contrib

        result['total_charge'] += result['electride_contribution']

        return result

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 60)
        lines.append("BadELF Analysis Results")
        lines.append("=" * 60)

        lines.append(f"\nTotal electrons: {self.total_electrons:.4f}")

        lines.append("\nSpecies statistics:")
        lines.append("-" * 40)
        for sp, stats in self.species_stats.items():
            lines.append(f"  {sp:3s}: {stats['mean_oxidation']:+.3f} ± {stats['std_oxidation']:.3f} "
                        f"({stats['count']} atoms, {stats['mean_electrons']:.2f} e/atom)")

        if len(self.electride_sites) > 0:
            lines.append(f"\nElectride sites: {len(self.electride_sites)}")
            lines.append("-" * 40)
            total_e = 0
            for i, (site, e) in enumerate(zip(self.electride_sites, self.electride_electrons)):
                lines.append(f"  Site {i}: {e:.4f} electrons (ELF={site.elf_value:.3f})")
                total_e += e
            lines.append(f"  Total electride electrons: {total_e:.4f}")

        # Charge neutrality
        neutrality = self.get_charge_neutrality()
        lines.append("\nCharge neutrality check:")
        lines.append("-" * 40)
        lines.append(f"  Total charge: {neutrality['total_charge']:+.4f}")

        return "\n".join(lines)
