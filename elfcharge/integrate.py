"""
Charge integration module for BadELF

Integrates charge density within partitioned regions to obtain:
- Electron counts for each atom
- Electron counts for each electride site
- Oxidation states
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from .io import GridData
from .partition import PartitionResult


# Atomic numbers for common elements
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
}


@dataclass
class ChargeResult:
    """Result of charge integration"""
    # Per-region charges
    region_charges: Dict[int, float]     # label -> electron count

    # Atomic charges (electrons in each atomic region)
    atom_electrons: np.ndarray           # (n_atoms,) electrons per atom

    # Electride charges (electrons in each electride region)
    electride_electrons: np.ndarray      # (n_electride,) electrons per electride

    # Total electrons
    total_electrons: float

    # Region volumes
    region_volumes: Dict[int, float]     # label -> volume in Å³


@dataclass
class OxidationResult:
    """Oxidation state calculation result"""
    # Oxidation states for atoms (nuclear charge - electrons)
    atom_oxidation_states: np.ndarray    # (n_atoms,)

    # Effective charge for electrides (= -electrons)
    electride_charges: np.ndarray        # (n_electride,)

    # Per-species average oxidation states
    species_avg_oxidation: Dict[str, float]

    # Detailed info
    atom_electrons: np.ndarray
    electride_electrons: np.ndarray


class ChargeIntegrator:
    """
    Integrates charge density within partitioned regions
    """

    def __init__(self, chg_data: GridData, elf_data: Optional[GridData] = None):
        """
        Initialize integrator

        Parameters
        ----------
        chg_data : GridData
            Charge density grid (CHGCAR)
        elf_data : GridData, optional
            ELF grid for structure reference
        """
        self.chg_data = chg_data
        self.elf_data = elf_data if elf_data else chg_data
        self.chg_grid = chg_data.grid

        # Check if grids match
        self.grids_match = (elf_data is None or
                           chg_data.ngrid == elf_data.ngrid)

    def _resample_labels(self, labels: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Resample partition labels to a different grid size using nearest neighbor

        Parameters
        ----------
        labels : np.ndarray
            Original label array
        target_shape : tuple
            Target grid shape

        Returns
        -------
        np.ndarray
            Resampled label array
        """
        from scipy.ndimage import zoom

        source_shape = labels.shape
        zoom_factors = tuple(t / s for t, s in zip(target_shape, source_shape))

        # Use order=0 for nearest neighbor interpolation (preserves integer labels)
        resampled = zoom(labels.astype(float), zoom_factors, order=0)
        return resampled.astype(np.int32)

    def integrate(self, partition: PartitionResult) -> ChargeResult:
        """
        Integrate charge in each partitioned region

        Parameters
        ----------
        partition : PartitionResult
            Partitioning result with labels

        Returns
        -------
        ChargeResult
            Integrated charges for each region
        """
        labels = partition.labels
        n_atoms = partition.n_atoms
        n_electride = partition.n_electride_sites

        # Resample labels if grid sizes differ
        chg_shape = self.chg_data.ngrid
        if labels.shape != chg_shape:
            print(f"  Resampling labels from {labels.shape} to {chg_shape}")
            labels = self._resample_labels(labels, chg_shape)

        volume = self.chg_data.volume
        n_grid = np.prod(self.chg_data.ngrid)
        dv = volume / n_grid  # Volume per grid point

        # CHGCAR stores ρ × V_cell, so:
        # electrons = Σ(CHGCAR value) / N_grid
        region_charges = {}
        region_volumes = {}

        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == 0:
                continue  # Skip unlabeled

            mask = labels == label
            # Sum charge density values
            charge_sum = np.sum(self.chg_grid[mask])
            # Convert to electron count
            electrons = charge_sum / n_grid
            region_charges[label] = electrons

            # Volume of region
            region_volumes[label] = np.sum(mask) * dv

        # Separate into atoms and electrides
        atom_electrons = np.zeros(n_atoms)
        electride_electrons = np.zeros(n_electride)

        for label in partition.atom_labels:
            idx = label - 1  # Convert to 0-indexed
            if label in region_charges:
                atom_electrons[idx] = region_charges[label]

        for i, label in enumerate(partition.electride_labels):
            if label in region_charges:
                electride_electrons[i] = region_charges[label]

        total_electrons = sum(region_charges.values())

        return ChargeResult(
            region_charges=region_charges,
            atom_electrons=atom_electrons,
            electride_electrons=electride_electrons,
            total_electrons=total_electrons,
            region_volumes=region_volumes
        )

    def integrate_simple(self, labels: np.ndarray, n_atoms: int) -> ChargeResult:
        """
        Simple integration without full PartitionResult

        Parameters
        ----------
        labels : np.ndarray
            Label array (atoms: 1 to n_atoms, electrides: > n_atoms)
        n_atoms : int
            Number of atoms

        Returns
        -------
        ChargeResult
            Integrated charges
        """
        # Resample labels if grid sizes differ
        chg_shape = self.chg_data.ngrid
        if labels.shape != chg_shape:
            print(f"  Resampling labels from {labels.shape} to {chg_shape}")
            labels = self._resample_labels(labels, chg_shape)

        volume = self.chg_data.volume
        n_grid = np.prod(self.chg_data.ngrid)
        dv = volume / n_grid

        region_charges = {}
        region_volumes = {}

        unique_labels = np.unique(labels)
        max_label = int(np.max(unique_labels))
        n_electride = max(0, max_label - n_atoms)

        for label in unique_labels:
            if label == 0:
                continue

            mask = labels == label
            charge_sum = np.sum(self.chg_grid[mask])
            electrons = charge_sum / n_grid
            region_charges[int(label)] = electrons
            region_volumes[int(label)] = np.sum(mask) * dv

        # Separate
        atom_electrons = np.zeros(n_atoms)
        electride_electrons = np.zeros(n_electride)

        for i in range(n_atoms):
            label = i + 1
            if label in region_charges:
                atom_electrons[i] = region_charges[label]

        for i in range(n_electride):
            label = n_atoms + i + 1
            if label in region_charges:
                electride_electrons[i] = region_charges[label]

        return ChargeResult(
            region_charges=region_charges,
            atom_electrons=atom_electrons,
            electride_electrons=electride_electrons,
            total_electrons=sum(region_charges.values()),
            region_volumes=region_volumes
        )


class OxidationCalculator:
    """
    Calculate oxidation states from integrated charges
    """

    def __init__(self, structure: GridData, zval: Optional[Dict[str, float]] = None):
        """
        Initialize calculator

        Parameters
        ----------
        structure : GridData
            Structure data with species information
        zval : Dict[str, float], optional
            Valence electrons per species (from POTCAR).
            If provided, uses these values instead of atomic numbers.
            E.g., {'Y': 11, 'C': 4, 'F': 7}
        """
        self.structure = structure
        self.species_list = structure.get_species_list()
        self.zval = zval

    def calculate(self, charges: ChargeResult) -> OxidationResult:
        """
        Calculate oxidation states

        Oxidation state = Nuclear charge - Electron count

        Parameters
        ----------
        charges : ChargeResult
            Integrated charges

        Returns
        -------
        OxidationResult
            Oxidation states for all atoms and electrides
        """
        n_atoms = len(self.species_list)
        atom_ox = np.zeros(n_atoms)

        for i, species in enumerate(self.species_list):
            # Use ZVAL if provided, otherwise fall back to atomic number
            if self.zval is not None and species in self.zval:
                z = self.zval[species]
            else:
                z = self.get_nuclear_charge(species)
            electrons = charges.atom_electrons[i]
            atom_ox[i] = z - electrons

        # Electride charges (no nucleus, so charge = -electrons)
        electride_charges = -charges.electride_electrons

        # Species average
        species_avg = {}
        species_counts = {}
        for i, species in enumerate(self.species_list):
            if species not in species_avg:
                species_avg[species] = 0.0
                species_counts[species] = 0
            species_avg[species] += atom_ox[i]
            species_counts[species] += 1

        for species in species_avg:
            species_avg[species] /= species_counts[species]

        return OxidationResult(
            atom_oxidation_states=atom_ox,
            electride_charges=electride_charges,
            species_avg_oxidation=species_avg,
            atom_electrons=charges.atom_electrons,
            electride_electrons=charges.electride_electrons
        )

    def get_nuclear_charge(self, species: str) -> int:
        """
        Get nuclear charge (atomic number) for a species

        Parameters
        ----------
        species : str
            Element symbol

        Returns
        -------
        int
            Atomic number
        """
        # Clean up species string (remove numbers, etc.)
        clean = ''.join(c for c in species if c.isalpha())
        clean = clean.capitalize()

        if clean in ATOMIC_NUMBERS:
            return ATOMIC_NUMBERS[clean]
        else:
            raise ValueError(f"Unknown element: {species}")


def integrate_charges(
    chg_data: GridData,
    labels: np.ndarray,
    n_atoms: int
) -> ChargeResult:
    """
    Convenience function to integrate charges

    Parameters
    ----------
    chg_data : GridData
        Charge density data
    labels : np.ndarray
        Partitioning labels
    n_atoms : int
        Number of atoms

    Returns
    -------
    ChargeResult
        Integration result
    """
    integrator = ChargeIntegrator(chg_data)
    return integrator.integrate_simple(labels, n_atoms)


def calculate_oxidation_states(
    structure: GridData,
    charges: ChargeResult
) -> OxidationResult:
    """
    Convenience function to calculate oxidation states

    Parameters
    ----------
    structure : GridData
        Structure data
    charges : ChargeResult
        Integrated charges

    Returns
    -------
    OxidationResult
        Oxidation states
    """
    calculator = OxidationCalculator(structure)
    return calculator.calculate(charges)
