"""
Core BadELF analysis module

Provides high-level interface for running complete BadELF analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .io import GridData, read_elfcar, read_chgcar, write_chgcar
from .analysis import ELFAnalyzer, BondPair, ElectrideSite, AtomRadii
from .partition import BadELFPartitioner, PartitionResult
from .integrate import ChargeIntegrator, OxidationCalculator, ChargeResult, OxidationResult


@dataclass
class BadELFResult:
    """Complete result from BadELF analysis"""
    # Input data
    elf_data: GridData
    chg_data: GridData

    # Bond analysis
    bonds: List[BondPair]
    atom_radii: Dict[int, AtomRadii]

    # Electride sites
    electride_sites: List[ElectrideSite]

    # Partitioning
    partition: PartitionResult

    # Charges
    charges: ChargeResult
    oxidation: OxidationResult

    # Summary
    total_electrons: float
    species_list: List[str]


class BadELFAnalyzer:
    """
    Main BadELF analyzer class

    Implements the complete BadELF workflow:
    1. Read ELFCAR and CHGCAR
    2. Find neighbor pairs
    3. Analyze ELF along bonds
    4. Detect electride sites
    5. Partition space
    6. Integrate charges
    7. Calculate oxidation states
    """

    def __init__(
        self,
        elfcar_path: str,
        chgcar_path: str,
        neighbor_cutoff: float = 4.0,
        elf_threshold: float = 0.5,
        use_pymatgen_nn: bool = True
    ):
        """
        Initialize BadELF analyzer

        Parameters
        ----------
        elfcar_path : str
            Path to ELFCAR file
        chgcar_path : str
            Path to CHGCAR file (should be CHGCAR_sum with core electrons)
        neighbor_cutoff : float
            Distance cutoff for neighbor detection [Å]
        elf_threshold : float
            Minimum ELF value for electride site detection
        use_pymatgen_nn : bool
            Whether to use pymatgen's CrystalNN for neighbor detection
        """
        self.elfcar_path = Path(elfcar_path)
        self.chgcar_path = Path(chgcar_path)
        self.neighbor_cutoff = neighbor_cutoff
        self.elf_threshold = elf_threshold
        self.use_pymatgen_nn = use_pymatgen_nn

        # Load data
        print(f"Loading ELFCAR from {self.elfcar_path}")
        self.elf_data = read_elfcar(str(self.elfcar_path))
        print(f"  Grid size: {self.elf_data.ngrid}")
        print(f"  ELF range: [{self.elf_data.grid.min():.4f}, {self.elf_data.grid.max():.4f}]")

        print(f"Loading CHGCAR from {self.chgcar_path}")
        self.chg_data = read_chgcar(str(self.chgcar_path))
        print(f"  Grid size: {self.chg_data.ngrid}")

        # Check grid sizes
        if self.elf_data.ngrid != self.chg_data.ngrid:
            print(f"  Note: Grid sizes differ - ELFCAR {self.elf_data.ngrid}, CHGCAR {self.chg_data.ngrid}")
            print(f"  Will interpolate/resample as needed for charge integration")
            self.grids_match = False
        else:
            self.grids_match = True

        # Initialize analyzers
        self.elf_analyzer = ELFAnalyzer(self.elf_data)
        self.partitioner = BadELFPartitioner(self.elf_data, self.chg_data)
        self.integrator = ChargeIntegrator(self.chg_data, self.elf_data)
        self.ox_calculator = OxidationCalculator(self.elf_data)

    def run(self) -> BadELFResult:
        """
        Run complete BadELF analysis

        Returns
        -------
        BadELFResult
            Complete analysis results
        """
        print("\n=== BadELF Analysis ===\n")

        # Step 1: Find neighbor pairs
        print("Step 1: Finding neighbor pairs...")
        if self.use_pymatgen_nn:
            try:
                bonds = self.elf_analyzer.get_neighbor_pairs_pymatgen()
                print(f"  Found {len(bonds)} bonds using CrystalNN")
            except ImportError:
                print("  pymatgen not available, using distance-based method")
                bonds = self.elf_analyzer.get_neighbor_pairs(self.neighbor_cutoff)
                print(f"  Found {len(bonds)} bonds within {self.neighbor_cutoff} Å")
        else:
            bonds = self.elf_analyzer.get_neighbor_pairs(self.neighbor_cutoff)
            print(f"  Found {len(bonds)} bonds within {self.neighbor_cutoff} Å")

        # Step 2: Analyze ELF along bonds
        print("\nStep 2: Analyzing ELF along bonds...")
        for i, bond in enumerate(bonds):
            bonds[i] = self.elf_analyzer.analyze_bond_elf(bond)

        # Report some statistics
        min_elf = min(b.elf_minimum_value for b in bonds if b.elf_minimum_value is not None)
        max_elf = max(b.elf_minimum_value for b in bonds if b.elf_minimum_value is not None)
        print(f"  ELF minimum range: [{min_elf:.3f}, {max_elf:.3f}]")

        # Step 3: Compute atom radii
        print("\nStep 3: Computing atomic radii...")
        atom_radii = self.elf_analyzer.compute_atom_radii(bonds)
        species_list = self.elf_data.get_species_list()
        for i, r in atom_radii.items():
            if r.elf_radius is not None:
                print(f"  Atom {i} ({species_list[i]}): ELF radius = {r.elf_radius:.3f} Å, "
                      f"boundary = {r.boundary_radius:.3f} Å")

        # Step 4: Find electride sites
        print("\nStep 4: Detecting electride sites...")
        boundary_radii = {i: r.boundary_radius for i, r in atom_radii.items() if r.boundary_radius}
        electride_sites = self.elf_analyzer.find_electride_sites(
            boundary_radii,
            elf_threshold=self.elf_threshold
        )
        print(f"  Found {len(electride_sites)} potential electride sites")
        for i, site in enumerate(electride_sites):
            print(f"    Site {i}: ELF = {site.elf_value:.3f} at {site.frac_coord}")

        # Step 5: Partition space
        print("\nStep 5: Partitioning space...")
        partition = self.partitioner.partition(bonds, electride_sites, method='watershed')
        print(f"  {partition.n_atoms} atomic regions")
        print(f"  {partition.n_electride_sites} electride regions")

        # Step 6: Integrate charges
        print("\nStep 6: Integrating charges...")
        charges = self.integrator.integrate(partition)
        print(f"  Total electrons: {charges.total_electrons:.4f}")

        # Step 7: Calculate oxidation states
        print("\nStep 7: Calculating oxidation states...")
        oxidation = self.ox_calculator.calculate(charges)

        # Print results
        print("\n=== Results ===\n")
        print("Atomic oxidation states:")
        for i, species in enumerate(species_list):
            print(f"  Atom {i:3d} ({species:2s}): electrons = {charges.atom_electrons[i]:8.4f}, "
                  f"oxidation = {oxidation.atom_oxidation_states[i]:+7.3f}")

        print("\nSpecies average oxidation states:")
        for species, ox in oxidation.species_avg_oxidation.items():
            print(f"  {species}: {ox:+.3f}")

        if len(electride_sites) > 0:
            print("\nElectride sites:")
            for i, site in enumerate(electride_sites):
                print(f"  Site {i}: electrons = {charges.electride_electrons[i]:.4f}, "
                      f"charge = {oxidation.electride_charges[i]:+.4f}")

        return BadELFResult(
            elf_data=self.elf_data,
            chg_data=self.chg_data,
            bonds=bonds,
            atom_radii=atom_radii,
            electride_sites=electride_sites,
            partition=partition,
            charges=charges,
            oxidation=oxidation,
            total_electrons=charges.total_electrons,
            species_list=species_list
        )

    def run_atoms_only(self) -> BadELFResult:
        """
        Run BadELF analysis for atoms only (no electride detection)

        Returns
        -------
        BadELFResult
            Analysis results without electride sites
        """
        print("\n=== BadELF Analysis (Atoms Only) ===\n")

        # Step 1: Find neighbor pairs
        print("Step 1: Finding neighbor pairs...")
        if self.use_pymatgen_nn:
            try:
                bonds = self.elf_analyzer.get_neighbor_pairs_pymatgen()
            except ImportError:
                bonds = self.elf_analyzer.get_neighbor_pairs(self.neighbor_cutoff)
        else:
            bonds = self.elf_analyzer.get_neighbor_pairs(self.neighbor_cutoff)
        print(f"  Found {len(bonds)} bonds")

        # Step 2: Analyze ELF along bonds
        print("\nStep 2: Analyzing ELF along bonds...")
        for i, bond in enumerate(bonds):
            bonds[i] = self.elf_analyzer.analyze_bond_elf(bond)

        # Step 3: Compute atom radii
        print("\nStep 3: Computing atomic radii...")
        atom_radii = self.elf_analyzer.compute_atom_radii(bonds)

        # Step 4: Partition (atoms only)
        print("\nStep 4: Partitioning space...")
        labels = self.partitioner.partition_atoms_only(bonds)
        n_atoms = self.elf_data.n_atoms

        partition = PartitionResult(
            labels=labels,
            n_atoms=n_atoms,
            n_electride_sites=0,
            atom_labels=list(range(1, n_atoms + 1)),
            electride_labels=[]
        )

        # Step 5: Integrate
        print("\nStep 5: Integrating charges...")
        charges = self.integrator.integrate(partition)
        print(f"  Total electrons: {charges.total_electrons:.4f}")

        # Step 6: Oxidation states
        print("\nStep 6: Calculating oxidation states...")
        oxidation = self.ox_calculator.calculate(charges)

        species_list = self.elf_data.get_species_list()

        print("\n=== Results ===\n")
        print("Species average oxidation states:")
        for species, ox in oxidation.species_avg_oxidation.items():
            print(f"  {species}: {ox:+.3f}")

        return BadELFResult(
            elf_data=self.elf_data,
            chg_data=self.chg_data,
            bonds=bonds,
            atom_radii=atom_radii,
            electride_sites=[],
            partition=partition,
            charges=charges,
            oxidation=oxidation,
            total_electrons=charges.total_electrons,
            species_list=species_list
        )

    def save_partition(self, result: BadELFResult, output_path: str):
        """
        Save partition labels to CHGCAR-format file

        Parameters
        ----------
        result : BadELFResult
            Analysis result
        output_path : str
            Output file path
        """
        # Create GridData with labels as the grid
        label_data = GridData(
            lattice=self.elf_data.lattice,
            species=self.elf_data.species,
            num_atoms=self.elf_data.num_atoms,
            frac_coords=self.elf_data.frac_coords,
            grid=result.partition.labels.astype(float),
            ngrid=self.elf_data.ngrid
        )
        write_chgcar(output_path, label_data, comment="BadELF partition labels")
        print(f"Partition saved to {output_path}")


def run_elfcharge(
    elfcar_path: str,
    chgcar_path: str,
    elf_threshold: float = 0.5,
    neighbor_cutoff: float = 4.0,
    atoms_only: bool = False
) -> BadELFResult:
    """
    Convenience function to run BadELF analysis

    Parameters
    ----------
    elfcar_path : str
        Path to ELFCAR file
    chgcar_path : str
        Path to CHGCAR file
    elf_threshold : float
        ELF threshold for electride detection
    neighbor_cutoff : float
        Distance cutoff for neighbors
    atoms_only : bool
        If True, skip electride detection

    Returns
    -------
    BadELFResult
        Analysis results
    """
    analyzer = BadELFAnalyzer(
        elfcar_path=elfcar_path,
        chgcar_path=chgcar_path,
        neighbor_cutoff=neighbor_cutoff,
        elf_threshold=elf_threshold
    )

    if atoms_only:
        return analyzer.run_atoms_only()
    else:
        return analyzer.run()


def run_badelf_analysis(
    elfcar_path: str,
    chgcar_path: str,
    zval: Optional[Dict[str, float]] = None,
    oxidation_states: Optional[Dict[str, float]] = None,
    core_radii: Optional[Dict[str, float]] = None,
    atom_cutoffs: Optional[Dict[str, float]] = None,
    elf_threshold: float = 0.5,
    use_elf_planes: bool = False,
    apply_smooth: bool = False,
    smooth_size: int = 3,
    save_dir: Optional[str] = None,
    save_plots: bool = True,
    save_cif: bool = True,
    verbose: bool = True
) -> BadELFResult:
    """
    High-level wrapper for complete BadELF analysis with visualization and export.

    This is the recommended entry point for easy-use analysis. It combines
    all steps: loading data, bond analysis, electride detection, partitioning,
    charge integration, and optionally saves plots and structure files.

    Parameters
    ----------
    elfcar_path : str
        Path to ELFCAR file
    chgcar_path : str
        Path to CHGCAR file (should be CHGCAR_sum with core electrons)
    zval : Dict[str, float], optional
        Valence electrons per species. E.g., {'La': 11, 'Mg': 8, 'H': 1}
        If not provided, uses default values from the package.
    oxidation_states : Dict[str, float], optional
        Expected oxidation states per species for improved CrystalNN accuracy.
        E.g., {'La': 3, 'Mg': 2, 'H': -1}
    core_radii : Dict[str, float], optional
        Core radii to exclude when finding ELF minima along bonds.
        E.g., {'La': 0.8, 'Mg': 0.5, 'H': 0.0}
    atom_cutoffs : Dict[str, float], optional
        Species-specific cutoffs for electride detection (fallback if bonds
        not analyzed). E.g., {'La': 1.5, 'Mg': 1.2, 'H': 0.5}
    elf_threshold : float
        Minimum ELF value for electride site detection (default: 0.5)
    use_elf_planes : bool
        If True, use ELF minimum planes to adjust atom-atom boundaries.
        If False (default), use simple Voronoi partitioning which is more stable.
    apply_smooth : bool
        If True, apply smoothing to ELF data for cleaner visualization.
        Default: False
    smooth_size : int
        Size of smoothing kernel (default: 3)
    save_dir : str, optional
        Directory to save output files. If None, uses current directory.
    save_plots : bool
        If True, generate and save visualization plots. Default: True
    save_cif : bool
        If True, export structure with electride sites to CIF. Default: True
    verbose : bool
        If True, print progress messages. Default: True

    Returns
    -------
    BadELFResult
        Complete analysis results including charges, oxidation states, etc.

    Example
    -------
    >>> from elfcharge import run_badelf_analysis
    >>> result = run_badelf_analysis(
    ...     elfcar_path="ELFCAR",
    ...     chgcar_path="CHGCAR_sum",
    ...     zval={'La': 11, 'Mg': 8, 'H': 1},
    ...     oxidation_states={'La': 3, 'Mg': 2, 'H': -1},
    ...     core_radii={'La': 0.8, 'Mg': 0.5, 'H': 0.0},
    ...     apply_smooth=True,
    ...     save_dir="./badelf_results"
    ... )
    >>> print(result.oxidation.species_avg_oxidation)
    """
    from .io import read_elfcar, read_chgcar, check_grid_resolution
    from .structure import create_structure_with_electrides

    # Setup output directory
    if save_dir is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    if verbose:
        print("\n" + "="*60)
        print(" BadELF Analysis - Complete Workflow")
        print("="*60)
        print(f"\n[1/7] Loading data...")
        print(f"  ELFCAR: {elfcar_path}")
        print(f"  CHGCAR: {chgcar_path}")

    elf_data = read_elfcar(
        str(elfcar_path),
        smooth=apply_smooth,
        smooth_size=smooth_size,
        check_resolution=verbose
    )
    chg_data = read_chgcar(str(chgcar_path))

    if verbose:
        print(f"  ELF grid: {elf_data.ngrid}, range: [{elf_data.grid.min():.3f}, {elf_data.grid.max():.3f}]")
        print(f"  CHG grid: {chg_data.ngrid}")
        if apply_smooth:
            print(f"  Smoothing applied (size={smooth_size})")

        # Check resolution
        res_info = check_grid_resolution(elf_data)
        print(f"  Resolution: {res_info['average']:.1f} voxels/Å")

    # Step 2: Find neighbor pairs
    if verbose:
        print(f"\n[2/7] Finding neighbor pairs (CrystalNN)...")

    elf_analyzer = ELFAnalyzer(elf_data)
    bonds = elf_analyzer.get_neighbor_pairs_pymatgen(oxidation_states=oxidation_states)

    if verbose:
        print(f"  Found {len(bonds)} bonds")

    # Step 3: Analyze ELF along bonds
    if verbose:
        print(f"\n[3/7] Analyzing ELF along bonds...")

    for i, bond in enumerate(bonds):
        bonds[i] = elf_analyzer.analyze_bond_elf(bond, core_radii=core_radii)

    # Compute atom radii
    atom_radii = elf_analyzer.compute_atom_radii(bonds)
    species_list = elf_data.get_species_list()

    if verbose:
        print("  Atom radii (ELF / Boundary):")
        for i, r in atom_radii.items():
            if r.elf_radius is not None:
                print(f"    {i:3d} ({species_list[i]:2s}): {r.elf_radius:.3f} / {r.boundary_radius:.3f} Å")

    # Step 4: Find electride sites
    if verbose:
        print(f"\n[4/7] Detecting electride sites (ELF >= {elf_threshold})...")

    electride_sites = elf_analyzer.find_interstitial_electrides(
        elf_threshold=elf_threshold,
        atom_cutoffs=atom_cutoffs,
        atom_radii=atom_radii,
        use_boundary_radius=True
    )

    if verbose:
        print(f"  Found {len(electride_sites)} electride sites")
        for i, site in enumerate(electride_sites):
            print(f"    Site {i}: ELF={site.elf_value:.3f} at ({site.frac_coord[0]:.3f}, "
                  f"{site.frac_coord[1]:.3f}, {site.frac_coord[2]:.3f})")

    # Step 5: Partition space
    if verbose:
        print(f"\n[5/7] Partitioning space...")
        if use_elf_planes:
            print("  Using ELF plane boundaries (experimental)")
        else:
            print("  Using simple Voronoi partitioning")

    partitioner = BadELFPartitioner(elf_data, chg_data)
    partition = partitioner.partition(
        bonds, electride_sites,
        method='watershed',
        use_elf_planes=use_elf_planes
    )

    if verbose:
        print(f"  Atomic regions: {partition.n_atoms}")
        print(f"  Electride regions: {partition.n_electride_sites}")

    # Step 6: Integrate charges
    if verbose:
        print(f"\n[6/7] Integrating charges...")

    integrator = ChargeIntegrator(chg_data, elf_data)
    charges = integrator.integrate(partition)

    # Step 7: Calculate oxidation states
    if verbose:
        print(f"\n[7/7] Calculating oxidation states...")

    ox_calculator = OxidationCalculator(elf_data, zval=zval)
    oxidation = ox_calculator.calculate(charges)

    # Print results summary
    if verbose:
        print("\n" + "="*60)
        print(" Results Summary")
        print("="*60)
        print(f"\nTotal electrons: {charges.total_electrons:.4f}")

        print("\nAtom charges:")
        for i, species in enumerate(species_list):
            print(f"  {i:3d} ({species:2s}): e={charges.atom_electrons[i]:7.3f}, "
                  f"ox={oxidation.atom_oxidation_states[i]:+6.2f}")

        print("\nSpecies average oxidation states:")
        for species, ox in oxidation.species_avg_oxidation.items():
            print(f"  {species}: {ox:+.3f}")

        if len(electride_sites) > 0:
            print("\nElectride charges:")
            total_e_charge = 0.0
            for i in range(len(electride_sites)):
                e_charge = oxidation.electride_charges[i]
                total_e_charge += e_charge
                print(f"  Site {i}: {e_charge:+.4f} e")
            print(f"  Total: {total_e_charge:+.4f} e")

    # Create result object
    result = BadELFResult(
        elf_data=elf_data,
        chg_data=chg_data,
        bonds=bonds,
        atom_radii=atom_radii,
        electride_sites=electride_sites,
        partition=partition,
        charges=charges,
        oxidation=oxidation,
        total_electrons=charges.total_electrons,
        species_list=species_list
    )

    # Save outputs
    if save_cif and len(electride_sites) > 0:
        if verbose:
            print(f"\nSaving structure with electride sites...")
        try:
            struct = create_structure_with_electrides(elf_data, electride_sites)
            cif_path = save_dir / "structure_with_electrides.cif"
            struct.to(filename=str(cif_path))
            if verbose:
                print(f"  Saved: {cif_path}")
        except Exception as e:
            print(f"  Warning: Could not save CIF: {e}")

    if save_plots:
        if verbose:
            print(f"\nGenerating plots...")
        try:
            from .visualize import (
                plot_elf_slice_with_partition,
                plot_electron_distribution,
                HAS_VISUALIZATION
            )
            if not HAS_VISUALIZATION:
                print("  Warning: matplotlib not available, skipping plots")
            else:
                import matplotlib.pyplot as plt

                # Plot 1: ELF slice with partition (xy plane at z=0.5)
                fig1 = plot_elf_slice_with_partition(
                    elf_data, partition.labels,
                    plane='xy', position=0.5,
                    electride_sites=electride_sites,
                    interpolate=True,
                    interpolate_factor=3
                )
                fig1.savefig(save_dir / "elf_partition_xy.png", dpi=150, bbox_inches='tight')
                plt.close(fig1)

                # Plot 2: Electron distribution
                fig2 = plot_electron_distribution(
                    charges.atom_electrons,
                    species_list,
                    oxidation.atom_oxidation_states
                )
                fig2.savefig(save_dir / "electron_distribution.png", dpi=150, bbox_inches='tight')
                plt.close(fig2)

                if verbose:
                    print(f"  Saved: elf_partition_xy.png")
                    print(f"  Saved: electron_distribution.png")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    if verbose:
        print("\n" + "="*60)
        print(" Analysis complete!")
        print("="*60 + "\n")

    return result
