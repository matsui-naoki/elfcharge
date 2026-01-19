"""
Visualization utilities for BadELF analysis

Functions for plotting ELF profiles, partitioning results, and electron distributions.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import warnings

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
    HAS_VISUALIZATION = True  # Alias for backwards compatibility
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_VISUALIZATION = False
    warnings.warn("matplotlib not available. Visualization functions will not work.")

from .io import GridData
from .analysis import BondPair, ElectrideSite


def plot_elf_along_bond(
    elf_data: GridData,
    bond: BondPair,
    ax: Optional[Any] = None,
    n_points: int = 100,
    show_minimum: bool = True,
    **kwargs
) -> Any:
    """
    Plot ELF values along a bond path.

    Parameters
    ----------
    elf_data : GridData
        ELF grid data
    bond : BondPair
        Bond pair to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    n_points : int
        Number of points along the path
    show_minimum : bool
        Whether to mark the ELF minimum position
    **kwargs
        Additional keyword arguments for plt.plot()

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Get positions
    frac_coords = elf_data.frac_coords
    lattice = elf_data.lattice

    pos_i = frac_coords[bond.atom_i]
    pos_j = frac_coords[bond.atom_j] + np.array(bond.jimage)

    # Generate path
    t = np.linspace(0, 1, n_points)
    path_frac = pos_i[None, :] + t[:, None] * (pos_j - pos_i)[None, :]

    # Interpolate ELF
    from .analysis import ELFAnalyzer
    analyzer = ELFAnalyzer(elf_data)
    elf_values = analyzer.interpolate_elf(path_frac)

    # Calculate distances
    path_cart = path_frac @ lattice
    distances = np.linalg.norm(path_cart - path_cart[0], axis=1)

    # Plot
    species_list = elf_data.get_species_list()
    label = f"{species_list[bond.atom_i]}-{species_list[bond.atom_j]}"
    ax.plot(distances, elf_values, label=label, **kwargs)

    # Mark minimum
    if show_minimum and bond.elf_minimum_frac is not None:
        ax.axvline(x=bond.distance_i, color='red', linestyle='--', alpha=0.5,
                   label=f'ELF min (d={bond.distance_i:.2f}Å)')
        ax.scatter([bond.distance_i], [bond.elf_minimum_value],
                   color='red', s=50, zorder=5)

    ax.set_xlabel('Distance from atom i (Å)')
    ax.set_ylabel('ELF')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_elf_profiles_by_type(
    elf_data: GridData,
    bonds: List[BondPair],
    figsize: Tuple[int, int] = (12, 8),
    max_bonds_per_type: int = 5
) -> Any:
    """
    Plot ELF profiles grouped by bond type.

    Parameters
    ----------
    elf_data : GridData
        ELF grid data
    bonds : List[BondPair]
        List of analyzed bonds
    figsize : tuple
        Figure size
    max_bonds_per_type : int
        Maximum number of bonds to plot per type

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Group bonds by type
    species_list = elf_data.get_species_list()
    bond_types = {}
    for bond in bonds:
        sp_i = species_list[bond.atom_i]
        sp_j = species_list[bond.atom_j]
        bond_type = f"{sp_i}-{sp_j}" if sp_i <= sp_j else f"{sp_j}-{sp_i}"
        if bond_type not in bond_types:
            bond_types[bond_type] = []
        bond_types[bond_type].append(bond)

    n_types = len(bond_types)
    ncols = min(3, n_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax_idx, (bond_type, bond_list) in enumerate(sorted(bond_types.items())):
        ax = axes[ax_idx]

        for i, bond in enumerate(bond_list[:max_bonds_per_type]):
            plot_elf_along_bond(elf_data, bond, ax=ax, show_minimum=False,
                               alpha=0.7, linewidth=1)

        ax.set_title(f'{bond_type} ({len(bond_list)} bonds)')
        ax.legend().set_visible(False)

    # Hide unused axes
    for ax_idx in range(len(bond_types), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_radial_elf(
    elf_data: GridData,
    atom_index: int,
    r_max: float = 3.0,
    n_radii: int = 30,
    n_angles: int = 20,
    ax: Optional[Any] = None
) -> Any:
    """
    Plot radially averaged ELF around an atom.

    Parameters
    ----------
    elf_data : GridData
        ELF grid data
    atom_index : int
        Index of the atom to analyze
    r_max : float
        Maximum radius in Angstroms
    n_radii : int
        Number of radial points
    n_angles : int
        Number of angular samples for averaging
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    from .analysis import ELFAnalyzer
    analyzer = ELFAnalyzer(elf_data)

    atom_frac = elf_data.frac_coords[atom_index]
    lattice = elf_data.lattice
    species = elf_data.get_species_list()[atom_index]

    radii = np.linspace(0.1, r_max, n_radii)
    radial_elf = []

    for r in radii:
        elf_at_r = []
        for theta in np.linspace(0, np.pi, n_angles):
            for phi in np.linspace(0, 2*np.pi, n_angles):
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)

                cart_point = (atom_frac @ lattice) + np.array([dx, dy, dz])
                frac_point = cart_point @ np.linalg.inv(lattice)
                elf_val = analyzer.interpolate_elf(frac_point)
                elf_at_r.append(elf_val)

        radial_elf.append(np.mean(elf_at_r))

    ax.plot(radii, radial_elf, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='ELF=0.5')
    ax.set_xlabel('Distance from atom (Å)')
    ax.set_ylabel('Average ELF')
    ax.set_title(f'Radial ELF profile around {species} (atom {atom_index})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_electron_distribution(
    atom_electrons: np.ndarray,
    species_list: List[str],
    reference: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Any:
    """
    Plot electron distribution by species.

    Parameters
    ----------
    atom_electrons : np.ndarray
        Electron counts for each atom
    species_list : List[str]
        Species for each atom
    reference : dict, optional
        Reference electron counts (e.g., ZVAL)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get unique species
    unique_species = sorted(set(species_list))

    # Left: Violin/box plot of electron counts
    ax1 = axes[0]
    data_by_species = []
    for sp in unique_species:
        electrons = [atom_electrons[i] for i, s in enumerate(species_list) if s == sp]
        data_by_species.append(electrons)

    parts = ax1.violinplot(data_by_species, showmeans=True)
    ax1.set_xticks(range(1, len(unique_species) + 1))
    ax1.set_xticklabels(unique_species)
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Electrons')
    ax1.set_title('Electron distribution by species')

    # Add reference lines
    if reference:
        for i, sp in enumerate(unique_species):
            if sp in reference:
                ax1.axhline(y=reference[sp], xmin=(i+0.5)/len(unique_species),
                           xmax=(i+1.5)/len(unique_species),
                           color='red', linestyle='--', alpha=0.7)

    # Right: Oxidation state histogram
    ax2 = axes[1]
    if reference:
        oxidation_states = []
        colors = []
        color_map = plt.cm.tab10

        for i, sp in enumerate(unique_species):
            ref = reference.get(sp, 0)
            ox_states = [ref - atom_electrons[j] for j, s in enumerate(species_list) if s == sp]
            oxidation_states.extend(ox_states)

            # Create histogram for this species
            ax2.hist(ox_states, bins=20, alpha=0.6, label=sp,
                    color=color_map(i / len(unique_species)))

        ax2.set_xlabel('Oxidation State')
        ax2.set_ylabel('Count')
        ax2.set_title('Oxidation state distribution')
        ax2.legend()
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_electride_elf_histogram(
    electride_sites: List[ElectrideSite],
    electride_electrons: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> Any:
    """
    Plot histogram of ELF values at electride sites.

    Parameters
    ----------
    electride_sites : List[ElectrideSite]
        List of electride sites
    electride_electrons : np.ndarray, optional
        Electron counts for each site
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ELF values
    elf_values = [site.elf_value for site in electride_sites]

    ax1 = axes[0]
    ax1.hist(elf_values, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(elf_values), color='red', linestyle='--',
                label=f'Mean: {np.mean(elf_values):.3f}')
    ax1.set_xlabel('ELF value')
    ax1.set_ylabel('Count')
    ax1.set_title(f'ELF at electride sites (n={len(electride_sites)})')
    ax1.legend()

    # Electron counts
    ax2 = axes[1]
    if electride_electrons is not None:
        ax2.hist(electride_electrons, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(electride_electrons), color='red', linestyle='--',
                    label=f'Mean: {np.mean(electride_electrons):.3f}')
        ax2.set_xlabel('Electrons per site')
        ax2.set_ylabel('Count')
        ax2.set_title('Electron count at electride sites')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No electron data', ha='center', va='center',
                transform=ax2.transAxes)

    plt.tight_layout()
    return fig


def plot_elf_slice_with_partition(
    elf_data: GridData,
    labels: np.ndarray,
    plane: str = 'xy',
    index: Optional[int] = None,
    position: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    # Interpolation
    interpolate: bool = False,
    interpolate_factor: int = 2,
    # Coordinate system
    coord_system: str = 'cartesian',  # 'cartesian' or 'fractional'
    # Atom display options
    show_atoms: bool = True,
    show_atom_symbol: bool = True,
    atom_symbol: str = 'o',
    atom_symbol_size: int = 200,
    atom_symbol_color: Optional[Dict[str, str]] = None,
    # Element label options
    show_element_label: bool = True,
    element_label_size: int = 10,
    element_label_color: str = 'black',
    # Electride display options
    show_electrides: bool = True,
    electride_sites: Optional[List[ElectrideSite]] = None,
    electride_symbol: str = '*',
    electride_symbol_size: int = 300,
    electride_symbol_color: str = 'yellow',
    electride_label: str = 'e⁻',
    electride_label_size: int = 10,
    electride_label_color: str = 'black',
    # Other options
    cmap: str = 'RdYlBu_r',
    boundary_color: str = 'black',
    boundary_width: float = 0.5,
    slice_tolerance: float = 0.08,
    equal_aspect: bool = True
) -> Any:
    """
    Plot a 2D slice of the ELF with partition boundaries.

    Parameters
    ----------
    elf_data : GridData
        ELF grid data
    labels : np.ndarray
        Partition labels array (NGX, NGY, NGZ)
    plane : str
        Plane to slice ('xy', 'xz', or 'yz')
    index : int, optional
        Grid index for the slice (if None, uses position)
    position : float, optional
        Fractional position for the slice (0-1)
    figsize : tuple, optional
        Figure size. If None, auto-calculated for equal aspect ratio
    interpolate : bool
        If True, interpolate data for higher resolution image
    interpolate_factor : int
        Factor to increase resolution (e.g., 2 = double resolution)
    coord_system : str
        Coordinate system for axes: 'cartesian' (Å) or 'fractional'
    show_atoms : bool
        Whether to show atom positions
    show_atom_symbol : bool
        Whether to show atom markers
    atom_symbol : str
        Marker style for atoms (matplotlib marker, e.g., 'o', 's', '^')
    atom_symbol_size : int
        Marker size for atoms
    atom_symbol_color : dict, optional
        Element-specific colors, e.g., {'La': 'blue', 'Mg': 'green'}
        If None, uses default colors
    show_element_label : bool
        Whether to show element labels next to atoms
    element_label_size : int
        Font size for element labels
    element_label_color : str
        Color for element labels
    show_electrides : bool
        Whether to show electride positions
    electride_sites : List[ElectrideSite], optional
        Electride sites to plot
    electride_symbol : str
        Marker style for electride sites (matplotlib marker)
    electride_symbol_size : int
        Marker size for electride sites
    electride_symbol_color : str
        Color for electride markers
    electride_label : str
        Label text for electride sites
    electride_label_size : int
        Font size for electride labels
    electride_label_color : str
        Color for electride labels
    cmap : str
        Colormap name
    boundary_color : str
        Color for partition boundaries
    boundary_width : float
        Line width for boundaries
    slice_tolerance : float
        Fractional tolerance for including atoms near the slice
    equal_aspect : bool
        If True, use equal aspect ratio with real dimensions

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    from scipy.ndimage import zoom as scipy_zoom

    ngx, ngy, ngz = elf_data.ngrid
    grid = elf_data.grid
    lattice = elf_data.lattice

    # Calculate cell dimensions
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])

    # Default atom colors
    default_atom_colors = {
        'La': '#4169E1',  # Royal blue
        'Mg': '#32CD32',  # Lime green
        'H': '#FF6347',   # Tomato red
    }
    if atom_symbol_color is not None:
        default_atom_colors.update(atom_symbol_color)
    atom_colors = default_atom_colors

    # Determine slice
    if plane == 'xy':
        if index is None:
            index = int((position or 0.5) * ngz)
        slice_data = grid[:, :, index].T
        slice_labels = labels[:, :, index].T
        coord_indices = (0, 1, 2)
        dim_x, dim_y = a, b
    elif plane == 'xz':
        if index is None:
            index = int((position or 0.5) * ngy)
        slice_data = grid[:, index, :].T
        slice_labels = labels[:, index, :].T
        coord_indices = (0, 2, 1)
        dim_x, dim_y = a, c
    elif plane == 'yz':
        if index is None:
            index = int((position or 0.5) * ngx)
        slice_data = grid[index, :, :].T
        slice_labels = labels[index, :, :].T
        coord_indices = (1, 2, 0)
        dim_x, dim_y = b, c
    else:
        raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    # Set extent and labels based on coordinate system
    if coord_system == 'cartesian':
        extent = [0, dim_x, 0, dim_y]
        axis_labels = {
            'xy': ('x (Å)', 'y (Å)'),
            'xz': ('x (Å)', 'z (Å)'),
            'yz': ('y (Å)', 'z (Å)')
        }
    else:  # fractional
        extent = [0, 1, 0, 1]
        axis_labels = {
            'xy': ('Fractional x', 'Fractional y'),
            'xz': ('Fractional x', 'Fractional z'),
            'yz': ('Fractional y', 'Fractional z')
        }
    xlabel, ylabel = axis_labels[plane]

    # Interpolate for higher resolution
    if interpolate and interpolate_factor > 1:
        slice_data = scipy_zoom(slice_data, interpolate_factor, order=3)
        # Use nearest neighbor for labels to preserve boundaries
        slice_labels = scipy_zoom(slice_labels.astype(float), interpolate_factor, order=0).astype(int)

    # Auto-calculate figsize for equal aspect
    if figsize is None:
        base_size = 8
        if coord_system == 'cartesian':
            aspect = dim_y / dim_x
        else:
            aspect = 1.0
        figsize = (base_size, base_size * aspect + 1)  # +1 for colorbar

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ELF with interpolation
    im = ax.imshow(
        slice_data, origin='lower', extent=extent,
        cmap=cmap, vmin=0, vmax=1,
        aspect='equal' if equal_aspect else 'auto',
        interpolation='bicubic' if interpolate else 'nearest'
    )
    plt.colorbar(im, ax=ax, label='ELF', shrink=0.8)

    # Plot partition boundaries
    # Find edges where labels change
    edges_x = np.abs(np.diff(slice_labels, axis=1)) > 0
    edges_y = np.abs(np.diff(slice_labels, axis=0)) > 0

    # Create boundary mask
    boundary_mask = np.zeros_like(slice_labels, dtype=bool)
    boundary_mask[:, :-1] |= edges_x
    boundary_mask[:, 1:] |= edges_x
    boundary_mask[:-1, :] |= edges_y
    boundary_mask[1:, :] |= edges_y

    # Overlay boundaries
    boundary_rgba = np.zeros((*slice_labels.shape, 4))
    boundary_rgba[boundary_mask] = mcolors.to_rgba(boundary_color, alpha=0.8)
    ax.imshow(
        boundary_rgba, origin='lower', extent=extent,
        aspect='equal' if equal_aspect else 'auto'
    )

    # Show atoms
    frac_coords = elf_data.frac_coords
    species_list = elf_data.get_species_list()
    slice_pos = index / [ngx, ngy, ngz][coord_indices[2]]

    if show_atoms:
        for i, (coord, sp) in enumerate(zip(frac_coords, species_list)):
            if abs(coord[coord_indices[2]] - slice_pos) < slice_tolerance:
                # Get position based on coordinate system
                if coord_system == 'cartesian':
                    plot_x = coord[coord_indices[0]] * dim_x
                    plot_y = coord[coord_indices[1]] * dim_y
                else:
                    plot_x = coord[coord_indices[0]]
                    plot_y = coord[coord_indices[1]]

                # Show atom symbol
                if show_atom_symbol:
                    color = atom_colors.get(sp, 'gray')
                    ax.scatter(
                        plot_x, plot_y, s=atom_symbol_size, marker=atom_symbol,
                        c=color, edgecolors='black', linewidths=1.5, zorder=10
                    )

                # Show element label
                if show_element_label:
                    ax.annotate(
                        sp, (plot_x, plot_y),
                        fontsize=element_label_size, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', color=element_label_color
                    )

    # Show electrides
    if show_electrides and electride_sites:
        for site in electride_sites:
            coord = site.frac_coord
            if abs(coord[coord_indices[2]] - slice_pos) < slice_tolerance:
                # Get position based on coordinate system
                if coord_system == 'cartesian':
                    plot_x = coord[coord_indices[0]] * dim_x
                    plot_y = coord[coord_indices[1]] * dim_y
                else:
                    plot_x = coord[coord_indices[0]]
                    plot_y = coord[coord_indices[1]]

                ax.scatter(
                    plot_x, plot_y, s=electride_symbol_size, marker=electride_symbol,
                    c=electride_symbol_color, edgecolors='black', linewidths=1.5, zorder=11
                )
                ax.annotate(
                    electride_label, (plot_x, plot_y),
                    fontsize=electride_label_size, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 10),
                    textcoords='offset points', color=electride_label_color
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'ELF slice with partition ({plane} plane, z={slice_pos:.2f})')

    plt.tight_layout()
    return fig


def plot_elf_slice(
    elf_data: GridData,
    plane: str = 'xy',
    index: Optional[int] = None,
    position: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_atoms: bool = True,
    show_electrides: bool = True,
    electride_sites: Optional[List[ElectrideSite]] = None,
    cmap: str = 'RdYlBu_r'
) -> Any:
    """
    Plot a 2D slice of the ELF.

    Parameters
    ----------
    elf_data : GridData
        ELF grid data
    plane : str
        Plane to slice ('xy', 'xz', or 'yz')
    index : int, optional
        Grid index for the slice (if None, uses position)
    position : float, optional
        Fractional position for the slice (0-1)
    figsize : tuple
        Figure size
    show_atoms : bool
        Whether to show atom positions
    show_electrides : bool
        Whether to show electride positions
    electride_sites : List[ElectrideSite], optional
        Electride sites to plot
    cmap : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    ngx, ngy, ngz = elf_data.ngrid
    grid = elf_data.grid

    # Determine slice
    if plane == 'xy':
        if index is None:
            index = int((position or 0.5) * ngz)
        slice_data = grid[:, :, index].T
        xlabel, ylabel = 'x', 'y'
        extent = [0, 1, 0, 1]
        coord_indices = (0, 1, 2)
    elif plane == 'xz':
        if index is None:
            index = int((position or 0.5) * ngy)
        slice_data = grid[:, index, :].T
        xlabel, ylabel = 'x', 'z'
        extent = [0, 1, 0, 1]
        coord_indices = (0, 2, 1)
    elif plane == 'yz':
        if index is None:
            index = int((position or 0.5) * ngx)
        slice_data = grid[index, :, :].T
        xlabel, ylabel = 'y', 'z'
        extent = [0, 1, 0, 1]
        coord_indices = (1, 2, 0)
    else:
        raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ELF
    im = ax.imshow(slice_data, origin='lower', extent=extent,
                   cmap=cmap, vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='ELF')

    # Show atoms
    if show_atoms:
        frac_coords = elf_data.frac_coords
        species_list = elf_data.get_species_list()
        slice_pos = index / [ngx, ngy, ngz][coord_indices[2]]

        # Tolerance for atoms near the slice plane
        tol = 0.1

        for i, (coord, sp) in enumerate(zip(frac_coords, species_list)):
            if abs(coord[coord_indices[2]] - slice_pos) < tol:
                ax.scatter(coord[coord_indices[0]], coord[coord_indices[1]],
                          s=100, marker='o', edgecolors='black', linewidths=1,
                          c='white', alpha=0.8)
                ax.annotate(sp, (coord[coord_indices[0]], coord[coord_indices[1]]),
                           fontsize=8, ha='center', va='bottom')

    # Show electrides
    if show_electrides and electride_sites:
        slice_pos = index / [ngx, ngy, ngz][coord_indices[2]]
        tol = 0.1

        for site in electride_sites:
            coord = site.frac_coord
            if abs(coord[coord_indices[2]] - slice_pos) < tol:
                ax.scatter(coord[coord_indices[0]], coord[coord_indices[1]],
                          s=150, marker='*', c='yellow', edgecolors='black',
                          linewidths=1, zorder=10)

    ax.set_xlabel(f'Fractional {xlabel}')
    ax.set_ylabel(f'Fractional {ylabel}')
    ax.set_title(f'ELF slice ({plane} plane, index={index})')

    return fig
