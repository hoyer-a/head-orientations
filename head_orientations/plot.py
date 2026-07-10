import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
import numpy as np
import pyfar as pf

from .head_orientation_class import HeadOrientations
from .metrics import HeadOrientationsMetrics
from .utils import spectral_difference, mean_spectral_difference

from typing import Sequence, Union


def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions into finite polygons.
    Adapted from SciPy cookbook.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram.
    radius : float, optional
        Radius for bounding infinite regions. If None, uses 2x the extent
        of the points.

    Returns
    -------
    new_regions : list
        List of polygon regions (vertex indices).
    new_vertices : ndarray
        Array of vertex coordinates.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # map ridge vertices to ridges
    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(
            vor.ridge_points,
            vor.ridge_vertices):

        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # reconstruct infinite regions
    for p1, region_idx in enumerate(vor.point_region):

        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]

        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:

            if v1 >= 0 and v2 >= 0:
                continue

            # compute missing endpoint
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)

            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)

            direction = np.sign(
                np.dot(midpoint - center, normal)
            ) * normal

            far_point = vor.vertices[
                v1 if v1 >= 0 else v2
            ] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # sort polygon vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])

        centroid = vs.mean(axis=0)

        angles = np.arctan2(
            vs[:, 1] - centroid[1],
            vs[:, 0] - centroid[0]
        )

        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def subplot_spectral_difference(head_orientations, reference,
                                plane='horizontal', ear='left',
                                db_threshold=None, limits=None,
                                sort=False):
    """
    Plot spectral difference for multiple head orientations against one
    reference head orientation.

    Parameters
    ----------

    head_orientations : HeadOrientations
        Object containing one or more head orientations to be evaluated.
    reference : HeadOrientations
        Reference object containing exactly one head orientation.
    plane : string
        Plane to be plotted against frequency, can be 'median', 'frontal' or
        'horizontal'.
    db_threshold : float, None
        Threshold for plot. Differences below this value will be set to 0.
        Default is 1.
    sort : bool, optional
        If ``True``, sort plotted orientations by
        ``(bend, elevation, azimuth)``.

    """
    if ear == 'left':
        ear_id = 0
    elif ear == 'right':
        ear_id = 1
    else:
        raise ValueError("ear must be 'left' or 'right'.")

    if limits:
        vmin = limits[0]
        vmax = limits[1]
    else:
        vmin = None
        vmax = None

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_white_red',
        ['blue', 'white', 'red'],
        N=256,
    )

    # coords in plane
    angles = np.arange(0, 2 * np.pi, 2 / 180 * np.pi)

    if plane == 'median':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(0, angles, radius=1)
    elif plane == 'frontal':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(
                np.pi / 2,
                angles,
                radius=1,
            )
    elif plane == 'horizontal':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(angles, 0, radius=1)
    else:
        raise ValueError("plane must be 'median', 'frontal', or 'horizontal'.")

    if reference.n_orientations != 1:
        raise ValueError(
            "reference must contain exactly one head orientation."
        )

    reference_hrirs = reference.hrirs[0, :]
    orientation_values = head_orientations.head_orientations
    plot_indices = np.arange(head_orientations.n_orientations)

    idx_ref = reference.source_positions.find_nearest(coords2find)[0]
    idx_ho = head_orientations.source_positions.find_nearest(coords2find)[0]

    reference_hrirs = reference_hrirs[idx_ref]

    if sort:
        sort_order = np.lexsort(
            (
                orientation_values[:, 2],
                orientation_values[:, 1],
                orientation_values[:, 0],
            )
        )
        plot_indices = plot_indices[sort_order]

    # prepare plot
    n_files = plot_indices.size
    columns = 2
    rows = int(np.ceil(n_files / columns))

    fig = plt.figure(figsize=(10, 4 * rows), dpi=300)
    fig.suptitle(f"{plane} plane\n{ear} ear")
    gs = GridSpec(rows, columns)

    # create subplots
    for i, plot_idx in enumerate(plot_indices):
        b, e, a = orientation_values[plot_idx]
        hrirs = head_orientations.hrirs[plot_idx, :]

        hrirs = hrirs[idx_ho]

        print(hrirs)
        print(reference_hrirs)

        spec_diff = spectral_difference(hrirs, reference_hrirs)

        print(spectral_difference)

        if db_threshold:
            db, prefix = \
                pf.dsp.decibel(spec_diff, return_prefix=True)

            idx_threshold = np.where(
                (db > -db_threshold) & (db < db_threshold)
            )
            db[idx_threshold] = 0
            spec_diff = \
                pf.FrequencyData(10 ** (db / prefix), hrirs.frequencies)

        row = i // columns
        col = i % columns

        ax = fig.add_subplot(gs[row, col])
        pf.plot.freq_2d(
            spec_diff[:, ear_id].flatten(),
            ax=ax,
            indices=np.rad2deg(angles),
            orientation='horizontal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"bend: {b}; elev: {e}, azimuth: {a}")
        ax.set_ylabel('angle in degree')
        # ax.set_xlim(1e3, 6e3)
        # ax.set_xscale('linear')
    plt.tight_layout()
    plt.show()


def plot_single_spectral_difference(ho1: HeadOrientations,
                                    ho2: HeadOrientations = None,
                                    plane: str = 'horizontal',
                                    ear: str = 'left',
                                    db_threshold: float | None = None,
                                    limits: Sequence[float] | None = None):
    """"""

    if ear == 'left':
        ear_id = 0
    elif ear == 'right':
        ear_id = 1
    else:
        raise ValueError("ear must be 'left' or 'right'.")

    if limits:
        vmin = limits[0]
        vmax = limits[1]
    else:
        vmin = None
        vmax = None

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_white_red',
        ['blue', 'white', 'red'],
        N=256,
    )

    # coords in plane
    angles = np.arange(0, 2 * np.pi, 2 / 180 * np.pi)

    if plane == 'median':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(0, angles, radius=1)
    elif plane == 'frontal':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(
                np.pi / 2,
                angles,
                radius=1,
            )
    elif plane == 'horizontal':
        coords2find = \
            pf.Coordinates.from_spherical_elevation(angles, 0, radius=1)
    else:
        raise ValueError("plane must be 'median', 'frontal', or 'horizontal'.")

    hrirs1 = ho1.hrirs
    idx_1 = ho1.source_positions.find_nearest(coords2find)[0]
    hrirs1 = hrirs1[:, *idx_1]

    if ho2:
        hrirs2 = ho2.hrirs
        idx_2 = ho2.source_positions.find_nearest(coords2find)[0]
        hrirs2 = hrirs2[:, *idx_2]

        # plot
        spec_diff = spectral_difference(hrirs1, hrirs2)

        if db_threshold:
            db, prefix = \
                pf.dsp.decibel(spec_diff, return_prefix=True)

            idx_threshold = np.where(
                (db > -db_threshold) & (db < db_threshold)
            )
            db[idx_threshold] = 0
            spec_diff = \
                pf.FrequencyData(10 ** (db / prefix), hrirs1.frequencies)

        data = spec_diff[0, ...]
    else:
        data = hrirs1[0, ...]

    ax = pf.plot.freq_2d(
            data[:, ear_id].flatten(),
            indices=np.rad2deg(angles),
            orientation='horizontal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )[0]

    b, e, a = ho1.head_orientations[0]
    ax[0].set_ylabel('angle in degree')
    ax[0].set_title(f"bend: {b}; elev: {e}, azimuth: {a}")
    plt.show()


def plot_mean_spectral_difference(head_orientation: HeadOrientations,
                                  reference: HeadOrientations,
                                  ear: str = "left",
                                  average_type: str = 'mean_db',
                                  limits_db: Sequence = None):
    """"""
    if head_orientation.n_orientations !=1:
        raise ValueError("Single head orientation must be passed, but "
                         f"{head_orientation.n_orientations} were passed.")

    if ear == "left":
        ear_id = 0
    elif ear == "right":
        ear_id = 1
    else:
        raise ValueError("ear must be 'left' or 'right'")

    hrirs = head_orientation.hrirs
    hrirs_ref = reference.hrirs

    mean_sdif = mean_spectral_difference(hrirs, hrirs_ref)[0, :, ear_id]

    source = head_orientation.source_positions

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_red",
        ["blue", "white", "red"],
        N=256)

    fig, ax = plt.subplots(constrained_layout=True,
                           subplot_kw={"projection": "mollweide"})
    contour = _plot_sd_source_map(ax, source, mean_sdif, cmap, limits_db)

    # add a simple horizontal colorbar
    if contour is not None:
        cbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="horizontal",
            fraction=0.04,
            pad=0.05,
        )
        cbar.set_label("Spectral difference in dB")

    plt.show()


def _plot_sd_source_map(ax, source, values, cmap, limits_db):
    """"""
    azimuth = np.asarray(source.azimuth).squeeze()
    elevation = np.asarray(source.elevation).squeeze()

    # Wrap longitudes to [-pi, pi] for geographic Mollweide coordinates.
    azimuth = ((azimuth + np.pi) % (2 * np.pi)) - np.pi

    if limits_db is None:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
    else:
        vmin, vmax = limits_db

    triangulation = mtri.Triangulation(azimuth, elevation)
    contour = ax.tricontourf(
        triangulation,
        values,
        levels=21,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    xticks_deg = [-150, -90, -30, 30, 90, 150]
    xticks_rad = np.deg2rad(xticks_deg)
    yticks_deg = [-60, -30, 0, 30, 60]
    yticks_rad = np.deg2rad(yticks_deg)

    ax.set_xticks(xticks_rad)
    ax.set_xticklabels([f"{tick}°" for tick in xticks_deg])
    ax.set_yticks(yticks_rad)
    ax.set_yticklabels([f"{tick}°" for tick in yticks_deg])
    ax.grid(True, alpha=0.2, lw=0.4)
    return contour


def plot_localization_1_dof(
    metrics: Union[HeadOrientationsMetrics, Sequence[HeadOrientationsMetrics]],
    metric: str,
    limits: Sequence = None,
    ax=None,
    label: str = None,
    marker: str = 'x',
    color=None,
    **kwargs):
    """
    Plot localization metrics over a single rotational axis.

    Parameters
    ----------
    metrics : HeadOrientationsMetrics or list thereof
        Metrics to plot. When a list is passed, each entry is plotted as a
        separate series.
    metric : str
        Name of the metric attribute to plot.
    limits : Sequence, optional
        Y-axis limits as ``[ymin, ymax]``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If ``None``, the current axes (``plt.gca()``) are
        used, so successive calls automatically share the same plot.
    label : str, optional
        Label for the plotted series. Overrides ``m.comment`` when provided.
        Only used when ``metrics`` is a single ``HeadOrientationsMetrics``
        instance; for lists the individual ``comment`` attributes are used.
    marker : str, optional
        Matplotlib marker style. Default is ``'x'``.
    color : color or list of colors, optional
        Matplotlib color or sequence of colors, one per series. If ``None``,
        the default color cycle is used.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the data was plotted on.
    """
    if ax is None:
        ax = plt.gca()

    # Normalize to a list
    if isinstance(metrics, HeadOrientationsMetrics):
        metrics_list = [metrics]
        labels = [label if label is not None else metrics.comment]
    else:
        metrics_list = list(metrics)
        labels = [m.comment for m in metrics_list]

    # Normalize colors to a per-series list
    if color is None:
        colors = [None] * len(metrics_list)
    elif isinstance(color, (list, tuple)) and len(color) == len(metrics_list):
        colors = list(color)
    else:
        colors = [color] * len(metrics_list)

    for m, lbl, c in zip(metrics_list, labels, colors):
        head_orientations = m.head_orientations

        bend = head_orientations[:, 0]
        elev = head_orientations[:, 1]
        azim = head_orientations[:, 2]

        is_zero = [np.all(bend == 0), np.all(elev == 0), np.all(azim == 0)]

        if sum(is_zero) != 2:
            raise ValueError("Can only plot metrics for head orientations in one"
                             "rotational axis, e.g. bend=0 and elev=0 for all orientations")

        values = m.__getattribute__(metric)

        axis_idx = np.where(~np.array(is_zero))[0]
        angles = head_orientations[:, axis_idx]

        plot_kwargs = dict(marker=marker, ls='', label=lbl, **kwargs)
        if c is not None:
            plot_kwargs['color'] = c
        ax.plot(angles, values, **plot_kwargs)

    ax.set_ylim(limits)
    ax.set_ylabel(metric)
    ax.grid(True)

    return ax


def plot_localization_scatter(
        metrics: HeadOrientationsMetrics,
        rotation: float = None,
        metric: str = None):
    """
    Plot localization metric as Voronoi cells over
    lateral bend / flexion-extension space.
    """
    metric_ = getattr(metrics, metric)
    rotation = np.atleast_1d(rotation)

    for rotation_ in rotation:

        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]
        colors = metric_[mask]
        fig, ax = plt.subplots(figsize=(7, 6))

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c=colors,
            s=10,
            zorder=10
        )


        ax.set_title(f'{rotation_}° rotation')
        ax.set_xlabel('Lateral Bend [°]')
        ax.set_ylabel('Extension/Flexion [°]')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()


def plot_localization_map(
        metrics: HeadOrientationsMetrics,
        reference: HeadOrientationsMetrics = None,
        rotation: float = None,
        metric: str = None,
        limits: Sequence = None,
        cmap: str = 'blue_white_red',
        rom_fill: Sequence = None,
        rom_fill_rotation: Union[float, Sequence] = None):
    """
    Plot localization metric as a Voronoi cell map over
    lateral bend / flexion-extension space.

    Parameters
    ----------
    metrics : HeadOrientationsMetrics
        Metrics object containing head orientations and metric data.
    rotation : float, optional
        Rotation angle(s) to filter by. Can be a scalar or array.
    metric : str
        Name of the metric attribute to plot (e.g., 'querr', 'pe_raw').
    limits : Sequence, optional
        Value range [vmin, vmax] for the colormap. If None, uses data min/max.
    rom_fill : Sequence, optional
        Tuple of (bend, flex) arrays defining the ROM region to fill.
    rom_fill_rotation : float, tuple, optional
        Rotation angle(s) for which to plot the ROM fill. Can be a scalar or tuple.
        If None, ROM fill is plotted for all rotations.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    metric_ = getattr(metrics, metric)
    if reference:
        metric_ref = getattr(reference, metric)
    rotation = np.atleast_1d(rotation)

    # Convert rom_fill_rotation to array for consistent handling
    if rom_fill_rotation is not None:
        rom_fill_rotation = np.atleast_1d(rom_fill_rotation)

    if cmap and cmap == "blue_white_red":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'blue_white_red',
            ['blue', 'white', 'red'],
            N=256)

    # -------------------------------------------------------------------------

    for rotation_ in rotation:
        if reference and not np.array_equal(metrics.head_orientations,
                                            reference.head_orientations):
            raise ValueError('Reference and metrics must contain same head'
                             'orientations')

        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]

        if reference:
            colors = metric_[mask] - metric_ref[mask]
        else:
            colors = metric_[mask]

        points = np.column_stack((lateral_bend, flexex))

        # compute Voronoi tessellation
        vor = Voronoi(points)

        # reconstruct finite polygons
        regions, vertices = _voronoi_finite_polygons_2d(vor)

        fig, ax = plt.subplots(figsize=(7, 6))

        patches = []

        for region in regions:

            polygon = vertices[region]

            patches.append(
                Polygon(polygon, closed=True)
            )

        if limits is None:
            vmin = np.nanmin(colors)
            vmax = np.nanmax(colors)
        else:
            vmin, vmax = limits

        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )

        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            linewidth=0.5
        )

        collection.set_array(colors)

        ax.add_collection(collection)

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c='black',
            s=10,
            zorder=10
        )

        margin = 5

        ax.set_xlim(
            lateral_bend.min() - margin,
            lateral_bend.max() + margin
        )

        ax.set_ylim(
            flexex.min() - margin,
            flexex.max() + margin
        )

        # Plot ROM fill only if this rotation is in rom_fill_rotation
        if rom_fill and (rom_fill_rotation is None or rotation_ in rom_fill_rotation):
            bend = rom_fill[0]
            flex = rom_fill[1]
            ax.fill(bend, -flex, color='none', edgecolor='k')

        ax.set_title(f'{rotation_}° rotation')
        ax.set_xlabel('Lateral Bend [°]')
        ax.set_ylabel('Extension/Flexion [°]')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

        cbar = fig.colorbar(collection, ax=ax)
        cbar.set_label(metric)

        plt.tight_layout()
        plt.show()


def plot_localization_map_subplots(
        metrics: HeadOrientationsMetrics,
        reference: HeadOrientationsMetrics = None,
        rotation: float = None,
        metric: str = None,
        limits: Sequence = None,
        cmap: str = 'blue_white_red',
        rom_fill: Sequence = None,
        rom_fill_rotation: Union[float, Sequence] = None,
        cols: int = 4,
        figsize: Sequence = None):
    """
    Plot localization metric as Voronoi cell maps in a grid of subplots,
    one subplot per rotation value.

    Parameters
    ----------
    metrics : HeadOrientationsMetrics
        Metrics object containing head orientations and metric data.
    reference : HeadOrientationsMetrics, optional
        Reference metrics to compute difference against. If provided,
        colors will represent metric_ - metric_ref.
    rotation : float or array-like, optional
        Rotation angle(s) to plot. If None, all unique rotation values are plotted.
    metric : str
        Name of the metric attribute to plot (e.g., 'querr', 'pe_raw').
    limits : Sequence, optional
        Value range [vmin, vmax] for the colormap. If None, uses data min/max.
    cmap : str, optional
        Colormap name. Default is 'blue_white_red'.
    rom_fill : Sequence, optional
        Tuple of (bend, flex) arrays defining the ROM region to fill.
    rom_fill_rotation : float or Sequence, optional
        Rotation angle(s) for which to plot the ROM fill. If None, plots for all.
    cols : int, optional
        Number of columns in the subplot grid. Default is 4.
    figsize : Sequence, optional
        Figure size as (width, height). If None, auto-calculated based on grid size.
    """
    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    metric_ = getattr(metrics, metric)
    if reference:
        metric_ref = getattr(reference, metric)

    # Get all unique rotation values if not specified
    if rotation is None:
        rotation = np.unique(metrics.head_orientations[:, 2])
    else:
        rotation = np.atleast_1d(rotation)

    # Convert rom_fill_rotation to array for consistent handling
    if rom_fill_rotation is not None:
        rom_fill_rotation = np.atleast_1d(rom_fill_rotation)

    if cmap and cmap == "blue_white_red":
        cmap_obj = mcolors.LinearSegmentedColormap.from_list(
            'blue_white_red',
            ['blue', 'white', 'red'],
            N=256)
    else:
        cmap_obj = cmap

    # Calculate number of rows needed
    n_plots = len(rotation)
    rows = int(np.ceil(n_plots / cols))

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.15)

    # Collect all colors for global normalization if needed
    all_colors = []
    plot_data = []

    if reference and not np.array_equal(metrics.head_orientations,
                                        reference.head_orientations):
        raise ValueError('Reference and metrics must contain same head'
                         'orientations')

    # First pass: collect data for all rotations
    for rotation_ in rotation:
        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]

        if reference:
            colors = metric_[mask] - metric_ref[mask]
        else:
            colors = metric_[mask]

        all_colors.extend(colors)

        points = np.column_stack((lateral_bend, flexex))
        vor = Voronoi(points)
        regions, vertices = _voronoi_finite_polygons_2d(vor)

        plot_data.append({
            'rotation': rotation_,
            'lateral_bend': lateral_bend,
            'flexex': flexex,
            'colors': colors,
            'regions': regions,
            'vertices': vertices,
        })

    # Determine global color limits
    if limits is None:
        vmin = np.nanmin(all_colors)
        vmax = np.nanmax(all_colors)
    else:
        vmin, vmax = limits

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Second pass: create subplots
    for idx, data in enumerate(plot_data):
        row = idx // cols
        col = idx % cols

        ax = fig.add_subplot(gs[row, col])

        rotation_ = data['rotation']
        lateral_bend = data['lateral_bend']
        flexex = data['flexex']
        colors = data['colors']
        regions = data['regions']
        vertices = data['vertices']

        patches = []
        for region in regions:
            polygon = vertices[region]
            patches.append(Polygon(polygon, closed=True))

        collection = PatchCollection(
            patches,
            cmap=cmap_obj,
            norm=norm,
            edgecolor='black',
            linewidth=0.5
        )
        collection.set_array(colors)
        ax.add_collection(collection)

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c='black',
            s=10,
            zorder=10
        )

        margin = 5
        ax.set_xlim(
            lateral_bend.min() - margin,
            lateral_bend.max() + margin
        )
        ax.set_ylim(
            flexex.min() - margin,
            flexex.max() + margin
        )

        # Plot ROM fill only if this rotation is in rom_fill_rotation
        if rom_fill and (rom_fill_rotation is None or rotation_ in rom_fill_rotation):
            bend = rom_fill[0]
            flex = rom_fill[1]
            ax.fill(bend, -flex, color='none', edgecolor='k')

        ax.set_title(f'{rotation_}° rotation', fontsize=11, fontweight='bold')
        ax.set_xlabel('Lateral Bend [°]', fontsize=10)
        ax.set_ylabel('Extension/Flexion [°]', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

    # Add colorbar to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        cax=cbar_ax
    )
    cbar.set_label(metric, fontsize=10)

    plt.show()


def _aggregate_pbc(pbc, aggregate):
    """Aggregate pbc values across source positions per head orientation."""
    if pbc.ndim == 1:
        return pbc
    if aggregate == 'mean':
        return np.mean(pbc, axis=1)
    elif aggregate == 'median':
        return np.median(pbc, axis=1)
    elif aggregate == 'upper_95':
        return np.percentile(pbc, 95, axis=1)
    elif aggregate == 'lower_95':
        return np.percentile(pbc, 5, axis=1)
    else:
        raise ValueError(
            "aggregate must be 'mean', 'median', 'upper_95', or 'lower_95'."
        )


def _add_coloration_annotations(ax, pbc, colors_all, annotate):
    """Add a text box with statistics over all head orientations to an axes."""
    if not annotate:
        return
    lines = []
    for ann in annotate:
        if ann == 'median':
            val = np.median(colors_all)
            lines.append(f'median: {val:.2f}')
        elif ann == 'mean':
            val = np.mean(colors_all)
            lines.append(f'mean: {val:.2f}')
        elif ann == 'max_lower_95':
            if pbc.ndim > 1:
                val = np.max(np.percentile(pbc, 5, axis=1))
            else:
                val = np.max(pbc)
            lines.append(f'max lower 95: {val:.2f}')
        elif ann == 'min_upper_95':
            if pbc.ndim > 1:
                val = np.min(np.percentile(pbc, 95, axis=1))
            else:
                val = np.min(pbc)
            lines.append(f'min upper 95: {val:.2f}')
        else:
            raise ValueError(
                f"Unknown annotation '{ann}'. Must be one of: "
                "'median', 'mean', 'max_lower_95', 'min_upper_95'."
            )
    text = '\n'.join(lines)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )


def plot_coloration_map(
        metrics: HeadOrientationsMetrics,
        rotation: float = None,
        limits: Sequence = None,
        cmap: str = 'viridis',
        rom_fill: Sequence = None,
        rom_fill_rotation: Union[float, Sequence] = None,
        aggregate: str = 'mean',
        annotate: Sequence = None):
    """
    Plot coloration metric (pbc aggregated across source positions) as a Voronoi
    cell map over lateral bend / flexion-extension space.

    Parameters
    ----------
    metrics : HeadOrientationsMetrics
        Metrics object containing head orientations and pbc data per source position.
    rotation : float, optional
        Rotation angle(s) to filter by. Can be a scalar or array.
    limits : Sequence, optional
        Value range [vmin, vmax] for the colormap. If None, uses data min/max.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    rom_fill : Sequence, optional
        Tuple of (bend, flex) arrays defining the ROM region to fill.
    rom_fill_rotation : float, tuple, optional
        Rotation angle(s) for which to plot the ROM fill. Can be a scalar or tuple.
        If None, ROM fill is plotted for all rotations.
    aggregate : str, optional
        Aggregation method over source positions. One of ``'mean'`` (default),
        ``'median'``, ``'upper_95'`` (95th percentile), ``'lower_95'``
        (5th percentile).
    annotate : list of str, optional
        Statistics to annotate on each subplot, computed over all head
        orientations. Supported values: ``'median'``, ``'mean'``,
        ``'max_lower_95'`` (max of lower 5th percentile per orientation),
        ``'min_upper_95'`` (min of upper 95th percentile per orientation).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    # Get pbc data and aggregate across source positions
    pbc = metrics.pbc  # Shape: (n_orientations, n_source_positions)
    colors_all = _aggregate_pbc(pbc, aggregate)

    rotation = np.atleast_1d(rotation)

    # Convert rom_fill_rotation to array for consistent handling
    if rom_fill_rotation is not None:
        rom_fill_rotation = np.atleast_1d(rom_fill_rotation)

    # Use specified colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # -------------------------------------------------------------------------

    for rotation_ in rotation:
        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]

        colors = colors_all[mask]

        points = np.column_stack((lateral_bend, flexex))

        # compute Voronoi tessellation
        vor = Voronoi(points)

        # reconstruct finite polygons
        regions, vertices = _voronoi_finite_polygons_2d(vor)

        fig, ax = plt.subplots(figsize=(7, 6))

        patches = []

        for region in regions:

            polygon = vertices[region]

            patches.append(
                Polygon(polygon, closed=True)
            )

        if limits is None:
            vmin = np.nanmin(colors)
            vmax = np.nanmax(colors)
        else:
            vmin, vmax = limits

        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )

        collection = PatchCollection(
            patches,
            cmap=cmap_obj,
            norm=norm,
            edgecolor='black',
            linewidth=0.5
        )

        collection.set_array(colors)

        ax.add_collection(collection)

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c='black',
            s=10,
            zorder=10
        )

        margin = 5

        ax.set_xlim(
            lateral_bend.min() - margin,
            lateral_bend.max() + margin
        )

        ax.set_ylim(
            flexex.min() - margin,
            flexex.max() + margin
        )

        # Plot ROM fill only if this rotation is in rom_fill_rotation
        if rom_fill and (rom_fill_rotation is None or rotation_ in rom_fill_rotation):
            bend = rom_fill[0]
            flex = rom_fill[1]
            ax.fill(bend, -flex, color='none', edgecolor='k')

        ax.set_title(f'{rotation_}° rotation')
        ax.set_xlabel('Lateral Bend [°]')
        ax.set_ylabel('Extension/Flexion [°]')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        _add_coloration_annotations(ax, pbc, colors_all, annotate)

        cbar = fig.colorbar(collection, ax=ax)
        cbar.set_label('Coloration (PBC)')

        plt.tight_layout()
        plt.show()


def plot_coloration_map_subplots(
        metrics: HeadOrientationsMetrics,
        rotation: float = None,
        limits: Sequence = None,
        cmap: str = 'viridis',
        rom_fill: Sequence = None,
        rom_fill_rotation: Union[float, Sequence] = None,
        cols: int = 4,
        figsize: Sequence = None,
        aggregate: str = 'mean',
        annotate: Sequence = None):
    """
    Plot coloration metric (pbc aggregated across source positions) as Voronoi cell
    maps in a grid of subplots, one subplot per rotation value.

    Parameters
    ----------
    metrics : HeadOrientationsMetrics
        Metrics object containing head orientations and pbc data per source position.
    rotation : float or array-like, optional
        Rotation angle(s) to plot. If None, all unique rotation values are plotted.
    limits : Sequence, optional
        Value range [vmin, vmax] for the colormap. If None, uses data min/max.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    rom_fill : Sequence, optional
        Tuple of (bend, flex) arrays defining the ROM region to fill.
    rom_fill_rotation : float or Sequence, optional
        Rotation angle(s) for which to plot the ROM fill. If None, plots for all.
    cols : int, optional
        Number of columns in the subplot grid. Default is 4.
    figsize : Sequence, optional
        Figure size as (width, height). If None, auto-calculated based on grid size.
    aggregate : str, optional
        Aggregation method over source positions. One of ``'mean'`` (default),
        ``'median'``, ``'upper_95'`` (95th percentile), ``'lower_95'``
        (5th percentile).
    annotate : list of str, optional
        Statistics to annotate on each subplot, computed over all head
        orientations. Supported values: ``'median'``, ``'mean'``,
        ``'max_lower_95'`` (max of lower 5th percentile per orientation),
        ``'min_upper_95'`` (min of upper 95th percentile per orientation).
    """
    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    # Get pbc data and aggregate across source positions
    pbc = metrics.pbc  # Shape: (n_orientations, n_source_positions)
    colors_all = _aggregate_pbc(pbc, aggregate)

    # Get all unique rotation values if not specified
    if rotation is None:
        rotation = np.unique(metrics.head_orientations[:, 2])
    else:
        rotation = np.atleast_1d(rotation)

    # Convert rom_fill_rotation to array for consistent handling
    if rom_fill_rotation is not None:
        rom_fill_rotation = np.atleast_1d(rom_fill_rotation)

    # Use specified colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Calculate number of rows needed
    n_plots = len(rotation)
    rows = int(np.ceil(n_plots / cols))

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (cols * 5, rows * 4.5)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.15)

    # Collect all colors for global normalization if needed
    all_colors = []
    plot_data = []

    # First pass: collect data for all rotations
    for rotation_ in rotation:
        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]

        colors = colors_all[mask]

        all_colors.extend(colors)

        points = np.column_stack((lateral_bend, flexex))
        vor = Voronoi(points)
        regions, vertices = _voronoi_finite_polygons_2d(vor)

        plot_data.append({
            'rotation': rotation_,
            'lateral_bend': lateral_bend,
            'flexex': flexex,
            'colors': colors,
            'regions': regions,
            'vertices': vertices,
        })

    # Determine global color limits
    if limits is None:
        vmin = np.nanmin(all_colors)
        vmax = np.nanmax(all_colors)
    else:
        vmin, vmax = limits

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Second pass: create subplots
    for idx, data in enumerate(plot_data):
        row = idx // cols
        col = idx % cols

        ax = fig.add_subplot(gs[row, col])

        rotation_ = data['rotation']
        lateral_bend = data['lateral_bend']
        flexex = data['flexex']
        colors = data['colors']
        regions = data['regions']
        vertices = data['vertices']

        patches = []
        for region in regions:
            polygon = vertices[region]
            patches.append(Polygon(polygon, closed=True))

        collection = PatchCollection(
            patches,
            cmap=cmap_obj,
            norm=norm,
            edgecolor='black',
            linewidth=0.5
        )
        collection.set_array(colors)
        ax.add_collection(collection)

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c='black',
            s=10,
            zorder=10
        )

        margin = 5
        ax.set_xlim(
            lateral_bend.min() - margin,
            lateral_bend.max() + margin
        )
        ax.set_ylim(
            flexex.min() - margin,
            flexex.max() + margin
        )

        # Plot ROM fill only if this rotation is in rom_fill_rotation
        if rom_fill and (rom_fill_rotation is None or rotation_ in rom_fill_rotation):
            bend = rom_fill[0]
            flex = rom_fill[1]
            ax.fill(bend, -flex, color='none', edgecolor='k')

        ax.set_title(f'{rotation_}° rotation', fontsize=11, fontweight='bold')
        ax.set_xlabel('Lateral Bend [°]', fontsize=10)
        ax.set_ylabel('Extension/Flexion [°]', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        _add_coloration_annotations(ax, pbc, colors_all, annotate)

    # Add colorbar to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        cax=cbar_ax
    )
    cbar.set_label('Coloration (PBC)', fontsize=10)

    plt.show()

