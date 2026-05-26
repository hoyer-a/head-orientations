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

    fig = plt.figure(figsize=(10, 4 * rows))
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
                                    ho2: HeadOrientations,
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
    hrirs2 = ho2.hrirs

    idx_1 = ho1.source_positions.find_nearest(coords2find)[0]
    idx_2 = ho2.source_positions.find_nearest(coords2find)[0]

    hrirs1 = hrirs1[:, *idx_1]
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

    spec_diff = spec_diff[0, ...]
    print(spec_diff)
    print(angles.shape)

    ax = pf.plot.freq_2d(
            spec_diff[:, ear_id].flatten(),
            indices=np.rad2deg(angles),
            orientation='horizontal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )[0]
    ax[0].set_ylabel('angle in degree')
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
    limits: Sequence = None):
    """
    Plot localization metrics over a single rotational axis.
    """
    # Normalize to a list
    if isinstance(metrics, HeadOrientationsMetrics):
        metrics = [metrics]
    for m in metrics:
        # your plotting logic here
        head_orientations = m.head_orientations

        bend = head_orientations[:, 0]
        elev = head_orientations[:, 1]
        azim = head_orientations[:, 2]

        is_zero = [np.all(bend == 0), np.all(elev == 0), np.all(azim == 0)]

        if sum(is_zero)!=2:
            raise ValueError("Can only plot metrics for head orientations in one" \
            "rotational axis, e.g. bend=0 and elev=0 for all orientations")

        values = m.__getattribute__(metric)

        axis_idx = np.where(~np.array(is_zero))[0]
        angles = head_orientations[:, axis_idx]

        plt.plot(angles, values, marker='x', ls='', label=m.comment)
        plt.ylim(limits)
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()

    return values


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
        ax.set_ylabel('Flexion/Extension [°]')

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
    # helper: reconstruct infinite Voronoi regions into finite polygons
    # adapted from SciPy cookbook
    # -------------------------------------------------------------------------

    def voronoi_finite_polygons_2d(vor, radius=None):

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
        regions, vertices = voronoi_finite_polygons_2d(vor)

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
            ax.fill(bend, -flex, color='k', alpha=0.125, edgecolor='none')

        ax.set_title(f'{rotation_}° rotation')
        ax.set_xlabel('Lateral Bend [°]')
        ax.set_ylabel('Flexion/Extension [°]')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

        cbar = fig.colorbar(collection, ax=ax)
        cbar.set_label(metric)

        plt.tight_layout()
        plt.show()
