import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pyfar as pf

from .head_orientation_class import HeadOrientations
from .metrics import HeadOrientationsMetrics

from typing import Sequence, Union


def subplot_spectral_difference(reference, head_orientations,
                                plane='horizontal', ear='left',
                                db_threshold=None, limits=None,
                                sort=False):
    """
    Plot spectral difference for multiple head orientations against one
    reference head orientation.

    Parameters
    ----------

    reference : HeadOrientations
        Reference object containing exactly one head orientation.
    head_orientations : HeadOrientations
        Object containing one or more head orientations to be evaluated.
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

        spectral_difference = (
            np.abs(hrirs.freq_raw) / np.abs(reference_hrirs.freq_raw)
        )
        spectral_difference = pf.FrequencyData(
            spectral_difference, hrirs.frequencies
        )

        print(spectral_difference)

        if db_threshold:
            db, prefix = \
                pf.dsp.decibel(spectral_difference, return_prefix=True)

            idx_threshold = np.where(
                (db > -db_threshold) & (db < db_threshold)
            )
            db[idx_threshold] = 0
            spectral_difference = \
                pf.FrequencyData(10 ** (db / prefix), hrirs.frequencies)

        row = i // columns
        col = i % columns

        ax = fig.add_subplot(gs[row, col])
        pf.plot.freq_2d(
            spectral_difference[:, ear_id].flatten(),
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
    spectral_difference = (
        np.abs(hrirs1.freq_raw) / np.abs(hrirs2.freq_raw)
    )
    spectral_difference = pf.FrequencyData(
        spectral_difference, hrirs1.frequencies
    )

    if db_threshold:
        db, prefix = \
            pf.dsp.decibel(spectral_difference, return_prefix=True)

        idx_threshold = np.where(
            (db > -db_threshold) & (db < db_threshold)
        )
        db[idx_threshold] = 0
        spectral_difference = \
            pf.FrequencyData(10 ** (db / prefix), hrirs1.frequencies)

    spectral_difference = spectral_difference[0, ...]
    print(spectral_difference)
    print(angles.shape)

    ax = pf.plot.freq_2d(
            spectral_difference[:, ear_id].flatten(),
            indices=np.rad2deg(angles),
            orientation='horizontal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )[0]
    ax[0].set_ylabel('angle in degree')
    plt.show()


def plot_mean_signed_spectral_difference(head_orientation: HeadOrientations,
                                         ):
    if head_orientation.n_orientations !=1:
        raise ValueError("Single head orientation must be passed, but "
                         f"{head_orientation.n_orientations} were passed.")
    pass


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


def plot_localization_map(
        metrics: HeadOrientationsMetrics,
        rotation: float = None,
        metric: str = None):
    """
    Plot localization metric as Voronoi cells over
    lateral bend / flexion-extension space.
    """

    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    metric_ = getattr(metrics, metric)
    rotation = np.atleast_1d(rotation)

    for rotation_ in rotation:

        azi = metrics.head_orientations[:, 2]
        mask = (azi == rotation_)

        lateral_bend = metrics.head_orientations[mask, 0]
        flexex = metrics.head_orientations[mask, 1]
        colors = metric_[mask]

        points = np.column_stack((lateral_bend, flexex))

        # compute Voronoi tessellation
        vor = Voronoi(points)

        fig, ax = plt.subplots(figsize=(7, 6))

        patches = []
        patch_colors = []

        for point_idx, region_idx in enumerate(vor.point_region):

            region = vor.regions[region_idx]

            # skip infinite regions
            if -1 in region or len(region) == 0:
                continue

            polygon = [vor.vertices[i] for i in region]

            patches.append(Polygon(polygon, closed=True))
            patch_colors.append(colors[point_idx])

        norm = Normalize(
            vmin=np.nanmin(colors),
            vmax=np.nanmax(colors)
        )

        collection = PatchCollection(
            patches,
            cmap='viridis',
            norm=norm,
            edgecolor='black',
            linewidth=0.5
        )

        collection.set_array(np.array(patch_colors))

        ax.add_collection(collection)

        # overlay sample points
        ax.scatter(
            lateral_bend,
            flexex,
            c='black',
            s=10,
            zorder=10
        )

        ax.set_xlim(
            lateral_bend.min() - 5,
            lateral_bend.max() + 5
        )

        ax.set_ylim(
            flexex.min() - 5,
            flexex.max() + 5
        )

        ax.set_title(f'{rotation_}° rotation')
        ax.set_xlabel('Lateral Bend [°]')
        ax.set_ylabel('Flexion/Extension [°]')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)

        cbar = fig.colorbar(collection, ax=ax)
        cbar.set_label(metric)

        plt.tight_layout()
        plt.show()
