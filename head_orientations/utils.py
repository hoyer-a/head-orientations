import os
import re
import pyfar as pf
import numpy as np


def find_orientation_directory(base_dir=None, bend=None, elev=None, azim=None,
                               ignore_neutral=True):
    """Find orientation directories that match bend/elevation/azimuth filters.

    The function scans ``base_dir`` for subdirectory names containing the
    pattern ``bend<value>_elev<value>_azim<value>`` and returns matching names
    sorted by bend, elevation, and azimuth.

    Parameters
    ----------
    base_dir : str
        Path containing orientation subdirectories.
    bend : float, None, optional
        If given, only return directories with this bend angle.
    elev : float, None, optional
        If given, only return directories with this elevation angle.
    azim : float, None, optional
        If given, only return directories with this azimuth angle.
    ignore_neutral : bool, optional
        If ``True`` (default), directories with ``bend=0``, ``elev=0``, and
        ``azim=0`` are excluded.

    Returns
    -------
    dirs : list of str
        Sorted list of matching directory names.
    """
    pattern = re.compile(
        r"bend(?P<bend>-?\d+(?:\.\d+)?)_"
        r"elev(?P<elev>-?\d+(?:\.\d+)?)_"
        r"azim(?P<azim>-?\d+(?:\.\d+)?)"
        )
    subdirs = os.listdir(base_dir)
    dirs = []

    for subdir in subdirs:
        match = pattern.search(subdir)
        if not match:
            continue

        b = float(match.group("bend"))
        e = float(match.group("elev"))
        a = float(match.group("azim"))

        if ignore_neutral:
            if (b == 0 and e == 0 and a == 0):
                continue

        if (
            (bend is None or b == bend) and
            (elev is None or e == elev) and
            (azim is None or a == azim)
        ):
            dirs.append(subdir)

    dirs.sort(
        key=lambda d: (
            float(pattern.search(d).group("bend")),
            float(pattern.search(d).group("elev")),
            float(pattern.search(d).group("azim")),
        )
    )

    return dirs


def spectral_difference(sig1: pf.Signal, sig2: pf.Signal) -> pf.FrequencyData:
    r"""
    Calculate the log-magnitude spectral difference between two signals.

    The spectral difference is computed element-wise from the magnitude
    spectra of ``sig1`` and ``sig2`` as

    .. math::
        S(f) = 20\log_{10}\left(\frac{|H_1(f)|}{|H_2(f)|}\right)

    Parameters
    ----------
    sig1 : pyfar.Signal
        Signal in the numerator of the spectral ratio.
    sig2 : pyfar.Signal
        Signal in the denominator of the spectral ratio.

    Returns
    -------
    pyfar.FrequencyData
        Spectral difference in dB at the frequency bins of ``sig1``.

    Notes
    -----
    The calculation uses only spectral magnitudes and does not include phase.
    If ``sig2`` contains zero-magnitude bins, division can produce ``inf`` or
    ``nan`` values.

    """
    spec_diff = np.abs(sig1.freq_raw) / np.abs(sig2.freq_raw)
    spec_diff = pf.FrequencyData(spec_diff, sig1.frequencies)

    return spec_diff


def mean_spectral_difference(sig1: pf.Signal,
                             sig2: pf.Signal,
                             method: str = "mean_db"):
    """"""
    spec_diff = spectral_difference(sig1, sig2)
    db = pf.dsp.decibel(spec_diff)

    if method == "mean_db":
        mean_sdif = np.mean(db, axis=-1)
    else:
        raise NotImplementedError("Only mean_db is implemented.")

    return mean_sdif


def find_indices_in_region(head_orientations, bend, flex, tolerance=0.0, rotation=None):
    """
    Find indices of head orientations that lie within a flex/bend polygon region.

    Parameters
    ----------
    head_orientations : ndarray
        Array of shape (n, 3) where columns are [bend, elevation, azimuth]
    bend : ndarray
        Array defining the lateral bend boundary (x-coordinates of polygon)
    flex : ndarray
        Array defining the flexion/extension boundary (y-coordinates of polygon)
    tolerance : float, optional
        Tolerance in degrees. Points within this distance of the boundary
        are also considered inside. Default is 0.0
    rotation : float, tuple, or None, optional
        Azimuth rotation angle(s) to filter by. Can be a scalar or tuple of values.
        If None, all head orientations are considered. Default is None.

    Returns
    -------
    inside : ndarray
        Boolean array indicating which head orientations are in the region

    Examples
    --------
    >>> flex_range_max = [-15, 15]
    >>> bend_range_max = [-10, 10]
    >>> flex_range = flex_range_max[1] - flex_range_max[0]
    >>> gamma = np.deg2rad(np.arange(0, 361))
    >>> bend_boundary = bend_range_max[1] * np.sin(gamma)**2 * np.sign(np.sin(gamma))
    >>> flex_boundary = flex_range * ((1-np.cos(gamma)) / 2)**(5/4) - flex_range / 2

    >>> # Get all indices in region
    >>> mask = find_indices_in_region(head_orientations, bend_boundary, flex_boundary, tolerance=5.0)
    >>> indices = np.where(mask)[0]

    >>> # Get indices in region for specific rotation
    >>> mask = find_indices_in_region(head_orientations, bend_boundary, flex_boundary,
    ...                              tolerance=5.0, rotation=40)

    >>> # Get indices in region for multiple rotations
    >>> mask = find_indices_in_region(head_orientations, bend_boundary, flex_boundary,
    ...                              tolerance=5.0, rotation=(0, 10, 20))
    """
    from matplotlib.path import Path

    # Extract bend, flex, and azimuth from head orientations
    ho_bend = head_orientations[:, 0]
    ho_flex = head_orientations[:, 1]
    ho_azim = head_orientations[:, 2]

    # Create polygon vertices from bend and flex boundaries
    if not (isinstance(bend, np.ndarray) and isinstance(flex, np.ndarray)):
        bend = np.asarray(bend)
        flex = np.asarray(flex)

    if bend.shape[0] != flex.shape[0]:
        raise ValueError("bend and flex arrays must have the same length")

    # Create vertices by pairing bend and flex
    vertices = np.column_stack((bend, flex))

    # Create matplotlib Path for point-in-polygon test
    polygon_path = Path(vertices)

    # Stack points to test
    points = np.column_stack((ho_bend, ho_flex))

    # Test which points are inside
    inside = polygon_path.contains_points(points)

    if tolerance > 0:
        # For points outside, check if they're within tolerance distance
        inside_with_tolerance = polygon_path.contains_points(
            points, radius=tolerance
        )
        inside = inside | inside_with_tolerance

    # Filter by rotation if specified
    if rotation is not None:
        rotation = np.atleast_1d(rotation)
        rotation_mask = np.isin(ho_azim, rotation)
        inside = inside & rotation_mask

    return inside


def find_indices_in_bbox(head_orientations, bend_min, bend_max,
                        flex_min, flex_max, tolerance=0.0):
    """
    Find indices of head orientations within a rectangular bounding box.

    Parameters
    ----------
    head_orientations : ndarray
        Array of shape (n, 3) where columns are [bend, elevation, azimuth]
    bend_min, bend_max : float
        Lateral bend range
    flex_min, flex_max : float
        Flexion/extension range
    tolerance : float, optional
        Tolerance in degrees. Default is 0.0

    Returns
    -------
    indices : ndarray
        Boolean array indicating which head orientations are in the region
    """

    bend = head_orientations[:, 0]
    flexion = head_orientations[:, 1]

    inside = (
        (bend >= bend_min - tolerance) & (bend <= bend_max + tolerance) &
        (flexion >= flex_min - tolerance) & (flexion <= flex_max + tolerance)
    )

    return inside


