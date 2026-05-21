import pyfar as pf
import spharpy
import numpy as np
from .head_orientation_class import HeadOrientations
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull


def interpolate_sh(head_orientations: HeadOrientations,
                   target_coordinates: pf.Coordinates,
                   n_max: int,
                   grid: str = "lebedev",
                   rotate: bool = True,
                   calculate_weights: bool = True):
    """
    Interpolate HRIRs in a global (torso-centered) coordinate system.

    The function projects each HRIR onto a spherical-harmonic (SH) basis,
    rotates the SH coefficients according to the corresponding head
    orientation, and synthesizes the HRIR at the requested target positions.

    Parameters
    ----------
    head_orientations : HeadOrientations
        Input container with HRIRs, source positions, and per-measurement
        head orientation angles (in degrees).
    target_coordinates : pf.Coordinates
        Positions where HRIRs should be reconstructed after SH rotation.
    n_max : int
        Maximum SH order used for analysis and synthesis.
    grid : str, optional
        Sampling strategy for SH analysis.
        - ``"lebedev"`` (default): map source positions to a 44-point
          Lebedev grid at the source radius.
        - ``None``: use the original source positions directly.
    rotate : bool, optional
        Applies the inverse head rotation to the SH-signal to rotate from
        head-centered to torso centered coordinates. Default is ``True``.

    Returns
    -------
    HeadOrientations
        A new `HeadOrientations` object containing interpolated HRIRs at
        `target_coordinates` and the original head-orientation metadata.
    """
    source = head_orientations.source_positions
    # Get grid for SH-transform
    radius = source.radius
    if grid == "lebedev":
        sampling = spharpy.samplings.lebedev(44, radius=radius[0])
        idx = source.find_nearest(sampling)[0]
        source = source[idx]
    elif grid is None:
        idx = slice(None)
        source = source[idx]
    else:
        raise ValueError("grid must be lebedev or None")

    # check radii of source and target sampling
    if not np.isclose(radius[0], target_coordinates.radius[0]):
        raise ValueError("Source and target samplings must have same radius.")

    # Get SamplingSphere for source and target coordinates
    sampling = spharpy.SamplingSphere.from_coordinates(source)
    if calculate_weights:
        sampling.weights = spharpy.samplings.calculate_sampling_weights(sampling)

    if type(target_coordinates) == spharpy.SamplingSphere:
        target_sampling = target_coordinates
    else:
        target_sampling = \
            spharpy.SamplingSphere.from_coordinates(target_coordinates)

    # Get basis matrix
    sh_definition = spharpy.SphericalHarmonicDefinition(n_max)
    y_nm = \
        spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                   sampling,
                                                   inverse_method='auto')
    hrirs = head_orientations.hrirs[:, *idx].copy()

    # Iterate over head orientations: sh-transform, rotate & interpolate
    for n, head_orientation in enumerate(head_orientations):
        hrir = hrirs[n, ...]
        hrir.time = hrir.time.squeeze()
        # time align
        hrir_onset = pf.dsp.resample(hrir, hrir.sampling_rate * 10,
                                     post_filter=True)
        onsets = pf.dsp.find_impulse_response_start(hrir_onset) / 10
        hrir = pf.dsp.fractional_time_shift(hrir, -onsets, mode='cyclic')

        toa_interpolator = LinearNDInterpolator(target_coordinates.cartesian,
                                                onsets)

        target_onsets = toa_interpolator(target_coordinates.cartesian)
        orientation = head_orientation.head_orientations
        hrir_nm = (y_nm.basis_inv @ hrir).T

        hrir_nm = spharpy.SphericalHarmonicSignal.from_definition(
            sh_definition, hrir_nm.time, hrir_nm.sampling_rate)

        orientation = np.deg2rad(orientation).squeeze()

        if rotate:
            Rotation = \
                spharpy.transforms.SphericalHarmonicRotation.from_euler(
                    'XYZ', [orientation[0], orientation[1], orientation[2]])

            rotated_nm = Rotation.apply(hrir_nm)
        else:
            rotated_nm = hrir_nm

        target_nm = \
            spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                       target_sampling,
                                                       inverse_method='auto')

        interpolated = pf.matrix_multiplication((target_nm.basis, rotated_nm),
                                                domain='time',
                                                axes=[(0, 1), (1, 0), (0, 1)])

        interpolated_signal = pf.Signal(interpolated, hrir.sampling_rate)
        if n == 0:
            output_hrir = interpolated_signal[None, :]
        else:
            output_hrir = \
                pf.utils.concatenate_channels((output_hrir,
                                              interpolated_signal[None, :]))
    # apply interpolated TOA
    output_hrir = pf.dsp.fractional_time_shift(output_hrir, target_onsets,
                                               mode='cyclic')

    return HeadOrientations(output_hrir, target_coordinates,
                            head_orientations.head_orientations, None)


def interpolate_barycentric(head_orientations: HeadOrientations,
                            target_coordinates: pf.Coordinates):
    pass


def interpolate_head_orientation(head_orientation_1: HeadOrientations,
                                 head_orientation_2: HeadOrientations,
                                 n_max: int | None = None,
                                 interpolation_grid: str = "lebedev",
                                 target_grid: str = "source"):
    """
    Interpolate between two head orientations.


    """
    source = head_orientation_1.source_positions
    if target_grid == 'source':
        target_source = source.copy()
        target_sampling = \
            spharpy.SamplingSphere.from_coordinates(target_source)

    # Get grid for SH-transform
    radius = source.radius
    if interpolation_grid == "lebedev":
        sampling = spharpy.samplings.lebedev(44, radius=radius[0])
        idx = source.find_nearest(sampling)[0]
        source = source[idx]
    elif interpolation_grid is None:
        idx = slice(None)
        source = source[idx]
    else:
        raise ValueError("grid must be lebedev or None")

    sampling = spharpy.SamplingSphere.from_coordinates(source)

    hrirs1 = head_orientation_1.hrirs[0, *idx]
    hrirs2 = head_orientation_2.hrirs[0, *idx]

    ho1 = head_orientation_1.head_orientations
    ho2 = head_orientation_2.head_orientations
    ho_interp = (ho1 + ho2) / 2
    print(ho_interp)

    # unregularized SH transform
    sh_definition = spharpy.SphericalHarmonicDefinition(n_max)
    y_nm = spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                      sampling,
                                                      'pseudo_inverse')
    target_nm = spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                           target_sampling,
                                                           'pseudo_inverse')

    hrirs1_nm = (y_nm.basis_inv @ hrirs1).T
    hrirs2_nm = (y_nm.basis_inv @ hrirs2).T

    interpolated_nm = (hrirs1_nm.time + hrirs2_nm.time) / 2

    hrirs_interpolated = \
        pf.matrix_multiplication((target_nm.basis, interpolated_nm),
                                 domain='time',
                                 axes=[(0, 1), (1, 0), (0, 1)])
    hrirs_interpolated = pf.Signal(hrirs_interpolated[None, :],
                                   hrirs1.sampling_rate)

    return HeadOrientations(hrirs_interpolated, target_source, ho_interp, None)


def vbap_weights(grid, src, norm=1, K=100, tol=1e-6):
    """Compute VBAP interpolation weights for source directions on a grid.

    The method builds a convex hull over ``grid`` and, for each source direction
    in ``src``, searches candidate triangular faces for a valid barycentric
    solution. Weights are normalized according to ``norm`` and stored at the
    corresponding grid vertex indices.

    Parameters
    ----------
    grid : ndarray, shape (n_grid, 3)
        Loudspeaker or sampling grid points in Cartesian coordinates.
    src : ndarray, shape (n_src, 3)
        Source directions in Cartesian coordinates.
    norm : int, optional
        Weight normalization mode. ``1`` for L1 normalization, ``2`` for L2
        normalization. Default is ``1``.
    K : int, optional
        Number of nearest face-plane candidates considered per source during
        prefiltering. Default is ``100``.
    tol : float, optional
        Tolerance for accepting barycentric weights as non-negative.

    Returns
    -------
    ndarray, shape (n_src, n_grid)
        VBAP weights for each source over all grid points.

    Raises
    ------
    ValueError
        If ``norm`` is not ``1`` or ``2``.
    """
    if norm not in (1, 2):
        raise ValueError("norm must be 1 or 2.")

    hull = ConvexHull(grid)
    simplices = hull.simplices

    # triangle vertices
    tris = grid[simplices]  # (F, 3, 3)

    A = tris[:, 0]
    B = tris[:, 1]
    C = tris[:, 2]

    # face normals
    n = np.cross(B - A, C - A)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / n_norm

    # precompute plane reference points
    # (A is fine as point on plane)

    n_src = src.shape[0]
    n_grid = grid.shape[0]

    weights = np.zeros((n_src, n_grid))

    for i, s in enumerate(src):

        # -------------------------------------------------------
        # 1. FAST GEOMETRIC PREFILTER (your idea)
        # -------------------------------------------------------
        v = s - A
        d = np.abs(np.einsum('ij,ij->i', v, n))  # signed plane distance

        sorted_idx = np.argsort(d)

        # limit search
        candidate_faces = sorted_idx[:K]

        # -------------------------------------------------------
        # 2. SPAUDIOPY-STYLE SEARCH WITH EARLY EXIT
        # -------------------------------------------------------
        for f in candidate_faces:
            simplex = simplices[f]

            V = grid[simplex].T  # (3, 3)

            try:
                g = np.linalg.solve(V, s)
            except np.linalg.LinAlgError:
                continue

            # normalization BEFORE test (important like spaudiopy)
            if norm == 1:
                g = g / np.sum(np.abs(g))
            else:
                g = g / np.linalg.norm(g)

            # spaudiopy-style acceptance criterion
            if np.all(g > -tol):
                g = np.maximum(g, 0)

                if norm == 1:
                    g /= g.sum() if g.sum() != 0 else 1.0
                else:
                    g /= np.linalg.norm(g) if np.linalg.norm(g) != 0 else 1.0

                weights[i, simplex] = g
                break  # <-- CRITICAL spaudiopy early exit

    return weights
