import pyfar as pf
import spharpy
import numpy as np
from head_orientation_class import HeadOrientations
from scipy.interpolate import make_interp_spline, LinearNDInterpolator


def interpolate_to_global_coords(head_orientations: HeadOrientations,
                                 target_coordinates: pf.Coordinates,
                                 n_max: int,
                                 grid: str = "lebedev"):
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

    # Get SamplingSphere for source and target coordinates
    sampling = spharpy.SamplingSphere.from_coordinates(source)
    target_sampling = \
        spharpy.SamplingSphere.from_coordinates(target_coordinates)

    # Get basis matrix
    sh_definition = spharpy.SphericalHarmonicDefinition(n_max)
    y_nm = \
        spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                   sampling,
                                                   inverse_method='pseudo_inverse')
    hrirs = head_orientations.hrirs
    # copy hrirs
    hrirs = hrirs[:, *idx].copy()

    # time align
    hrirs_onset = pf.dsp.resample(hrirs, hrirs.sampling_rate * 10,
                                  post_filter=True)
    hrirs_onset = pf.dsp.filter.butterworth(hrirs_onset, 10, 3e3)
    onsets = pf.dsp.find_impulse_response_start(hrirs_onset) / 10
    hrirs = pf.dsp.fractional_time_shift(hrirs, -onsets, mode='cyclic')

    toa_interpolator = LinearNDInterpolator(source.cartesian, onsets)
    target_onsets = toa_interpolator(target_coordinates.cartesian)

    # Iterate over head orientations: sh-transform, rotate & interpolate
    for n, head_orientation in enumerate(head_orientations):
        hrir = hrirs[n, ...]
        hrir.time = hrir.time.squeeze()
        orientation = head_orientation.head_orientations
        hrir_nm = (y_nm.basis_inv @ hrir).T

        hrir_nm = spharpy.SphericalHarmonicSignal.from_definition(
            sh_definition, hrir_nm.time, hrir_nm.sampling_rate)

        orientation = -np.deg2rad(orientation).squeeze()

        Rotation = \
            spharpy.transforms.SphericalHarmonicRotation.from_euler(
                'XYZ', [orientation[0], orientation[1], orientation[2]])

        rotated_nm = Rotation.apply(hrir_nm)
        target_nm = \
            spharpy.SphericalHarmonics.from_definition(sh_definition,
                                                       target_sampling,
                                                       inverse_method='pseudo_inverse')

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
                            head_orientations.head_orientations)


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

    return HeadOrientations(hrirs_interpolated, target_source, ho_interp)






