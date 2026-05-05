"""DSP functions for hrtf processing and normalization."""
import pyfar as pf
import numpy as np

from .head_orientation_class import HeadOrientations
from ._spherical_head import spherical_head


def far_field_correction(head_orientations: HeadOrientations,
                         r_ref: float = 100):
    """
    Apply a far-field correction to a head-orientation data set.

    The correction uses a spherical-head transfer function to scale the HRIRs
    from their current source distance to a reference far-field distance.

    Parameters
    ----------
    head_orientations : HeadOrientations
        Container with HRIRs, source positions, orientation values, and the
        originating SOFA file paths.
    r_ref : float, optional
        Reference radius used to approximate the far-field condition.
        Defaults to ``100``.

    Returns
    -------
    HeadOrientations
        A new container with the corrected HRIRs and the original metadata.

    See also
    --------
    [#]_ for the spherical-head model used here.

    .. [#] Bahu, H., Jot, J.-M., Carpentier, T., Noisternig, M., Mihocic, M.,
       Majdak, P., & Warusfel, O. (2026). Toward Improved Consistency Between
       Databases of Head-Related Transfer Functions. *Journal of the Audio
       Engineering Society*, 74(1/2), 35-49.
       https://doi.org/10.17743/jaes.2022.0247
    """
    hrirs = head_orientations.hrirs
    source = head_orientations.source_positions

    source_farfield = source.copy()
    source_farfield.radius = r_ref

    shtf = spherical_head(source, n_samples=hrirs.n_samples,
                        sampling_rate=hrirs.sampling_rate)
    shtf_farfield = spherical_head(source_farfield, n_samples=hrirs.n_samples,
                                sampling_rate=hrirs.sampling_rate)

    dvfs = shtf_farfield / shtf
    hrirs *= dvfs
    return HeadOrientations(hrirs, source, head_orientations.head_orientations,
                            None)


def directional_transfer_function(head_orientations: HeadOrientations,
                                  average_method: str='log_magnitude_zerophase'):
    """
    Compute the directional transfer function for head orientations.

    The directional transfer function is obtained by averaging the HRIRs of
    each head orientation and dividing the individual responses by that common
    transfer function.

    Parameters
    ----------
    head_orientations : HeadOrientations
        Container with HRIRs, source positions, orientation values, and the
        originating SOFA file paths.
    average_method : str, optional
        Averaging strategy passed to :func:`pyfar.dsp.average`.
        Defaults to ``'log_magnitude_zerophase'``.

    Returns
    -------
    HeadOrientations
        A new container holding the directional transfer function data and
        the original source-position and orientation metadata.

    See also
    --------
    [#]_ for the averaging approach used here.

    .. [#] Bahu, H., Jot, J.-M., Carpentier, T., Noisternig, M., Mihocic, M.,
       Majdak, P., & Warusfel, O. (2026). Toward Improved Consistency Between
       Databases of Head-Related Transfer Functions. *Journal of the Audio
       Engineering Society*, 74(1/2), 35-49.
       https://doi.org/10.17743/jaes.2022.0247
    """
    hrirs = head_orientations.hrirs
    caxes = tuple(np.arange(head_orientations.hrirs.cdim)[1:])
    ctf = pf.dsp.average(hrirs, average_method,
                         caxis=caxes)

    inverse_ctf = 1/ctf
    dtf = inverse_ctf[:, None, None] * hrirs

    return HeadOrientations(dtf, head_orientations.source_positions,
                            head_orientations.head_orientations, None)
