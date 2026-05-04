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
    spec_diff = pf.FrequencyData(spectral_difference, sig1.frequencies)

    return spec_diff
