import pyfar as pf
import sofar as sf
import scipy as sc
from importlib import import_module
from typing import Sequence
import tempfile
from .head_orientation_class import HeadOrientations
import os
import re
import numpy as np


_MATLAB_ENGINE = None

class HeadOrientationsMetrics:
    """Container for Barumerli localization metrics saved as .mat files.

    Scans a directory for files named like
    ``metrics_bend_<bend>elev_<elev>azim<azim>.mat`` and loads the
    stored metrics (e.g. ``rmsL``, ``rmsP``, ``querr``) per orientation.

    Parameters
    ----------
    base_dir : str or path-like
        Directory to scan for metric .mat files.
    """

    _pattern = re.compile(
        r"metrics_bend_(?P<bend>-?\d+(?:\.\d+)?)"
        r"elev_(?P<elev>-?\d+(?:\.\d+)?)"
        r"azim(?P<azim>-?\d+(?:\.\d+)?)\.mat$"
    )

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._filepaths = []
        self._head_orientations = []
        self._metrics = []

        self._find_files(base_dir)

    def __repr__(self):
        return f"HeadOrientationsMetrics with {self.n_orientations} entries"

    @property
    def head_orientations(self):
        return np.asarray(self._head_orientations, dtype=float)

    @property
    def mat_files(self):
        return np.asarray(self._filepaths)

    @property
    def n_orientations(self):
        return len(self._filepaths)

    def get_metrics(self, bend=None, elevation=None, azimuth=None, tol=1e-9,
                    return_indices=False):
        """Return metrics matching a query (similar semantics to
        HeadOrientationsDataset.find_head_orientations).
        """
        indices = self._find_orientation(bend=bend, elevation=elevation,
                                         azimuth=azimuth, tol=tol)
        if return_indices:
            return indices
        return [self._metrics[i] for i in indices]

    def _find_files(self, base_dir):
        """Scan the base directory (recursively) and load .mat metric files."""
        for root, _, files in os.walk(base_dir):
            for fname in files:
                match = self._pattern.search(fname)
                if not match:
                    continue

                filepath = os.path.join(root, fname)
                b = float(match.group("bend"))
                e = float(match.group("elev"))
                a = float(match.group("azim"))

                self._filepaths.append(filepath)
                self._head_orientations.append([b, e, a])

                try:
                    mat = sc.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
                except Exception:
                    mat = None

                metrics = {
                    "rmsL": None,
                    "rmsP": None,
                    "querr": None,
                    "raw": mat,
                }

                if mat is not None:
                    for key in ("rmsL", "rmsP", "querr"):
                        val = self._find_in_mat(mat, key)
                        if val is not None:
                            metrics[key] = np.asarray(val)

                self._metrics.append(metrics)

    def _find_in_mat(self, obj, key):
        """Recursively search a loaded .mat structure for a field named ``key``.

        Returns the first match or ``None`` if not found.
        """
        if obj is None:
            return None

        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                res = self._find_in_mat(v, key)
                if res is not None:
                    return res
            return None

        if isinstance(obj, np.ndarray):
            # iterate elements (handles struct arrays / object arrays)
            for el in obj.ravel():
                res = self._find_in_mat(el, key)
                if res is not None:
                    return res
            return None

        return None

    def _find_orientation(self, bend=None, elevation=None, azimuth=None,
                          tol=1e-9):
        if len(self._head_orientations) == 0:
            return np.array([], dtype=int)

        orientations = np.asarray(self._head_orientations, dtype=float)

        bend_query = self._normalize_query_values(bend)
        elev_query = self._normalize_query_values(elevation)
        azim_query = self._normalize_query_values(azimuth)

        is_triplet_query = (
            bend_query is not None
            and elev_query is not None
            and azim_query is not None
            and bend_query.size > 1
            and elev_query.size > 1
            and azim_query.size > 1
        )

        if is_triplet_query:
            if not (bend_query.size == elev_query.size == azim_query.size):
                raise ValueError("For triplet-list queries, bend/elevation/azimuth must have the same length.")
            query_orientations = np.column_stack((bend_query, elev_query, azim_query))
            comparison = np.isclose(orientations[:, None, :], query_orientations[None, :, :], atol=tol, rtol=0.0)
            mask = np.any(np.all(comparison, axis=2), axis=1)
            return np.flatnonzero(mask)

        mask = np.ones(orientations.shape[0], dtype=bool)
        mask &= self._axis_mask(orientations[:, 0], bend_query, tol)
        mask &= self._axis_mask(orientations[:, 1], elev_query, tol)
        mask &= self._axis_mask(orientations[:, 2], azim_query, tol)

        return np.flatnonzero(mask)

    @staticmethod
    def _normalize_query_values(values):
        if values is None:
            return None
        if np.isscalar(values):
            return np.asarray([values], dtype=float)
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _axis_mask(orientation_values, query_values, tol):
        if query_values is None:
            return np.ones(orientation_values.shape[0], dtype=bool)
        return np.any(np.isclose(orientation_values[:, None], query_values[None, :], atol=tol, rtol=0.0), axis=1)

def _get_matlab_engine():
    global _MATLAB_ENGINE

    if _MATLAB_ENGINE is None:
        matlab_engine = import_module("matlab.engine")
        shared_sessions = matlab_engine.find_matlab()

        if shared_sessions:
            _MATLAB_ENGINE = matlab_engine.connect_matlab(shared_sessions[0])
        else:
            _MATLAB_ENGINE = matlab_engine.start_matlab()

        _MATLAB_ENGINE.amt_start(nargout=0)
        _MATLAB_ENGINE.SOFAstart(nargout=0)

    return _MATLAB_ENGINE

def _load_tmp_sofa(head_orientation):
    """"""
    print("creating tempdir for sofa file")

    eng = _get_matlab_engine()

    sofa = sf.Sofa("SimpleFreeFieldHRIR")

    # pyfar coordinates are in radians, SOFA SourcePosition uses degrees.
    source_positions = np.asarray(
        head_orientation.source_positions.spherical_elevation,
        dtype=float,
    ).copy()
    source_positions[:, :2] = np.rad2deg(source_positions[:, :2])
    sofa.SourcePosition = source_positions

    sofa.Data_IR = head_orientation.hrirs.time[0]
    sofa.Data_SamplingRate = head_orientation.hrirs.sampling_rate

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = f"{tmpdir}/example.sofa"

        # Save the SOFA file
        sf.write_sofa(file_path, sofa)

        # You can read it back if needed
        sofa = eng.SOFAload(file_path, nargout=1)
    return sofa

def _get_subset(sofa, sampling):
    if sampling is None:
        return sofa

    eng = _get_matlab_engine()

    # pyfar/spharpy coordinate angles are in radians, SOFAfind expects degrees.
    azimuth_deg = np.rad2deg(np.asarray(sampling.azimuth, dtype=float))
    elevation_deg = np.rad2deg(np.asarray(sampling.elevation, dtype=float))

    idx = eng.SOFAfind(sofa, azimuth_deg,
                       elevation_deg, nargout=1)

    eng.workspace['sofa'] = sofa
    eng.workspace['idx'] = idx

    eng.eval("new_sofa=sofa;", nargout=0)
    eng.eval("new_sofa.Data.IR = sofa.Data.IR(idx,:,:);", nargout=0)
    eng.eval("new_sofa.SourcePosition = sofa.SourcePosition(idx,:,:);", nargout=0)

    sofa_subset = eng.eval("new_sofa", nargout=1)
    return sofa_subset


def barumerli_localization(
    template_head_orientations: HeadOrientations,
    target_head_orientations: HeadOrientations,
    subsampling: pf.Coordinates = None,
    output_dir: str = None,
    repetitions: int = 200,
    save_matrix: bool = False,
):
    """"""
    eng = _get_matlab_engine()

    if target_head_orientations.n_orientations != 1 \
        and target_head_orientations.n_orientations != \
            template_head_orientations.n_orientations:
        raise ValueError("Don't do this")

    if target_head_orientations.n_orientations == 1:
        print("single target orientation")
        if target_head_orientations.sofa_file_paths is None:
            sofa_target = _load_tmp_sofa(target_head_orientations[0])
        else:
            sofa_target = eng.SOFAload(
                str(target_head_orientations.sofa_file_paths[0]), nargout=1)
        sofa_target = _get_subset(sofa_target, subsampling)
        _, feat_target = eng.barumerli2023_NOINTERPOLATION_featureextraction(
            sofa_target,
            'pge',
            nargout=2)

    results = []

    for idx in range(template_head_orientations.n_orientations):
        # If the template has beed interpolated before, there is no
        # corresponding sofa file, so we create a temporary one
        if template_head_orientations.sofa_file_paths is None:
            sofa_template = _load_tmp_sofa(template_head_orientations[idx])
        else:
            # exctract template features for current ho
            sofa_template = \
                eng.SOFAload(str(template_head_orientations.sofa_file_paths[idx]),
                            nargout=1)

        if target_head_orientations.n_orientations != 1:
            if target_head_orientations.sofa_file_paths is None:
                sofa_target = _load_tmp_sofa(target_head_orientations[idx])
            else:
                # exctract target features for current ho
                sofa_target = \
                    eng.SOFAload(str(target_head_orientations.sofa_file_paths[idx]),
                                nargout=1)
            sofa_target = _get_subset(sofa_target, subsampling)
            _, feat_target = eng.barumerli2023_NOINTERPOLATION_featureextraction(
                sofa_target,
                'pge',
                nargout=2)

        sofa_template = _get_subset(sofa_template, subsampling)

        feat_template, _ = \
            eng.barumerli2023_NOINTERPOLATION_featureextraction(sofa_template,
                                                                'pge',
                                                                nargout=2)

        # get prediction
        prediction_matrix = eng.barumerli2023('template', feat_template,
                                              'target', feat_target,
                                              'num_exp', repetitions)

        metrics = eng.barumerli2023_metrics(prediction_matrix,
                                            'middle_metrics')

        if output_dir:
            orientation = template_head_orientations.head_orientations[idx]
            filename = f"metrics_bend_{int(orientation[0])}" \
                f"elev_{int(orientation[1])}" \
                    f"azim{int(orientation[2])}.mat"
            filepath = os.path.join(output_dir, filename)
            sc.io.savemat(filepath, metrics)
            print(f"saved to {filepath}")

        results.append(metrics)

    return results


def coloration_mc_kenzie(head_orientations: HeadOrientations,
                         reference: HeadOrientations,
                         frequency_range: Sequence = (300, 20e3)):
    """"""
    eng = _get_matlab_engine()
    results = []

    if frequency_range:
        settings_dict = {"minFreq": frequency_range[0],
                         "maxFreq": frequency_range[1],}
    else:
        settings_dict = eng.struct()

    hrirs = head_orientations.hrirs
    source = head_orientations.source_positions

    ref_hrirs = reference.hrirs
    ref_data = np.transpose(ref_hrirs.time.squeeze(axis=0), axes=[2, 0, 1]).copy()

    for id in range(head_orientations.n_orientations):
        data = np.transpose(hrirs.time.squeeze(axis=0), axes=[2, 0, 1]).copy()
        pbc = eng.mckenzie2025(ref_data, data, settings_dict, nargout=1)

        pbc = np.asarray(pbc).squeeze()
        results.append(pbc)

    return results


