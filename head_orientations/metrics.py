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
    """Container for localization metrics from .mat files or arrays.

    Can load metrics from .mat files in a directory by passing `base_dir`,
    or create from arrays using the `from_data` class method.

    Parameters
    ----------
    base_dir : str, path-like, or None
        Directory to scan for metric .mat files. If None, no files are loaded.
    metric_keys : tuple or list, optional
        Metric names to extract. Defaults to DEFAULT_METRIC_KEYS.
    comment : str, optional
        Optional comment/metadata string.
    """

    _pattern = re.compile(
        r"metrics_bend_(?P<bend>-?\d+(?:\.\d+)?)"
        r"elev_(?P<elev>-?\d+(?:\.\d+)?)"
        r"azim(?P<azim>-?\d+(?:\.\d+)?)\.mat$"
    )
    DEFAULT_METRIC_KEYS = ("accL", "rmsL", "accP", "rmsP", "querr", "gainP")

    def __init__(self, base_dir=None, metric_keys=None, comment=None):
        self._base_dir = base_dir
        self._filepaths = []
        self._head_orientations = []
        self._metric_keys = tuple(metric_keys) if metric_keys is not None else self.DEFAULT_METRIC_KEYS
        self._metrics = {key: [] for key in self._metric_keys}
        self._comment = comment

        if base_dir is not None:
            self._find_files(base_dir)

    def __repr__(self):
        return (
            f"HeadOrientationsMetrics with {self.n_orientations} entries "
            f"and metrics {self._metric_keys}"
        )

    @classmethod
    def from_data(cls, head_orientations, metric_keys=None, comment=None, **metrics_data):
        """Create HeadOrientationsMetrics from arrays.

        Parameters
        ----------
        head_orientations : array-like
            Head orientation values with shape (n_orientations, 3) as
            [bend, elevation, azimuth].
        metric_keys : tuple or list, optional
            Metric names. If None, inferred from metrics_data keys.
        comment : str, optional
            Optional comment/metadata string.
        **metrics_data
            Keyword arguments for each metric (e.g., querr=querr_array,
            rmsL=rmsL_array). Each must have shape (n_orientations,) or
            be broadcastable to that shape.

        Returns
        -------
        HeadOrientationsMetrics
            New instance with data populated from arrays.
        """
        head_orientations = np.asarray(head_orientations, dtype=float)
        if head_orientations.ndim == 1:
            head_orientations = head_orientations.reshape(1, -1)
        n_orientations = head_orientations.shape[0]

        # Infer metric keys from metrics_data if not provided
        if metric_keys is None:
            metric_keys = tuple(sorted(metrics_data.keys()))
        else:
            metric_keys = tuple(metric_keys)

        # Create instance without loading files
        instance = cls(base_dir=None, metric_keys=metric_keys, comment=comment)

        # Populate head orientations
        instance._head_orientations = head_orientations.tolist()
        instance._filepaths = [None] * n_orientations

        # Populate metrics
        for key in metric_keys:
            if key not in metrics_data:
                raise ValueError(
                    f"Missing metric key '{key}'. Provide it as a keyword argument."
                )
            values = np.asarray(metrics_data[key], dtype=float).ravel()
            if values.shape[0] != n_orientations:
                raise ValueError(
                    f"Metric '{key}' has {values.shape[0]} values, "
                    f"expected {n_orientations}."
                )
            instance._metrics[key] = values.tolist()

        return instance

    @property
    def comment(self):
        return str(self._comment)

    @comment.setter
    def comment(self, value):
        self._comment = str(value)

    @property
    def head_orientations(self):
        return np.asarray(self._head_orientations, dtype=float)

    @property
    def mat_files(self):
        return np.asarray(self._filepaths)

    @property
    def metric_keys(self):
        return self._metric_keys

    @property
    def n_orientations(self):
        return len(self._filepaths)

    @property
    def metrics_matrix(self):
        return np.column_stack([self.metric_values(key) for key in self._metric_keys])

    def metric_values(self, key):
        if key not in self._metrics:
            raise KeyError(f"Unknown metric key: {key}")
        return np.asarray(self._metrics[key], dtype=float)

    def get_metrics(self, bend=None, elevation=None, azimuth=None, tol=1e-9,
                    return_indices=False):
        """Return head orientations and metric arrays matching a query."""
        indices = self._find_orientation(bend=bend, elevation=elevation,
                                         azimuth=azimuth, tol=tol)
        if return_indices:
            return indices
        return self.head_orientations[indices], self.metrics_matrix[indices]

    def get_subset(self, bend=None, elevation=None, azimuth=None, tol=1e-9):
        """Return a new HeadOrientationsMetrics instance with matching entries.

        Accepts the same query patterns as ``get_metrics``: ``None`` leaves an
        axis unconstrained; a scalar or sequence selects specific values; three
        equal-length sequences are interpreted as triplets
        ``(bend[i], elevation[i], azimuth[i])``.

        Parameters
        ----------
        bend : float, array-like, or None, optional
            Bend angle(s) in degrees.
        elevation : float, array-like, or None, optional
            Elevation angle(s) in degrees.
        azimuth : float, array-like, or None, optional
            Azimuth angle(s) in degrees.
        tol : float, optional
            Absolute tolerance for floating-point comparisons. Default ``1e-9``.

        Returns
        -------
        HeadOrientationsMetrics
            New instance containing only the matching orientations and their
            associated metric values.
        """
        indices = self._find_orientation(bend=bend, elevation=elevation,
                                         azimuth=azimuth, tol=tol)

        subset_orientations = self.head_orientations[indices]
        metrics_data = {
            key: self.metric_values(key)[indices] for key in self._metric_keys
        }

        instance = type(self).from_data(
            subset_orientations,
            metric_keys=self._metric_keys,
            comment=self._comment,
            **metrics_data,
        )

        # Preserve original file paths when available
        instance._filepaths = [self._filepaths[i] for i in indices]

        return instance

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

                for key in self._metric_keys:
                    val = self._find_in_mat(mat, key)
                    self._metrics[key].append(self._coerce_metric_value(val))

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

        if hasattr(obj, "_fieldnames"):
            fieldnames = getattr(obj, "_fieldnames", None) or []
            if key in fieldnames:
                return getattr(obj, key)
            for field in fieldnames:
                res = self._find_in_mat(getattr(obj, field), key)
                if res is not None:
                    return res
            return None

        if hasattr(obj, key):
            return getattr(obj, key)

        if isinstance(obj, np.ndarray):
            # iterate elements (handles struct arrays / object arrays)
            for el in obj.ravel():
                res = self._find_in_mat(el, key)
                if res is not None:
                    return res
            return None

        return None

    @staticmethod
    def _coerce_metric_value(value):
        if value is None:
            return np.nan
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return float(arr.squeeze())
        return arr

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

    def __getattr__(self, name):
        if name in self._metrics:
            return np.asarray(self._metrics[name], dtype=float)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


for _metric_name in HeadOrientationsMetrics.DEFAULT_METRIC_KEYS:
    setattr(
        HeadOrientationsMetrics,
        _metric_name,
        property(lambda self, metric_name=_metric_name: np.asarray(self._metrics[metric_name], dtype=float)),
    )

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
    template_subsampling: pf.Coordinates = None,
    target_subsampling: pf.Coordinates = None,
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
        sofa_target = _get_subset(sofa_target, target_subsampling)
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
            sofa_target = _get_subset(sofa_target, target_subsampling)
            _, feat_target = eng.barumerli2023_NOINTERPOLATION_featureextraction(
                sofa_target,
                'pge',
                nargout=2)

        if template_subsampling:
            sofa_template = _get_subset(sofa_template, template_subsampling)

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

            if save_matrix:
                matrix_dir = os.path.join(output_dir, 'prediction_matrices')
                if not os.path.exists(matrix_dir):
                    os.mkdir(matrix_dir)
                matrix_filename = f"matrix_bend_{int(orientation[0])}" \
                    f"elev_{int(orientation[1])}" \
                        f"azim{int(orientation[2])}.mat"
                matrix_filepath = os.path.join(matrix_dir, matrix_filename)
                sc.io.savemat(matrix_filepath,
                              {"prediction_matrix": prediction_matrix}
)

        results.append(metrics)

    return results


def coloration_mc_kenzie(head_orientations: HeadOrientations,
                         reference: HeadOrientations,
                         frequency_range: Sequence = (300, 20e3),
                         output_dir: str = None):
    """"""
    eng = _get_matlab_engine()
    results = []

    if frequency_range:
        settings_dict = {"minFreq": frequency_range[0],
                         "maxFreq": frequency_range[1],}
    else:
        settings_dict = eng.struct()

    hrirs = head_orientations.hrirs
    # source = head_orientations.source_positions

    ref_hrirs = reference.hrirs
    ref_data = np.transpose(ref_hrirs.time.squeeze(axis=0), axes=[2, 0, 1]).copy()

    for id in range(head_orientations.n_orientations):
        data = np.transpose(hrirs.time[id], axes=[2, 0, 1]).copy()
        pbc = eng.mckenzie2025(ref_data, data, settings_dict, nargout=1)

        pbc = np.asarray(pbc).squeeze()

        if output_dir:
            orientation = head_orientations.head_orientations[id]
            filename = f"metrics_bend_{int(orientation[0])}" \
                f"elev_{int(orientation[1])}" \
                    f"azim{int(orientation[2])}.mat"
            filepath = os.path.join(output_dir, filename)
            sc.io.savemat(filepath, {'pbc': pbc})
            print(f"saved to {filepath}")

        results.append(pbc)

    return results


def baumgartner_localization(template_head_orientations: HeadOrientations,
                             target_head_orientations: HeadOrientations,
                             output_dir: str = None,
                             spectral_weighting: bool = False,
                             gamma = 6.0,
                             S = 1.0):
    """"""
    eng = _get_matlab_engine()

    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    sagittal_plane = pf.Coordinates.from_spherical_elevation(0, angles, 1)

    source = template_head_orientations.source_positions
    src_idx = source.find_nearest(sagittal_plane)[0]

    if target_head_orientations.n_orientations != 1 \
        and target_head_orientations.n_orientations != \
            template_head_orientations.n_orientations:
        raise ValueError("Don't do this")

    if target_head_orientations.n_orientations == 1:
        print("single target orientation")
        hrirs_target = target_head_orientations.hrirs[0]
        target = np.ascontiguousarray(
            np.moveaxis(hrirs_target.time, 2, 0)[:, *src_idx, :])

    results = []

    for idx in range(template_head_orientations.n_orientations):
        # If the template has beed interpolated before, there is no
        # corresponding sofa file, so we create a temporary one
        hrirs_template = template_head_orientations[idx].hrirs[0]
        template = np.ascontiguousarray(
            np.moveaxis(hrirs_template.time, 2, 0)[:, *src_idx, :])

        if target_head_orientations.n_orientations != 1:
            hrirs_target = target_head_orientations[idx].hrirs[0]
            target = np.ascontiguousarray(
                np.moveaxis(hrirs_target.time, 2, 0)[:, *src_idx, :])

        import scipy.io
        scipy.io.savemat('/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_scripts/head_orientations_package/head_orientations/debug_inputs.mat', {
            'target': target,
            'template': template,
            'spectw': np.asarray(spectral_weighting)
        })

        if spectral_weighting:
            err, _ = eng.baumgartner2014(target, template,
                                         'fs', hrirs_template.sampling_rate,
                                        'fsstim', hrirs_template.sampling_rate,
                                        'polsamp', np.rad2deg(angles),
                                        'tang', np.rad2deg(angles),
                                        'rangsamp', 2.0,
                                        'spectw', spectral_weighting,
                                        'gamma', gamma,
                                        'bwcoef', 1e-6,
                                        'S', S,
                                        'QE_PE_EB', nargout=2)
        else:
            err, _ = eng.baumgartner2014(target, template,
                                        'fs', hrirs_template.sampling_rate,
                                        'fsstim', hrirs_template.sampling_rate,
                                        'polsamp', np.rad2deg(angles),
                                        'tang', np.rad2deg(angles),
                                        'rangsamp', 2.0,
                                        'gamma', gamma,
                                        'bwcoef', 1e-6,
                                        'S', S,
                                        'QE_PE_EB', nargout=2)

        metrics = {'querr': err['qe'],
                   'rmsP': err['pe'],
                   'pb': err['pb']}

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
