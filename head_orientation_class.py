import numpy as np
import pyfar as pf
import os
import re


pattern = re.compile(
    r"bend(?P<bend>-?\d+(?:\.\d+)?)_"
    r"elev(?P<elev>-?\d+(?:\.\d+)?)_"
    r"azim(?P<azim>-?\d+(?:\.\d+)?)"
)


class HeadOrientationsDataset:
    """Dataset wrapper for head-orientation SOFA files in a directory tree.

    Parameters
    ----------
    base_dir : str or path-like
        Base directory containing orientation subfolders named like
        ``bendX_elevY_azimZ`` and at least one HRIR SOFA file per folder.

    Notes
    -----
    The class scans ``base_dir`` at initialization and stores:

    - orientation values with shape ``(n_orientations, 3)`` as
      ``[bend, elevation, azimuth]``
    - matching SOFA file paths for each orientation
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._filepaths = None
        self._head_orientations = None

        self._find_files(base_dir)

    # DUNDER METHODS
    def __repr__(self):
        repr_string = f"HeadOrientationsDataset with {self.n_orientations} " \
            "head orientations"
        return repr_string

    # PROPERTIES
    @property
    def head_orientations(self):
        """All parsed head orientations.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_orientations, 3)`` with columns
            ``[bend, elevation, azimuth]``.
        """
        return np.asarray(self._head_orientations)

    @property
    def sofa_files(self):
        """SOFA file paths aligned with ``head_orientations``.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of SOFA file paths.
        """
        return np.asarray(self._filepaths)

    @property
    def n_orientations(self):
        """Number of indexed head orientations.

        Returns
        -------
        int
            Number of available orientation entries.
        """
        return self.sofa_files.shape[0]

    # PUBLIC METHODS
    def find_head_orientations(self, bend=None, elevation=None, azimuth=None,
                               tol=1e-9, return_indices=False):
        """Return orientations matching a partial or full query.

        Parameters
        ----------
        bend : float, array-like, or None, optional
            Bend angle(s) in degrees. If ``None``, bend is not constrained.
        elevation : float, array-like, or None, optional
            Elevation angle(s) in degrees. If ``None``, elevation is not
            constrained.
        azimuth : float, array-like, or None, optional
            Azimuth angle(s) in degrees. If ``None``, azimuth is not
            constrained.

            Query modes:

            - scalar mode: each scalar constrains one axis
            - axis-list mode: each list constrains one axis with "any-of"
            - triplet-list mode: if all three are array-like with equal
              length, values are interpreted as orientation triplets
              ``(bend[i], elevation[i], azimuth[i])``
        tol : float, optional
            Absolute tolerance used by ``numpy.isclose`` for floating-point
            comparisons. Default is ``1e-9``.
        return_indices : bool, optional
            If ``True``, return matching indices instead of orientation values.
            Default is ``False``.

        Returns
        -------
        numpy.ndarray
            Filtered orientations with shape ``(n_matches, 3)`` when
            ``return_indices`` is ``False``. Otherwise, one-dimensional array
            of matching indices.
        """
        indices = self._find_orientation(
            bend=bend,
            elevation=elevation,
            azimuth=azimuth,
            tol=tol,
        )
        if return_indices:
            return indices
        return self.head_orientations[indices]

    def get_head_orientations(self, bend=None, elevation=None, azimuth=None,
                              tol=1e-9):
        """Load HRIR data for matching orientations and return a container.

        Parameters
        ----------
        bend : float, array-like, or None, optional
            Bend angle(s) in degrees. If ``None``, bend is not constrained.
        elevation : float, array-like, or None, optional
            Elevation angle(s) in degrees. If ``None``, elevation is not
            constrained.
        azimuth : float, array-like, or None, optional
            Azimuth angle(s) in degrees. If ``None``, azimuth is not
            constrained.
        tol : float, optional
            Absolute tolerance used for orientation matching. Default is
            ``1e-9``.

        Returns
        -------
        HeadOrientations
            Object containing stacked HRIRs, source coordinates, and matched
            orientation values.
        """
        indices = self._find_orientation(
            bend=bend,
            elevation=elevation,
            azimuth=azimuth,
            tol=tol,
        )
        for n, idx in enumerate(indices):
            if n == 0:
                hrirs, source, _ = pf.io.read_sofa(self.sofa_files[idx])
                hrirs = hrirs[None, :]
            else:
                hrirs2, source, _ = pf.io.read_sofa(self.sofa_files[idx])
                hrirs = pf.utils.concatenate_channels((hrirs, hrirs2[None, :]))

        head_orientations_data = \
            HeadOrientations(hrirs,
                             source,
                             self.head_orientations[indices],
                             self.sofa_files[indices])
        return head_orientations_data

    # PRIVATE METHODS
    def _find_files(self, base_dir):
        """Scan directory and populate orientation and file-path caches.

        Parameters
        ----------
        base_dir : str or path-like
            Base directory containing orientation subfolders.

        Notes
        -----
        For each matching subfolder, the first file matching
        ``'HRIR' in fname and fname.endswith('.sofa')`` is stored.
        """
        subdirs = os.listdir(base_dir)
        orientations = []
        files = []

        for subdir in subdirs:
            match = pattern.search(subdir)
            if not match:
                continue

            file = [
                os.path.join(base_dir, subdir, fname)
                for fname in os.listdir(os.path.join(base_dir, subdir))
                if 'HRIR' in fname and fname.endswith('.sofa')][0]

            b = float(match.group("bend"))
            e = float(match.group("elev"))
            a = float(match.group("azim"))

            orientations.append([b, e, a])
            files.append(file)

        self._head_orientations = orientations
        self._filepaths = files

    def _find_orientation(self, bend=None, elevation=None, azimuth=None,
                          tol=1e-9):
        """Return indices matching a partial or full orientation query.

        Parameters
        ----------
        bend : float, array-like, or None, optional
            Bend angle(s) in degrees. If ``None``, bend is not constrained.
        elevation : float, array-like, or None, optional
            Elevation angle(s) in degrees. If ``None``, elevation is not
            constrained.
        azimuth : float, array-like, or None, optional
            Azimuth angle(s) in degrees. If ``None``, azimuth is not
            constrained.
        tol : float, optional
            Absolute tolerance used by ``numpy.isclose`` for floating-point
            comparisons. Default is ``1e-9``.

        Returns
        -------
        numpy.ndarray
            One-dimensional integer array containing matching indices.
        """
        if self._head_orientations is None:
            return np.array([], dtype=int)

        orientations = np.asarray(self._head_orientations, dtype=float)
        if orientations.size == 0:
            return np.array([], dtype=int)

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
                raise ValueError(
                    "For triplet-list queries, bend/elevation/azimuth must "
                    "have the same length."
                )
            query_orientations = np.column_stack(
                (bend_query, elev_query, azim_query)
            )
            comparison = np.isclose(
                orientations[:, None, :],
                query_orientations[None, :, :],
                atol=tol,
                rtol=0.0,
            )
            mask = np.any(np.all(comparison, axis=2), axis=1)
            return np.flatnonzero(mask)

        mask = np.ones(orientations.shape[0], dtype=bool)
        mask &= self._axis_mask(orientations[:, 0], bend_query, tol)
        mask &= self._axis_mask(orientations[:, 1], elev_query, tol)
        mask &= self._axis_mask(orientations[:, 2], azim_query, tol)

        return np.flatnonzero(mask)

    @staticmethod
    def _normalize_query_values(values):
        """Convert scalar or array-like query values to 1D float arrays."""
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
        """Build a boolean mask for one axis using any-of matching."""
        if query_values is None:
            return np.ones(orientation_values.shape[0], dtype=bool)
        return np.any(
            np.isclose(
                orientation_values[:, None],
                query_values[None, :],
                atol=tol,
                rtol=0.0,
            ),
            axis=1,
        )

    # CLASSMETHODS
    @classmethod
    def from_directory(cls, base_dir):
        """Create dataset from a base directory.

        Parameters
        ----------
        base_dir : str or path-like
            Base directory containing orientation subfolders.

        Returns
        -------
        HeadOrientationsDataset
            Initialized dataset instance.
        """
        return cls(base_dir)


class HeadOrientations:
    """Store HRIR data and metadata for one or more head orientations.

    Parameters
    ----------
    hrirs : pyfar.Signal
        HRIR data, typically stacked over head orientations of cshape
        (head_orientations, source_positions, 2).
    source_positions : pyfar.Coordinates
        Source coordinates associated with the HRIR data.
    head_orientations : numpy.ndarray
        Orientation values with shape ``(n_orientations, 3)`` and columns
        ``[bend, elevation, azimuth]``.
    """
    def __init__(self, hrirs, source_positions, head_orientations, fp):
        self._hrirs = hrirs
        self._coordinates = source_positions
        self._head_orientations = head_orientations
        self._file_paths = fp

    # DUNDER METHODS
    def __repr__(self):
        repr_string = f"HeadOrientations object with {self.n_orientations} "\
            "head orientations"
        return repr_string

    def __iter__(self):
        """Iterate over individual head orientations.

        Yields
        ------
        HeadOrientations
            New object containing one orientation slice, shared source
            positions, and the corresponding orientation values.
        """
        orientations = np.asarray(self._head_orientations)
        for idx in range(self.n_orientations):
            yield HeadOrientations(
                self._hrirs[idx:idx + 1, :],
                self._coordinates,
                orientations[idx:idx + 1],
                self.sofa_file_paths[idx:idx + 1]
            )

    # PROPERTIES
    @property
    def sofa_file_paths(self):
        """"""
        return self._file_paths

    @property
    def source_positions(self):
        """Source coordinates associated with the HRIR data.

        Returns
        -------
        pyfar.Coordinates
            Coordinate object for the source positions.
        """
        return self._coordinates

    @property
    def hrirs(self):
        """HRIR data for selected head orientations.

        Returns
        -------
        pyfar.Signal
            HRIR signal object.
        """
        return self._hrirs

    @property
    def head_orientations(self):
        """Head orientation values.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_orientations, 3)``.
        """
        return self._head_orientations

    @property
    def n_orientations(self):
        """Number of stored head orientations.

        Returns
        -------
        int
            Number of orientation entries.
        """
        return self._head_orientations.shape[0]
