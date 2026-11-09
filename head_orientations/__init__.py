"""Head Orientations Package

Utilities for working with head-related impulse responses (HRIRs) across
multiple head orientations and spatial positions.

This package provides:

- Dataset management via `HeadOrientationsDataset` and `HeadOrientations`
- DSP operations for far-field corrections and directional transfer functions
- Spatial audio interpolation using spherical harmonics
- Plotting utilities for visualization
- MATLAB-based perceptual evaluation (optional)
"""

from .head_orientation_class import HeadOrientationsDataset, HeadOrientations
from .utils import (
    find_orientation_directory,
    spectral_difference,
    mean_spectral_difference,
    find_indices_in_region,
    find_indices_in_bbox,
)
from .dsp import far_field_correction, directional_transfer_function
from .interpolate import (
    interpolate_sh,
    interpolate_barycentric,
    interpolate_head_orientation,
    vbap_weights,
)
from .metrics import (
    HeadOrientationsMetrics,
    barumerli_localization,
    baumgartner_localization,
    coloration_mc_kenzie,
)
from .plot import (
    subplot_spectral_difference,
    plot_single_spectral_difference,
    plot_mean_spectral_difference,
    plot_localization_1_dof,
    plot_localization_scatter,
    plot_localization_map,
    plot_localization_map_subplots,
    plot_coloration_map,
    plot_coloration_map_subplots,
)

__all__ = [
    "HeadOrientationsDataset",
    "HeadOrientations",
    "find_orientation_directory",
    "spectral_difference",
    "mean_spectral_difference",
    "find_indices_in_region",
    "find_indices_in_bbox",
    "far_field_correction",
    "directional_transfer_function",
    "interpolate_sh",
    "interpolate_barycentric",
    "interpolate_head_orientation",
    "vbap_weights",
    "HeadOrientationsMetrics",
    "barumerli_localization",
    "baumgartner_localization",
    "coloration_mc_kenzie",
    "subplot_spectral_difference",
    "plot_single_spectral_difference",
    "plot_mean_spectral_difference",
    "plot_localization_1_dof",
    "plot_localization_scatter",
    "plot_localization_map",
    "plot_localization_map_subplots",
    "plot_coloration_map",
    "plot_coloration_map_subplots",
]

__version__ = "0.1.0"
__author__ = "HATO Contributors"
__license__ = "MIT"
