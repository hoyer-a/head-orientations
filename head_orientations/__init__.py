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
from .utils import find_orientation_directory
from .dsp import far_field_correction, directional_transfer_function
from .interpolate import interpolate
from .metrics import HeadOrientationsMetrics, barumerli_localization
from .plot import (
    subplot_spectral_difference,
    plot_single_spectral_difference,
)

__all__ = [
    "HeadOrientationsDataset",
    "HeadOrientations",
    "find_orientation_directory",
    "far_field_correction",
    "directional_transfer_function",
    "interpolate",
    "HeadOrientationsMetrics",
    "barumerli_localization",
    "subplot_spectral_difference",
    "plot_single_spectral_difference",
]

__version__ = "0.1.0"
__author__ = "HATO Contributors"
__license__ = "MIT"
