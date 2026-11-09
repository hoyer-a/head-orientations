# `head_orientations`

`head_orientations` is a Python package for loading, processing, interpolating,
evaluating, and plotting head-related impulse responses (HRIRs) measured at
multiple three-degree-of-freedom (3DoF) head orientations.

The package is intended for research scripts and analysis notebooks working
with SOFA HRIR datasets. It complements the project documentation by providing
the practical package information needed to install it and get started.

## What it provides

- **Dataset access**: scan orientation folders and load matching SOFA files
  into a common container.
- **Head-orientation containers**: keep HRIRs, source coordinates, orientation
  metadata, and source file paths together.
- **Signal processing**: calculate spectral differences, mean spectral
  differences, far-field corrections, and directional transfer functions.
- **Spatial interpolation**: interpolate HRIRs with spherical harmonics or
  barycentric/VBAP weights, including interpolation between head orientations.
- **Evaluation**: run localization and coloration metrics, including optional
  MATLAB-based perceptual models.
- **Visualization**: plot spectral differences, localization results, and
  coloration maps with Matplotlib.

## Installation

The package requires Python 3.11 or newer. From this directory, install it in
editable mode with:

```bash
python -m pip install -e .
```

The required dependencies are installed automatically:
NumPy, SciPy, pyfar, spharpy, sofar, and Matplotlib.

For development tools and tests:

```bash
python -m pip install -e ".[dev]"
```

The MATLAB-dependent metrics are optional:

```bash
python -m pip install -e ".[matlab]"
```

Installing the Python MATLAB Engine is not sufficient by itself; the MATLAB
functions used by the metric wrappers must also be available on the MATLAB
path. See the project documentation for the model-specific setup.

## Data layout

`HeadOrientationsDataset` expects a directory containing one subdirectory per
head orientation. Directory names must contain the following pattern:

```text
bend<BEND>_elev<ELEV>_azim<AZIM>/
└── ...HRIR....sofa
```

For example:

```text
hrir_data/
├── bend0_elev0_azim0/
│   └── HRIR_lebedev44_and_planes.sofa
├── bend10_elev0_azim0/
│   └── HRIR_lebedev44_and_planes.sofa
└── bend0_elev20_azim0/
    └── HRIR_lebedev44_and_planes.sofa
```

The orientation values are interpreted as degrees and stored in the order
`[bend, elevation, azimuth]`. Within each orientation directory, the first
file whose name contains `HRIR` and ends in `.sofa` is selected.

## Quick start

```python
from head_orientations import HeadOrientationsDataset

dataset = HeadOrientationsDataset.from_directory("/path/to/hrir_data")
print(dataset)
print(dataset.head_orientations)

# Load one orientation. Values are in degrees.
neutral = dataset.get_head_orientations(
    bend=0,
    elevation=0,
    azimuth=0,
)

# Load several orientations by constraining only selected axes.
front_flexion = dataset.get_head_orientations(
    bend=0,
    azimuth=0,
    elevation=[0, 20, 40],
)

print(neutral.hrirs)
print(neutral.source_positions)
```

`HeadOrientationsDataset.find_head_orientations(..., return_indices=True)`
returns matching dataset indices without loading the SOFA data. A loaded
`HeadOrientations` object can be iterated over or indexed to work with one
orientation at a time.

## Typical processing workflow

```python
import matplotlib.pyplot as plt
from head_orientations import (
    HeadOrientationsDataset,
    far_field_correction,
    interpolate_sh,
    plot_single_spectral_difference,
)

dataset = HeadOrientationsDataset.from_directory("/path/to/hrir_data")
reference = dataset.get_head_orientations(0, 0, 0)
target = dataset.get_head_orientations(20, 0, 0)

# Optional normalization to an approximate far-field reference.
target_farfield = far_field_correction(target, r_ref=100)

# Reconstruct the response on another spherical sampling.
interpolated = interpolate_sh(
    target_farfield,
    target_coordinates=reference.source_positions,
    n_max=44,
    grid="lebedev",
)

plot_single_spectral_difference(interpolated, reference)
plt.show()
```

The interpolation grid and target coordinates must use the same source radius.
Spherical-harmonic interpolation defaults to a 44-point Lebedev analysis grid;
pass `grid=None` to use the original source positions when appropriate.

## Public API

The most commonly used objects and functions are re-exported from the
top-level package:

### Data containers

- `HeadOrientationsDataset`: index orientation directories and load HRIR data.
- `HeadOrientations`: store HRIR signals, source positions, orientation values,
  and originating SOFA paths.
- `HeadOrientationsMetrics`: store metric arrays loaded from MATLAB `.mat`
  files or created with `HeadOrientationsMetrics.from_data(...)`.

### Processing and interpolation

- `spectral_difference`, `mean_spectral_difference`
- `far_field_correction`, `directional_transfer_function`
- `interpolate_sh`
- `interpolate_barycentric`
- `interpolate_head_orientation`
- `vbap_weights`
- `find_orientation_directory`, `find_indices_in_region`,
  `find_indices_in_bbox`

`interpolate_barycentric` operates on a single head orientation. It uses
Cartesian source coordinates and returns a new `HeadOrientations` object at the
requested target coordinates. `interpolate_head_orientation` combines two
orientations and reports their midpoint orientation metadata.

### Metrics

- `barumerli_localization`: MATLAB-based localization evaluation.
- `baumgartner_localization`: MATLAB-based sagittal-plane localization
  evaluation.
- `coloration_mc_kenzie`: MATLAB-based coloration evaluation.

These functions require the optional MATLAB dependency and the corresponding
external model code. They can return results directly and can optionally write
metric `.mat` files to an output directory.

### Plotting

The plotting module includes:

- spectral-difference plots for individual or multiple orientations;
- mean spectral-difference maps;
- localization scatter, 1DoF, and map plots;
- coloration maps and subplot grids.

Plot functions generally create and display Matplotlib figures. Functions with
an `output_file` parameter can save a figure while still returning or
displaying it according to their function-specific behavior.

## Examples

The `test.py` script contains exploratory examples covering SOFA loading,
spherical-harmonic and barycentric interpolation, spectral differences,
localization metrics, and plotting.

## Development

Run the package tests from this directory with:

```bash
python -m pytest
```

The repository currently contains exploratory scripts and generated analysis
outputs alongside the package. Keep reusable package code under
`head_orientations/`; use `test.py` and `test_plots/` for analysis examples
rather than treating them as a stable public API.

## Related documentation

This README is a package overview and quick-start guide, not a replacement for
the full project or API documentation. The configured documentation site is
<https://head-orientations.readthedocs.io>.

## License and project links

Project links and package metadata are defined in [`pyproject.toml`](pyproject.toml).
The repository also contains the broader research context, datasets, and
project-level documentation.
