# %%
import pyfar as pf
import sofar as sf
import spharpy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matlab.engine
import pandas as pd
import os
import tempfile
from scipy.interpolate import LinearNDInterpolator

from head_orientations.head_orientation_class import HeadOrientationsDataset
from head_orientations.plot import *
from head_orientations.interpolate import interpolate_sh, interpolate_head_orientation, vbap_weights
from head_orientations.metrics import barumerli_localization, HeadOrientationsMetrics, coloration_mc_kenzie, baumgartner_localization
from head_orientations.dsp import *
from head_orientations.utils import find_indices_in_region
# %%
base_dir2 = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius" \
    "/sofa/raw_data"
base_dir1 = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/1m radius" \
    "/sofa/raw_data/new fmm"

Dataset = HeadOrientationsDataset.from_directory(base_dir2)
HeadOrientations = Dataset.get_head_orientations(azimuth=0,
                                                 bend=0,
                                                 elevation=(0, -10, -20, -30, -40))
Fixed = Dataset.get_head_orientations(0, 0, 0)
Head = Dataset.get_head_orientations(0.5, 0.5, 0.5)

print(Dataset)
hrirs = HeadOrientations.hrirs
print(hrirs)

# %%
# ------------------- PLOTS: SPECTRAL DIFFERENCE SUBPLOTS ---------------------
subplot_spectral_difference(HeadOrientations, Head, db_threshold=1,
                            sort=True, limits=(-20, 20), ear='left',
                            plane='median')
# %%
# ------------------- PLOT: MEAN SIGNED SPECTRAL DIFFERENCE -------------------
ho = Dataset.get_head_orientations(40, 20, 10)
fixed = Dataset.get_head_orientations(0, 0, 0)
ho = far_field_correction(ho)
fixed = far_field_correction(fixed)

target = spharpy.samplings.lebedev(44, radius=ho.source_positions.radius[0])
idx = ho.source_positions.find_nearest(target)[0]

ho_subset = ho.copy()
ho_subset.hrirs = ho_subset.hrirs[:, *idx, :]
ho_subset.source_positions = ho_subset.source_positions[*idx]

fixed_subset = fixed.copy()
fixed_subset.hrirs = fixed_subset.hrirs[:, *idx, :]
fixed_subset.source_positions = fixed_subset.source_positions[*idx]

ho_interp = interpolate_sh(ho, target, 44, rotate=False)
fixed_interpolated = interpolate_sh(fixed, target, 44, rotate=False)
plot_mean_spectral_difference(ho, fixed, limits_db=(-20, 20))
plot_mean_spectral_difference(ho_interp, ho_subset, limits_db=(-20, 20))
plot_mean_spectral_difference(fixed_interpolated, fixed_subset, limits_db=(-20, 20))

# %%

np.testing.assert_almost_equal(ho_interp.source_positions.cartesian,
                               ho_subset.source_positions.cartesian)

# %%
angle = np.linspace(0, 2*np.pi, 180, endpoint=False)
global_median = pf.Coordinates.from_spherical_elevation(0, angle, radius=2)
# global_median.rotate('XYZ', [-40, 0, 0])
output = interpolate_sh(HeadOrientations, global_median, 44, grid='lebedev')
# %%
subplot_spectral_difference(Fixed, output, db_threshold=1,
                            sort=True, limits=(-20, 20), ear='left',
                            plane='median')
# %%
# -------------------- COMPARE 1 M RADIUS VS 2 M RADIUS -----------------------
Dataset_1m = HeadOrientationsDataset.from_directory(base_dir1)
Dataset_2m = HeadOrientationsDataset.from_directory(base_dir2)

ho1 = Dataset_2m.get_head_orientations(0, -55, 0)
ho2 = Dataset_2m.get_head_orientations(0, 0, 0)


plot_single_spectral_difference(ho1, ho2,
                                plane='median',
                                limits=(-20, 20))

# %%
ho3 = interpolate_head_orientation(ho1, ho2, 44)
# %%
# ----------------------- 3D LINEAR INTERPOLATION -----------------------------
sources = ho1.source_positions
points = sources.cartesian
hrirs = ho1.hrirs[0, ...]

hrirs_onset = pf.dsp.resample(hrirs, hrirs.sampling_rate * 10,
                              post_filter=True)
hrirs_onset = pf.dsp.filter.butterworth(hrirs_onset, 10, 3e3)
onsets = pf.dsp.find_impulse_response_start(hrirs_onset) / 10

toa_interpolator = LinearNDInterpolator(points, onsets)

print(onsets.shape)

# %%
# ------------------ LOCALIZATION METRICS: CHECK REPETITIONS ------------------
fixed = Dataset.get_head_orientations(0, 0, 0)
rot = Dataset.get_head_orientations(0, 0, 20)
sampling = spharpy.samplings.equal_area(0, n_points=500)
#%%
fig, ax = plt.subplots(3, 1)
rps = [1, 20, 50, 100, 200, 300, 500]

sampling = spharpy.samplings.equal_area(0, n_points=500)

for rep in rps:
    results = barumerli_localization(rot, fixed, subsampling=sampling,
                                     repetitions=rep)[0]
    ax[0].plot(results["rmsL"], marker='o', label=str(rep))
    ax[1].plot(results["rmsP"], marker='o', label=str(rep))
    ax[2].plot(results["querr"], marker='o', label=str(rep))

ax[0].set_title('rmsL')
ax[1].set_title('rmsP')
ax[2].set_title('querr')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ----------------- LOCALIZATION METRICS: CHECK SUBSAMPLING -------------------
fixed = Dataset.get_head_orientations(0, 0, 0)
rot = Dataset.get_head_orientations(40, 20, 30)

fig, ax = plt.subplots(3, 1)

template_sampling = spharpy.samplings.equal_area(0, n_points=2000)

samplings = [spharpy.samplings.equal_area(0, n_points=500),
             spharpy.samplings.equal_area(0, n_points=1000),
             spharpy.samplings.equal_area(0, n_points=2000),
             spharpy.samplings.lebedev(44, radius=2)]

for sampling in samplings:
    label = str(sampling.csize) if sampling else 'Full sampling'
    results = barumerli_localization(rot, fixed,
                                     template_subsampling=template_sampling,
                                     target_subsampling=sampling,
                                     repetitions=200)[0]
    ax[0].plot(results["rmsL"], marker='o', label=label)
    ax[1].plot(results["rmsP"], marker='o', label=label)
    ax[2].plot(results["querr"], marker='o', label=label)

ax[0].set_title('rmsL')
ax[1].set_title('rmsP')
ax[2].set_title('querr')
fig.suptitle(f"head orientation: {rot.head_orientations[0]}")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# --------------- FORWARD BACKWARDS SH TRANSFORM COMPARISON -------------------
rot = Dataset.get_head_orientations(0, 0, 0)

target = spharpy.samplings.lebedev(44, radius=rot.source_positions.radius[0])
idx = rot.source_positions.find_nearest(target)[0]

hrirs_orig = rot.hrirs[:, *idx]

front = pf.Coordinates.from_spherical_elevation(0/180*np.pi, -90/180*np.pi, 1)

for compute_weights in [True, False]:
    rot_interpolated = interpolate_sh(rot, target, 44, 'lebedev', False,
                                      compute_weights)
    id = rot_interpolated.source_positions.find_nearest(front)[0]

    if compute_weights:
        ax = pf.plot.time_freq(hrirs_orig[0, id, 0], label='original', unit='ms')
    pf.plot.time_freq(rot_interpolated.hrirs[0, id, 0], ax=ax, ls=':',
                      label=str(compute_weights), unit='ms')

# ax = pf.plot.time_freq(hrirs_orig[0, id, 0], label='original', unit='ms')
# pf.plot.time_freq(rot_interpolated.hrirs[0, id, 0], ax=ax, ls=':',
#                   label='interpolated', unit='ms')

# sdif = spectral_difference(hrirs_orig, rot_interpolated.hrirs)
# ax = pf.plot.freq(sdif[0, id, 0])

ax[0].set_title(f"head orientation: {rot.head_orientations[0]}\n"
    f"azimuth: {front.azimuth}, elevation: {front.elevation}")
ax[1].set_ylim(-45, 10)
plt.legend()
plt.show()
# %%
# ------------------- BAYRCENTRIC INTERPOLATION COMPARISON --------------------
hrirs, source, _ = pf.io.read_sofa("/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/sofa/raw_data/bend-10.0_elev-10.0_azim-20.0/HRIR_lebedev44_and_planes.sofa")

td = hrirs.time
coord2find = pf.Coordinates.from_spherical_elevation(0, -np.pi/2, 2)

idx = source.find_nearest(coord2find)[0]

td_excluding = np.delete(td, idx, axis=0)
grid = np.delete(source.cartesian, idx, axis=0)

src = source[idx].cartesian

weights = vbap_weights(grid, src)
interp_hrir = np.einsum('sp,peh->seh', weights, td_excluding)
interp_hrir = pf.Signal(interp_hrir, hrirs.sampling_rate)

expected = hrirs[idx]

ax = pf.plot.time_freq(expected[0])
pf.plot.time_freq(interp_hrir[:, 0], ax=ax, ls=':')
ax[1].set_ylim(-50, 20)
ax[0].set_title(f"azimuth: {coord2find.azimuth}, elevation: {coord2find.elevation}")
plt.show()
# %%
# ----- CREATE TEMPORARY SOFA FILE FOR INTERPOLATED LOCALIZATION METRICS ------
sofa = sf.Sofa("SimpleFreeFieldHRIR")
sofa.SourcePosition = rot_interpolated.source_positions.spherical_elevation

sofa.Data_IR = rot_interpolated.hrirs.time[0]
sofa.Data_SamplingRate = rot_interpolated.hrirs.sampling_rate

with tempfile.TemporaryDirectory() as tmpdir:
    file_path = f"{tmpdir}/example.sofa"

    # Save the SOFA file
    sf.write_sofa(file_path, sofa)

    # You can read it back if needed
    loaded = sf.read_sofa(file_path)

loaded.inspect()

print(rot_interpolated.sofa_file_paths)

# %%
# ------------ LOCALIZATION METRICS: INTERPOLATED VS ORIGINAL -----------------
fig, ax = plt.subplots(3, 1)
loc_sampling = spharpy.samplings.equal_area(0, n_points=2000)
interp_target = spharpy.samplings.lebedev(44, radius=2)

rot = Dataset.get_head_orientations(-30, 10, -40)
rot_interpolated = interpolate_sh(rot, interp_target, 44, 'lebedev', False, True)

for (ho_, label) in zip([rot, rot_interpolated], ["original", "interpolated"]):
    results = barumerli_localization(ho_, ho_, target_subsampling=loc_sampling,
                                     template_subsampling=loc_sampling,
                                     repetitions=200)[0]
    ax[0].plot(results["rmsL"], marker='o', label=label)
    ax[1].plot(results["rmsP"], marker='o', label=label)
    ax[2].plot(results["querr"], marker='o', label=label)

ax[0].set_title('rmsL')
ax[1].set_title('rmsP')
ax[2].set_title('querr')
fig.suptitle(f"head orientation: {rot.head_orientations[0]}")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ------------- LOCALIZATION METRICS: SAVE LOCALIZATION METRICS ---------------
azimuth_rotations = np.arange(-60, 70, 10)
head = Dataset.get_head_orientations(0.5, 0.5, 0.5)

head = far_field_correction(head)
head = directional_transfer_function(head)

for azi_ in azimuth_rotations:
    head_orientations = Dataset.get_head_orientations(None, None, azi_)


    out_dir = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics/natural_head"

    head_orientations = far_field_correction(head_orientations)
    head_orientations = directional_transfer_function(head_orientations)

    template_sampling = spharpy.samplings.equal_area(0, n_points=2000)
    target_sampling = spharpy.samplings.equal_area(0, n_points=1000)

    results = barumerli_localization(
        head_orientations,
        head,
        template_sampling,
        target_sampling,
        out_dir,
        200,
        True)

# %%
# --------------- COLORATION METRICS: SAVE COLORATION METRICS -----------------
azimuth_rotations = np.arange(-60, 70, 10)
head = Dataset.get_head_orientations(0.5, 0.5, 0.5)

head = far_field_correction(head)
head = directional_transfer_function(head)

for azi_ in azimuth_rotations:
    head_orientations = Dataset.get_head_orientations(None, None, azi_)

    out_dir = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/coloration/natural_head"

    head_orientations = far_field_correction(head_orientations)
    head_orientations = directional_transfer_function(head_orientations)

    results = coloration_mc_kenzie(head_orientations,
                                   head,
                                   output_dir=out_dir)

# %%
# ------------------ LOCALIZATION METRICS: METRICS CLASS ----------------------
base_dir = \
    "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_fixed_weighted"
base_dir_nat = \
    "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_natural_nr_weighted"
base_dir_head = \
    "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_head_weighted"


fixed = HeadOrientationsMetrics(base_dir, comment='fixed')
natural = HeadOrientationsMetrics(base_dir_nat, comment='natural')
head = HeadOrientationsMetrics(base_dir_head, comment='head')


flex_range_max = [15, -15]
bend_range_max = [-10, 10]

flex_range = flex_range_max[1] - flex_range_max[0]

gamma = np.deg2rad(np.arange(0, 361))

bend = bend_range_max[1] * np.sin(gamma)**2 * np.sign(np.sin(gamma))
flex = flex_range * ((1-np.cos(gamma)) / 2)**(5/4) - flex_range / 2

rom_rot = np.arange(-30, 40, 10)
loc_metric = 'rmsP'

# plot_localization_map(natural, None, np.arange(0, 70, 10), 'rmsP',
#                       None, cmap='inferno')

if loc_metric == 'rmsP':
    limit_1 = (30, 35)
elif loc_metric == 'querr':
    limit_1 = (10, 15)

print(f"Baumgartner / Llado localization weighted {loc_metric}: natural")
plot_localization_map_subplots(natural, None, np.arange(0, 70, 10), loc_metric,
                               limit_1, cmap='inferno', rom_fill=(bend, -flex),
                               rom_fill_rotation=rom_rot, cols=3)

print(f"Baumgartner / Llado localization {loc_metric}: natural - fixed")
plot_localization_map_subplots(natural, fixed, np.arange(0, 70, 10), loc_metric,
                               (-5, 5), rom_fill=(bend, -flex),
                               rom_fill_rotation=rom_rot, cols=3)

print(f"Baumgartner / Llado localization {loc_metric}: fixed - head")
plot_localization_map_subplots(fixed, head, np.arange(0, 70, 10), loc_metric,
                               (-5, 5), rom_fill=(bend, -flex),
                               rom_fill_rotation=rom_rot, cols=3)

print(f"Baumgartner / Llado localization {loc_metric}: natural - head")
plot_localization_map_subplots(natural, head, np.arange(0, 70, 10), loc_metric,
                               (-5, 5), rom_fill=(bend, -flex),
                               rom_fill_rotation=rom_rot, cols=3)


# %%

# %%

# %%
# ------------------ COLORATION METRICS: METRICS CLASS ----------------------
base_dir_fixed = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/coloration/natural_fixed"
base_dir_head = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/coloration/natural_head"

fixed = HeadOrientationsMetrics(base_dir_fixed, metric_keys=("pbc",),
                                comment='fixed')
head = HeadOrientationsMetrics(base_dir_head, metric_keys=("pbc",),
                                comment='head')

flex_range_max = [15, -15]
bend_range_max = [-10, 10]

flex_range = flex_range_max[1] - flex_range_max[0]

gamma = np.deg2rad(np.arange(0, 361))

bend = bend_range_max[1] * np.sin(gamma)**2 * np.sign(np.sin(gamma))
flex = flex_range * ((1-np.cos(gamma)) / 2)**(5/4) - flex_range / 2

rom_rot = np.arange(-30, 40, 10)

# plot_coloration_map(fixed, np.arange(0, 70, 10), (0, 25),
#                     rom_fill=(bend, -flex),
#                     rom_fill_rotation=rom_rot)

print("McKenzie percieved binaural coloration: natural vs fixed")
plot_coloration_map_subplots(fixed, np.arange(0, 70, 10), (0, 25),
                             rom_fill=(bend, -flex),
                             rom_fill_rotation=rom_rot,
                             cols=3)

print("McKenzie percieved binaural coloration: natural vs head")
plot_coloration_map_subplots(head, np.arange(0, 70, 10), (0, 25),
                             rom_fill=(bend, -flex),
                             rom_fill_rotation=rom_rot,
                             cols=3)

# %%
# ------------ HRTF NORMALIZATION: FARFIELD CORRECTION & DTF ------------------
head_orientations = Dataset.get_head_orientations(None, 0, 0)
ax = pf.plot.time_freq(head_orientations.hrirs[0, 0, 0], label='raw')
head_orientations = far_field_correction(head_orientations)
pf.plot.time_freq(head_orientations.hrirs[0, 0, 0],
                  label='far-field corrected', ax=ax)
head_orientations = directional_transfer_function(head_orientations)
pf.plot.time_freq(head_orientations.hrirs[0, 0, 0],
                  label='dtf', ax=ax)
plt.legend()
plt.show()
# %%
# ------------------ COLORATION: ORIGINAL VS INTERPOLATED ---------------------
rot = Dataset.get_head_orientations(40, 0, 0)

target = spharpy.samplings.lebedev(44, radius=rot.source_positions.radius[0])
idx = rot.source_positions.find_nearest(target)[0]

rot_interpolated = interpolate_sh(rot, target, 44, 'lebedev', False)

fixed = Dataset.get_head_orientations(0, 0, 0)
fixed.hrirs = fixed.hrirs[:, *idx]
fixed.source_positions = fixed.source_positions[idx]

rot.hrirs = rot.hrirs[:, *idx]
rot.source_positions = rot.source_positions[idx]

for (ho_, label) in zip([rot, rot_interpolated], ["original", "interpolated"]):
    plt.figure()

    coloration = coloration_mc_kenzie(ho_, fixed)[0]

    spharpy.plot.contour(target, coloration, limits=(0, 100))
    plt.xlabel('Azimuth in °')
    plt.ylabel('Elevation in °')
    plt.title(label + f" (mean coloration: {np.mean(coloration):.2f})")
    plt.tight_layout()
    plt.show()

#%%
# --------------------------- MEAN METRICS IN FROM ----------------------------
flex_range_max = [-15, 15]
bend_range_max = [-10, 10]

flex_range = flex_range_max[1] - flex_range_max[0]

gamma = np.deg2rad(np.arange(0, 361))

bend = bend_range_max[1] * np.sin(gamma)**2 * np.sign(np.sin(gamma))
flex = flex_range * ((1-np.cos(gamma)) / 2)**(5/4) - flex_range / 2

#fixed = HeadOrientationsMetrics(base_dir_fixed, metric_keys=('pbc',), comment='fixed')

flex_range_max = [-15, 15]
bend_range_max = [-10, 10]
rot_range = (-30, 40)

flex_range_max = [-60, 60]
bend_range_max = [-60, 60]
rot_range = (-60, 70)

flex_range = flex_range_max[1] - flex_range_max[0]

gamma = np.deg2rad(np.arange(0, 361))

bend = bend_range_max[1] * np.sin(gamma)**2 * np.sign(np.sin(gamma))
flex = flex_range * ((1-np.cos(gamma)) / 2)**(5/4) - flex_range / 2

for rot_ in np.arange(*rot_range, 10):
    # Find HOIs within this polygon region with 5° tolerance
    mask = find_indices_in_region(fixed.head_orientations, bend, -flex,
                                  tolerance=0.5, rotation=rot_)
    indices = np.where(mask)[0]
    print(f'Rotation: {rot_} degree')
    print(f"natural: {np.mean(natural.querr[indices]):.2f}",
          f"fixed: {np.mean(fixed.querr[indices]):.2f}",
          f"head: {np.mean(head.querr[indices]):.2f}"
          "\n---------------------------------------")

    # print(f"fixed: {np.mean(fixed.querr[indices]):.2f}",
    #       f"head: {np.mean(head.querr[indices]):.2f}"
    #       "\n---------------------------------------")

# %%
# --------------------------- PLOT HRTFS IN PLANE -----------------------------
fixed = Dataset.get_head_orientations(0, 0, 0)
#plot_single_spectral_difference(Dataset.get_head_orientations(0, 0, 0), None, 'median', limits=(-60, 20))
fig = plt.figure(dpi=300)
plot_single_spectral_difference(Dataset.get_head_orientations(0, 0, 0), None, 'median', ear='left', limits=(-25, 25))
plot_single_spectral_difference(Dataset.get_head_orientations(-30, 0, 0), None, 'median', ear='right', limits=(-25, 25))

# %%
# -------------------------- GREAT CIRLCE DISTANCE ----------------------------
def great_circle_distance(m):
    az1 = m[:, 0]
    el1 = m[:, 1]
    az2 = m[:, 2]
    el2 = m[:, 3]
    az1, el1, az2, el2 = np.radians([az1, el1, az2, el2])
    a = np.sin((el1 - el2) / 2)**2 + \
        np.cos(el1) * np.cos(el2) * np.sin((az1 - az2) / 2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

cmap = mcolors.LinearSegmentedColormap.from_list(
    'blue_white_red',
    ['blue', 'white', 'red'],
    N=256,
)

case = 'natural_fixed'
matrix_dir = f"/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics/{case}/prediction_matrices"
b, e, a = [0, -55, 0]
file = f'matrix_bend_{b}elev_{e}azim{a}.mat'
loc_matrix = sc.io.loadmat(os.path.join(matrix_dir, file))

loc_matrix = loc_matrix['prediction_matrix'].reshape((200, 1000, 8))
loc_matrix = np.mean(loc_matrix, axis=0)

case2 = 'natural_natural'
matrix_dir = f"/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics/{case2}/prediction_matrices"
b, e, a = [0, -55, 0]
file = f'matrix_bend_{b}elev_{e}azim{a}.mat'
loc_matrix2 = sc.io.loadmat(os.path.join(matrix_dir, file))

loc_matrix2 = loc_matrix2['prediction_matrix'].reshape((200, 1000, 8))
loc_matrix2 = np.mean(loc_matrix2, axis=0)

coords = pf.Coordinates.from_spherical_elevation(loc_matrix[:, 0]/180*np.pi,
                                                 loc_matrix[:, 1]/180*np.pi, 1)

spharpy.plot.pcolor_map(coords,
                        great_circle_distance(loc_matrix2) - great_circle_distance(loc_matrix),
                        cmap=cmap, limits=(-10, 10))


plt.title(f"gcd natural - gcd fixed\nbend: {b}; elev: {e}, azimuth: {a}")
plt.show()

spharpy.plot.pcolor_map(coords,
                        great_circle_distance(loc_matrix2), limits=(0, 100))
plt.title(f"{case2}\nbend: {b}; elev: {e}, azimuth: {a}")
plt.show()

# %%
# ----------------- COLORATION: PLOT COLORATION OVER SOURCE -------------------
fp_natural = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/sofa/raw_data/bend-10.0_elev-30.0_azim-50.0/HRIR_lebedev44_and_planes.sofa'
_, source, _ = pf.io.read_sofa(fp_natural)

base_dir_fixed = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/coloration/natural_fixed"

fixed = HeadOrientationsMetrics(base_dir_fixed, metric_keys=("pbc",),
                                comment='fixed')

ho = fixed.head_orientations
pbc = fixed.pbc

target = np.array([-40, 0, 0])

mask = np.all(ho == target, axis=1)   # True where row matches target
idx = np.where(mask)[0]

spharpy.plot.pcolor_map(source, pbc[*idx], limits=(0, 100))
plt.title(f"pbc\nbend: {target[0]}; elev: {target[1]}, azimuth: {target[2]}")

plt.show()
# %%
# -------------- LOCALIZATION METRICS: SAVE BAUMGARTNER / LLADO ---------------
# Start matlab engine
eng = matlab.engine.start_matlab()
eng.amt_start(nargout=0)
eng.SOFAstart(nargout=0)
# get weights and S and gamma parameters
eng.eval("tmp = amt_load('llado2025', 'w21_zonooz.mat');", nargout=0)
weights = eng.eval("tmp.cache.value.data.weights", nargout=1)
S = np.asarray(eng.eval("tmp.cache.value.data.S", nargout=23))
gamma = np.asarray(eng.eval("tmp.cache.value.data.gamma", nargout=23))

gamma_median = np.median(gamma)
S_median = np.median(S)

eng.eval("tmp = amt_load('llado2025', 'w1_flat.mat');", nargout=0)
weights_flat = eng.eval("tmp.cache.value.data.weights", nargout=1)
S_flat = np.asarray(eng.eval("tmp.cache.value.data.S", nargout=23))
gamma_flat = np.asarray(eng.eval("tmp.cache.value.data.gamma", nargout=23))

gamma_median_flat = np.median(gamma_flat)
S_median_flat = np.median(S_flat)

print(f"S\n--\nflat: {S_median_flat}\nNR: {S_median}\n")
print(f"gamma\n-----\nflat: {gamma_median_flat}\nNR: {gamma_median}\n")

# Calculate metrics per rotation

azimuth_rotations = np.arange(-60, 70, 10)
head = Dataset.get_head_orientations(0.5, 0.5, 0.5)

head = far_field_correction(head)
head = directional_transfer_function(head)

fixed = Dataset.get_head_orientations(0, 0, 0)

fixed = far_field_correction(fixed)
fixed = directional_transfer_function(fixed)

for azi_ in azimuth_rotations:
    ho = Dataset.get_head_orientations(None, None, azi_)

    ho = far_field_correction(ho)
    ho = directional_transfer_function(ho)

    results_natural_weighted = baumgartner_localization(ho, ho,
                                                        '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_natural_dt_weighted',
                                                        spectral_weighting='DT',
                                                        gamma=gamma_median,
                                                        S=S_median
                                                        )
    # results_natural_unweighted = baumgartner_localization(ho, ho,
                                                        # '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_natural_unweighted',
                                                        # spectral_weighting=weights_flat,
                                                        # gamma=gamma_median_flat,
                                                        # S=S_median_flat)
# %%
# ---------------------- PLOT LLADO SPECTRAL WEIGHTS --------------------------
import numpy as np

def erb_rate(f):
    return 21.4 * np.log10(4.37e-3 * f + 1)

def erb2hz(erb):
    return (10**(erb/21.4) - 1) / 4.37e-3

def erbspacebw(flow, fhigh, bw=1):
    audlimits = erb_rate(np.array([flow, fhigh]))
    audrange = audlimits[1] - audlimits[0]

    n = np.floor(audrange / bw)
    remainder = audrange - n * bw

    audpoints = audlimits[0] + np.arange(n + 1) * bw + remainder / 2

    return erb2hz(audpoints)

fc = erbspacebw(700, 18000)
weights = np.asarray(weights).squeeze()

weighting_NR = pf.FrequencyData(weights, fc)

plt.figure(figsize=(8, 3))
pf.plot.freq(weighting_NR, dB=False)
plt.xlim(200, 20e3)
plt.show()
# %%
# ------ CREATE PLOTS: LOCALIZATION MODEL COMPARISON FOR ISOLATED DOFs --------
bmgtn_dir_weighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_natural_weighted'
bmgtn_metrics_weighted = HeadOrientationsMetrics(bmgtn_dir_weighted)

bmgtn_dir_unweighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_natural_unweighted'
bmgtn_metrics_unweighted = HeadOrientationsMetrics(bmgtn_dir_unweighted)

bmgtn_fixed_dir_weighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_fixed_weighted'
bmgtn_fixed_metrics_weighted = HeadOrientationsMetrics(bmgtn_fixed_dir_weighted)

bmgtn_fixed_dir_unweighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_fixed_unweighted'
bmgtn_fixed_metrics_unweighted = HeadOrientationsMetrics(bmgtn_fixed_dir_unweighted)

bmgtn_head_dir_weighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_head_weighted'
bmgtn_head_metrics_weighted = HeadOrientationsMetrics(bmgtn_head_dir_weighted)

bmgtn_head_dir_unweighted = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_head_unweighted'
bmgtn_head_metrics_unweighted = HeadOrientationsMetrics(bmgtn_head_dir_unweighted)

brmrli_dir = "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics/natural_natural"
brmrli_metrics = HeadOrientationsMetrics(brmrli_dir)

metric = 'querr'
limits = (23, 42.5)
limits = (5, 27)

ax = plot_localization_1_dof(bmgtn_metrics_unweighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='natural unweighted',color='b', marker='o', alpha=0.3)

plot_localization_1_dof(bmgtn_metrics_weighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='natural weighted', color='b', marker='o')

plot_localization_1_dof(bmgtn_fixed_metrics_unweighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='fixed unweighted',color='r', marker='x', alpha=0.3)

plot_localization_1_dof(bmgtn_fixed_metrics_weighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='fixed weighted', color='r', marker='x')

plot_localization_1_dof(bmgtn_head_metrics_weighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='head weighted', color='g', marker='v')

plot_localization_1_dof(bmgtn_head_metrics_unweighted.get_subset(None, 0, 0), metric,
                             limits=limits, label='head unweighted', color='g', marker='v', alpha=0.3)

plot_localization_1_dof(brmrli_metrics.get_subset(None, 0, 0), metric,
                             limits=limits, label='barumerli', color='grey', marker='.')

plt.title('Bend')
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2  # optional: arrange entries in multiple columns
)
plt.xlabel('head orientation angle in degree')
plt.show()
# %%
# ------------------------- EXPORT METRICS TO EXCEL ---------------------------
base_dir_nat = \
    "/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner/natural_head_weighted"
natural = HeadOrientationsMetrics(base_dir_nat, comment='natural')

df = pd.DataFrame()
df['bend'] = natural.head_orientations[:, 0]
df['elev'] = natural.head_orientations[:, 1]
df['azim'] = natural.head_orientations[:, 2]
df['querr'] = natural.querr
df['rmsP'] = natural.rmsP

out_dir = '/Users/antonhoyer/Documents/HATO_Maya_Model_V4/_results/2m radius/localization_metrics_baumgartner'
df.to_csv(os.path.join(out_dir, 'natural_head.csv'))

