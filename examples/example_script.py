# %%
import logging
import pickle
from pathlib import Path

import mne
import numpy as np
from mne.datasets import sample

from cmb import (
    create_cerebellar_surface,
    get_cerebellum_data,
    get_subsampled_cerebellum_labels,
    join_cortical_source_spaces,
    segment_cerebellum,
    setup_full_source_space,
)
from cmb import (
    visualization as cmb_viz,
)

# Control the verbosity of CMB functions.
logging.basicConfig(level=logging.INFO)

# Where the cerebellum data is stored.
# Data will be downloaded to this location if not already present.
cmb_dir = Path("/u/69/taivait1/unix/cmb_data")

# Set paths to subject data.
data_path = Path(sample.data_path())

subject = "sample"
subjects_dir = data_path / "subjects"

sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
trans = sample_dir / "sample_audvis_raw-trans.fif"
fname_cov = sample_dir / "sample_audvis-cov.fif"
evoked_fname = sample_dir / "sample_audvis-ave.fif"

# %% Check if the required data are available and download if not.
get_cerebellum_data(cmb_dir=cmb_dir)

# %% Set source space parameters.
# Use spacing 2 to get an approximately equal grid density in cerebral
# and cerebellar cortices
cerebral_spacing = 2
cerebellum_subsampling = "sparse"

# %% Segment the cerebellum (or get existing segmentation).
segmentation = segment_cerebellum(
    subject,
    subjects_dir,
    cmb_dir,
    segmentation_fname=None,  # use the default location
)

# %% Fit the atlas to the cerebellum of the subject,
# yielding a cerebellar mesh in the subject space.
rr, tris = create_cerebellar_surface(
    subject,
    segmentation,
    subjects_dir,
    cmb_dir,
    cerebellum_subsampling=cerebellum_subsampling,
    save_mesh=True,  # use the default location
)
print(f"Cerebellar mesh created with {rr.shape[0]} vertices and {tris.shape[0]} faces.")
# The default save location.
mesh_fname = (
    subjects_dir / subject / "surf" / f"cerebellum_{cerebellum_subsampling}.white"
)

# %% Visualize the cerebellar mesh in the subject's MRI volume.
_ = cmb_viz.plot_sagittal(
    vol_fname=subjects_dir / subject / "mri" / "T1.mgz",
    mesh_fname=mesh_fname,
)

# %% # Setup source space with cerebellum and cortex using the created mesh.
# Newer MNE versions (in October 2026 only the dev version) can build the cerebellar
# source space directly from the mesh, giving a mixed source space [lh, rh, cerebellum].
# With older MNE versions, CMB handles source space building as [cortex, cerebellum],
# where the two cortical hemispheres are joined into one. The rest of this example works
# with both layouts.
use_mne_mixed_api = hasattr(mne, "setup_subcortical_source_space")

if use_mne_mixed_api:
    src_cerebellum = mne.setup_subcortical_source_space(  # pyright: ignore[reportAttributeAccessIssue]
        subject=subject,
        surface=mesh_fname,
        subjects_dir=subjects_dir,
        keep_largest_component=False,  # keep vertex indices aligned with the atlas
        add_dist=False,
    )
    src_cortex = mne.setup_source_space(
        subject=subject,
        subjects_dir=subjects_dir,
        spacing=cerebral_spacing,  # pyright: ignore[reportArgumentType]
        add_dist=False,
    )
    src = src_cortex + src_cerebellum
else:
    src = setup_full_source_space(
        subject,
        cerebellum_subsampling,
        subjects_dir,
        cerebellum_surf_fname=None,  # find from the default location
        spacing=cerebral_spacing,
    )

# %% Compute forward and inverse operators
conductivity = (0.3, 0.006, 0.3)
# Important not to use too large mindist because the cerebellar cortex and inner skull
# boundary are usually within 5 mm.
mindist = 3.0

model = mne.make_bem_model(
    subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)
# If too many source space points are lost, the inner skull boundary is too tight and
# needs to be expanded.
info = mne.io.read_info(raw_fname)
fwd = mne.make_forward_solution(
    info, trans, src, bem=bem, mindist=mindist, eeg=True, n_jobs=1
)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)

noise_cov = mne.read_cov(fname_cov)
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info,
    fwd,
    noise_cov,
    depth=None,  # pyright: ignore[reportArgumentType]
    fixed=True,  # pyright: ignore[reportArgumentType]
)

# %% Load the cerebellum geometry for simulations and visualization.
with open(cmb_dir / "data" / "cerebellum_geo", "rb") as f:
    cb_data = pickle.load(f)

# %% Get the cortical and cerebellar parts of the source space.
# Cerebellum source space is the last element in the SourceSpaces list.
cerebellum_src_index = len(fwd["src"]) - 1
# Cortical sources precede the cerebellar ones in the leadfield columns.
n_cortex_vertices = sum(s["nuse"] for s in fwd["src"][:cerebellum_src_index])
# The cmb plotting functions expect the cortex as a single source space, so join
# the hemispheres of a mixed [lh, rh, cerebellum] source space.
if cerebellum_src_index == 1:
    fwd_cortex_src = fwd["src"][0]
else:
    fwd_cortex_src = join_cortical_source_spaces(fwd["src"][:2])

# %% Example forward simulation from patch in right lobule VIIIa

labels = get_subsampled_cerebellum_labels(
    cb_data, subsampling=cerebellum_subsampling, set_hemi_cerebellum=True
)
label = labels[714]
cb_fwd_vertices = fwd["src"][cerebellum_src_index]["vertno"]

cerebellum_active_verts = np.where(np.isin(cb_fwd_vertices, label.vertices))[0]
cerebellum_activation = np.zeros(fwd["src"][cerebellum_src_index]["nuse"])
cerebellum_activation[cerebellum_active_verts] = 1

# %% Plot the cerebellar activation patch

# Need to interpolate the cerebellar data to the full chosen subsampling
# (sparse or dense) for visualization.
cerebellum_data_prepared = cmb_viz.morph_cerebellum_data(
    data=cerebellum_activation,
    fwd_cerebellum_src=fwd["src"][cerebellum_src_index],
    cerebellum_geo=cb_data,
    subsampling=cerebellum_subsampling,
    smoothing_steps=0,
)
# Normal 3D surface plot
_ = cmb_viz.plot_normal(
    src_cerebellum=fwd["src"][cerebellum_src_index],
    cerebellum_data=cerebellum_data_prepared,
    src_cortex=fwd_cortex_src,  # pass this to plot the cortex as well
    cortex_data=None,  # no data to plot on the cortex
    clim=None,  # determine automatically
    cmap="Reds",
)
# Inflated 3D plot of just cerebellum.
_ = cmb_viz.plot_inflated(
    cerebellum_geo=cb_data,
    cerebellum_data=cerebellum_data_prepared,
    subsampling=cerebellum_subsampling,
    clim=None,
    cmap="Reds",
)
# Flatmap plot of just cerebellum.
_ = cmb_viz.plot_flatmap(
    cb_data,
    cerebellum_data_prepared,
    cerebellum_subsampling,
    cmap=None,
    clim=None,
    offscreen=False,
    screenshot_fname=None,
)

# %% Project the simulated cerebellar activation to the sensor space.
# Read real evoked to use as a template.
evoked = mne.read_evokeds(evoked_fname)[0]  # pyright: ignore[reportIndexIssue]

channels = mne.pick_types(evoked.info, meg=True, eeg=True, exclude=[])  # pyright: ignore[reportArgumentType]
leadfield = fwd["sol"]["data"]

simulated_measurement = np.zeros(evoked.info["nchan"])
simulated_measurement[channels] = np.sum(
    leadfield[:, n_cortex_vertices + cerebellum_active_verts] * 10**-7, axis=1
)
# Overwrite all time points in the evoked data with the simulated measurement.
evoked._data[channels] = np.repeat(
    simulated_measurement[channels].reshape((len(channels), 1)),
    repeats=evoked._data.shape[1],
    axis=1,
)

# %% Estimate source activation from the simulated data.
estimate = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, 1 / 9, "sLORETA", verbose="WARNING"
)
assert isinstance(estimate, (mne.SourceEstimate, mne.MixedSourceEstimate))

# %% Extract the estimated activation for cerebellum and cortex.
estimate_cortex, estimate_cerebellum = cmb_viz.get_plot_data_from_stc(
    stc=estimate,
    fwd_src=fwd["src"],
    time_point=0,
    cerebellum_geo=cb_data,
    cerebellum_subsampling=cerebellum_subsampling,
    cerebellum_idx=cerebellum_src_index,
    cortex_smooth=None,
    cerebellum_smooth=0,
)

# %% Plot the estimated activation on cerebellum and cortex.

_ = cmb_viz.plot_normal(
    src_cerebellum=fwd["src"][cerebellum_src_index],
    cerebellum_data=estimate_cerebellum,
    src_cortex=fwd_cortex_src,
    cortex_data=estimate_cortex,
    cmap="coolwarm",
)
_ = cmb_viz.plot_flatmap(
    cb_data,
    estimate_cerebellum,
    cerebellum_subsampling,
    cmap="coolwarm",
)

# %% Sensitivity maps - cerebellum only
for ch_type in ["mag", "grad", "eeg"]:
    ch_inds = mne.channel_indices_by_type(fwd["info"])
    signal_norms_cb = np.linalg.norm(
        fwd["sol"]["data"][ch_inds[ch_type], n_cortex_vertices:], axis=0
    )
    signal_norms_cb_prepared = cmb_viz.morph_cerebellum_data(
        signal_norms_cb,
        fwd["src"][cerebellum_src_index],
        cb_data,
        cerebellum_subsampling,
        smoothing_steps=0,
    )

    _ = cmb_viz.plot_normal(
        fwd["src"][cerebellum_src_index], signal_norms_cb_prepared, cmap="Reds"
    )
    _ = cmb_viz.plot_inflated(
        cb_data, signal_norms_cb_prepared, cerebellum_subsampling, cmap="Reds"
    )
    _ = cmb_viz.plot_flatmap(
        cb_data, signal_norms_cb_prepared, cerebellum_subsampling, cmap="Reds"
    )

# %% Sensitivity maps - with cortex
for ch_type in ["mag", "grad", "eeg"]:
    ch_inds = mne.channel_indices_by_type(fwd["info"])
    signal_norms = np.linalg.norm(fwd["sol"]["data"][ch_inds[ch_type], :], axis=0)
    signal_norms_cortex_prepared = cmb_viz.morph_cortex_data(
        cort_data=signal_norms[:n_cortex_vertices], fwd_cortex_src=fwd_cortex_src
    )
    signal_norms_cerebellum_prepared = cmb_viz.morph_cerebellum_data(
        data=signal_norms[n_cortex_vertices:],
        fwd_cerebellum_src=fwd["src"][cerebellum_src_index],
        cerebellum_geo=cb_data,
        subsampling=cerebellum_subsampling,
        smoothing_steps=0,
    )
    _ = cmb_viz.plot_normal(
        src_cerebellum=fwd["src"][cerebellum_src_index],
        cerebellum_data=signal_norms_cerebellum_prepared,
        src_cortex=fwd_cortex_src,
        cortex_data=signal_norms_cortex_prepared,
        cmap="Reds",
        clim=None,
    )
