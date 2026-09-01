# %%
import pickle
from pathlib import Path

import mne
import numpy as np
from mne.datasets import sample

from cmb import (
    CMB_DATA_DIR,
    create_cerebellar_surface,
    get_cerebellum_data,
    segment_cerebellum,
    setup_full_source_space,
)
from cmb import (
    visualization as cmb_viz,
)

# Set paths to subject data.
data_path = Path(sample.data_path())

subject = "sample"
subjects_dir = data_path / "subjects"

sample_dir = data_path / "MEG" / "sample"
raw_fname = sample_dir / "sample_audvis_raw.fif"
trans = sample_dir / "sample_audvis_raw-trans.fif"
fname_cov = sample_dir / "sample_audvis-cov.fif"
evo_fname = sample_dir / "sample_audvis-ave.fif"

# %% Check if the required data are available and download if not.
# Use the default location (CMB_DATA_DIR) for the data.
get_cerebellum_data(cmb_path=None)

# %% Set parameters.
# Use spacing 2 to get an approximately equal grid density in cerebral
# and cerebellar cortices
cerebral_spacing = 2
cerebellum_subsampling = "sparse"

# %% Segment the cerebellum.
segmentation = segment_cerebellum(subject, subjects_dir)

# %% Fit the atlas to the cerebellum of the subject,
# yielding a cerebellar mesh in the subject space.
rr, tris = create_cerebellar_surface(
    subject,
    segmentation,
    subjects_dir,
    cerebellum_subsampling=cerebellum_subsampling,
    # save to <subjects_dir>/<subject>/surf/cerebellum_<cerebellum_subsampling>.white
    save_mesh=True,
)
print(f"Cerebellar mesh created with {rr.shape[0]} vertices and {tris.shape[0]} faces.")

# %% Visualize the cerebellar mesh in the subject's MRI volume.
_ = cmb_viz.plot_sagittal(
    vol_fname=subjects_dir / subject / "mri" / "orig.mgz",
    mesh_fname=subjects_dir
    / subject
    / "surf"
    / f"cerebellum_{cerebellum_subsampling}.white",
)

# %% # Setup source space with cerebellum and cortex using the created mesh.
src_whole = setup_full_source_space(
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
    info, trans, src_whole, bem=bem, mindist=mindist, eeg=True, n_jobs=1
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
with open(Path(CMB_DATA_DIR) / "data" / "cerebellum_geo", "rb") as f:
    cb_data = pickle.load(f)

# %% Example forward simulation from patch in right lobule VIIIa
label = cb_data["parcellation"]["fine labels"][714]
active_verts = np.where(
    np.isin(cb_data["dw_data"][cerebellum_subsampling], label.vertices)
)[0]
active_verts = np.where(np.isin(fwd["src"][1]["vertno"], active_verts))[0]
act_cerb = np.zeros(fwd["src"][1]["nuse"])
act_cerb[active_verts] = 1

# %% Plot the cerebellar activation patch

# Need to interpolate the cerebellar data to the full chosen subsampling
# (sparse or dense) for visualization.
cerebellum_data_prepared = cmb_viz.morph_cerebellum_data(
    act_cerb, fwd["src"][1], cb_data, cerebellum_subsampling, smoothing_steps=0
)

# Normal 3D surface plot
_ = cmb_viz.plot_normal(
    src_cerebellum=fwd["src"][1],
    cerebellum_data=cerebellum_data_prepared,
    src_cortex=fwd["src"][0],  # pass this to plot the cortex as well
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
# %% Estimate activation from simulated data
evo = mne.read_evokeds(evo_fname)[0]  # pyright: ignore[reportIndexIssue]
sens = np.zeros(evo.info["nchan"])
all_chs = mne.pick_types(evo.info, meg=True, eeg=True, exclude=[])  # pyright: ignore[reportArgumentType]
sens[all_chs] = np.sum(
    fwd["sol"]["data"][:, fwd["src"][0]["nuse"] + active_verts] * 10**-7, axis=1
)
evo._data[all_chs] = np.repeat(
    sens[all_chs].reshape((len(all_chs), 1)), repeats=evo._data.shape[1], axis=1
)
estimate = mne.minimum_norm.apply_inverse(
    evo, inverse_operator, 1 / 9, "sLORETA", verbose="WARNING"
)
assert isinstance(estimate, mne.SourceEstimate)
estimate_data = estimate.data
assert isinstance(estimate_data, np.ndarray)
# How many cortex vertices used in the forward solution.
n_verts_cortex_fwd = fwd["src"][0]["nuse"]

estimate_cerebellum = np.linalg.norm(estimate_data[n_verts_cortex_fwd:, :], axis=1)
estimate_cortex = np.linalg.norm(estimate_data[:n_verts_cortex_fwd, :], axis=1)

# %% Plot the estimated activation on cerebellum and cortex.
cerebellum_estimate_prepared = cmb_viz.morph_cerebellum_data(
    data=estimate_cerebellum,
    fwd_cerebellum_src=fwd["src"][1],
    cerebellum_geo=cb_data,
    subsampling=cerebellum_subsampling,
    smoothing_steps=0,
)
cortex_estimate_prepared = cmb_viz.morph_cortex_data(
    cort_data=estimate_cortex, fwd_cortex_src=fwd["src"][0]
)

_ = cmb_viz.plot_normal(
    src_cerebellum=fwd["src"][1],
    cerebellum_data=cerebellum_estimate_prepared,
    src_cortex=fwd["src"][0],
    cortex_data=cortex_estimate_prepared,
    cmap="Reds",
    clim=(0, 10000),
)
_ = cmb_viz.plot_flatmap(
    cb_data,
    cerebellum_estimate_prepared,
    cerebellum_subsampling,
    cmap="Reds",
    clim=(0, 10000),
)

# %% Sensitivity maps - cerebellum only
for ch_type in ["mag", "grad", "eeg"]:
    ch_inds = mne.channel_indices_by_type(fwd["info"])
    signal_norms_cb = np.linalg.norm(
        fwd["sol"]["data"][ch_inds[ch_type], fwd["src"][0]["nuse"] :], axis=0
    )
    signal_norms_cb_prepared = cmb_viz.morph_cerebellum_data(
        signal_norms_cb,
        fwd["src"][1],
        cb_data,
        cerebellum_subsampling,
        smoothing_steps=0,
    )

    _ = cmb_viz.plot_normal(fwd["src"][1], signal_norms_cb_prepared, cmap="Reds")
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
        cort_data=signal_norms[:n_verts_cortex_fwd], fwd_cortex_src=fwd["src"][0]
    )
    signal_norms_cerebellum_prepared = cmb_viz.morph_cerebellum_data(
        data=signal_norms[n_verts_cortex_fwd:],
        fwd_cerebellum_src=fwd["src"][1],
        cerebellum_geo=cb_data,
        subsampling=cerebellum_subsampling,
        smoothing_steps=0,
    )
    _ = cmb_viz.plot_normal(
        src_cerebellum=fwd["src"][1],
        cerebellum_data=signal_norms_cerebellum_prepared,
        src_cortex=fwd["src"][0],
        cortex_data=signal_norms_cortex_prepared,
        cmap="Reds",
        clim=None,
    )
