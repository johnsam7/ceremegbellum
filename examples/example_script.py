"""
====================
01. Running the code
====================

"""
###############################################################################
# First import the necessary python modules:

import os.path as op
import mne
import pickle
import numpy as np
from mne.datasets import sample

from cmb import (get_cerebellum_data,
                 setup_full_source_space,
                 plot_cerebellum_data)
from cmb.segmentation import segment_cerebellum

# %%
# Then set the paths to your data:

data_path = sample.data_path()
cmb_path = '/autofs/cluster/fusion/gbm6/Projects/cmb/cmb_data/'

# %%
# The following function will check for the presence of the necessary
# cerebellum files, and will attempt to copy them over to the given
# path if they are not present. Currently this only works if you
# have access to the martinos network cluster storage.
get_cerebellum_data(cmb_path)

# %%
# Setup the relevant source files:
sample_dir = op.join(data_path, 'MEG', 'sample',)
raw_fname = op.join(sample_dir, 'sample_audvis_raw.fif')
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
trans = op.join(sample_dir, 'sample_audvis_raw-trans.fif')
fname_cov = op.join(sample_dir, 'sample_audvis-cov.fif')
evo_fname = op.join(sample_dir,'sample_audvis-ave.fif')


# %%
# Load cerebellar data and define settings
cb_data = pickle.load(open(op.join(cmb_path,'data','cerebellum_geo'), 'rb'))
spacing = 2 # Use spacing 2 to get an approximately equal grid density in cerebral and cerebellar cortices

# %%
# Get subject segmentation
segment_cerebellum(subjects_dir, subject, cmb_path, debug_mode=False)

# %%
# Setup source space using the segmented data
cerebellum_subsampling = 'dense'
src_cort = mne.setup_source_space(subject=subject, subjects_dir=subjects_dir, spacing=spacing, add_dist=False)
src_whole = setup_full_source_space(subject, subjects_dir, cmb_path, cerebellum_subsampling,
                                    plot_cerebellum=False, spacing=spacing,debug_mode=False)

# %%
# Compute forward and inverse operators
conductivity=(0.3, 0.006, 0.3)
mindist = 3.0 
# important not to use too large mindist because the cerebellar cortex and inner skull boundary are usually within 5 mm
model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model) 
# IF too many source space points are lost, the inner skull boundary is too tight and need to be expanded.
info = mne.io.read_info(raw_fname)
fwd = mne.make_forward_solution(info, trans, src_whole, bem=bem, mindist=mindist, eeg=True, n_jobs=1)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)
noise_cov = mne.read_cov(fname_cov)
inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, depth=None, fixed=True)

# %%
# Example forward simulation from patch in right lobule VIIIa
label =  cb_data['parcellation']['fine labels'][714]
active_verts = np.where(np.isin(cb_data['dw_data'][cerebellum_subsampling], label.vertices))[0]
active_verts = np.where(np.isin(fwd['src'][1]['vertno'], active_verts))[0]
act_cerb = np.zeros((fwd['src'][1]['nuse']))
act_cerb[active_verts] = 1

# %%
# Plot activated patch (set view to 'all' if not remote connecting)
plot_cerebellum_data(act_cerb, fwd['src'], src_whole, cb_data, cort_data=np.zeros(fwd['src'][0]['nuse']), flatmap_cmap='bwr', mayavi_cmap='OrRd',
                     smoothing_steps=0, view='flatmap', sub_sampling=cerebellum_subsampling, cmap_lims=[0,100])

# %%
# Plot estimated activation (set view to 'all' if not remote connecting)
evo = mne.read_evokeds(evo_fname)[0]
sens = np.zeros(evo.info['nchan'])
all_chs = mne.pick_types(evo.info, meg=True, eeg=True, exclude=[])
sens[all_chs] = np.sum(fwd['sol']['data'][:, fwd['src'][0]['nuse']+active_verts]*10**-7,axis=1)
evo._data[all_chs] = np.repeat(sens[all_chs].reshape((len(all_chs),1)), repeats=evo._data.shape[1], axis=1)
estimate = mne.minimum_norm.apply_inverse(evo, inverse_operator, 1/9, 'sLORETA', verbose='WARNING')
estimate_cerb = np.linalg.norm(estimate.data[fwd['src'][0]['nuse']:estimate.shape[0], :], axis=1)
cort_data = np.linalg.norm(estimate.data[:fwd['src'][0]['nuse'], :], axis=1)
plot_cerebellum_data(estimate_cerb, fwd['src'], src_whole, cb_data, cort_data=cort_data, flatmap_cmap='bwr',
                         mayavi_cmap='OrRd', smoothing_steps=0, view='flatmap', sub_sampling=cerebellum_subsampling,
                         cmap_lims=[25,75])

# %%
# Sensitivity maps - cerebellum only
for ch_type in ['mag', 'grad', 'eeg']:
    ch_inds = mne.channel_indices_by_type(fwd['info'])
    signal_norms = np.linalg.norm(fwd['sol']['data'][ch_inds[ch_type], fwd['src'][0]['nuse']:], axis=0)
    plot_cerebellum_data(signal_norms, fwd['src'], src_whole, cb_data, cort_data=np.zeros(fwd['src'][0]['nuse']), flatmap_cmap='bwr',
                             mayavi_cmap='OrRd', smoothing_steps=0, view='flatmap', sub_sampling=cerebellum_subsampling,
                             cmap_lims=[25,75])

# %%
# Sensitivity maps - with cortex
for ch_type in ['mag', 'grad', 'eeg']:
    ch_inds = mne.channel_indices_by_type(fwd['info'])
    signal_norms = np.linalg.norm(fwd['sol']['data'][ch_inds[ch_type], :], axis=0)
    plot_cerebellum_data(signal_norms[fwd['src'][0]['nuse']:], fwd['src'], src_whole, cb_data, cort_data=signal_norms[:fwd['src'][0]['nuse']], flatmap_cmap='bwr',
                             mayavi_cmap='OrRd', smoothing_steps=0, view='normal', sub_sampling=cerebellum_subsampling,
                             cmap_lims=[25,75])

