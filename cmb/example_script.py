import os.path as op
import os
import mne
import pickle
import numpy as np
from mne.datasets import sample
from cmb.functions import get_cerebellum_data
from cmb.source_space import setup_full_source_space
from cmb.visualization import plot_cerebellum_data
data_path = sample.data_path()

# Paths to subject data
cmb_path = '/workspace/ceremegbellum/' #'/vast/fusion/john/ceremegbellum_git/' # path to the folder
sample_dir = op.join(data_path, 'MEG', 'sample',)
raw_fname = op.join(sample_dir, 'sample_audvis_raw.fif')
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
trans = op.join(sample_dir, 'sample_audvis_raw-trans.fif')
fname_cov = sample_dir + '/sample_audvis-cov.fif'
evo_fname = sample_dir + '/sample_audvis-ave.fif'
nnunet_results_path = os.environ['RESULTS_FOLDER']

# Check if the required data are available and download if not
get_cerebellum_data(cmb_path)

# Cerebellar specific
cb_data = pickle.load(open(cmb_path+'data/cerebellum_geo', 'rb'))
spacing = 2 # Use spacing 2 to get an approximately equal grid density in cerebral and cerebellar cortices

# Setup source space
cerebellum_subsampling = 'dense'
src_cort = mne.setup_source_space(subject=subject, subjects_dir=subjects_dir, spacing=spacing, add_dist=False)
src_whole = setup_full_source_space(subject, subjects_dir, cmb_path, cerebellum_subsampling,
                                    plot_cerebellum=False, spacing=spacing)

# Compute forward and inverse operators
conductivity=(0.3, 0.006, 0.3)
mindist = 3.0 # important not to use too large mindist because the cerebellar cortex and inner skull boundary are usually within 5 mm
model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model) # IF too many source space points are lost, the inner skull boundary is too tight and need to be expanded.
info = mne.io.read_info(raw_fname)
fwd = mne.make_forward_solution(info, trans, src_whole, bem=bem, mindist=mindist, eeg=True, n_jobs=1)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)
noise_cov = mne.read_cov(fname_cov)
inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, depth=None, fixed=True)

# Example forward simulation from patch in right lobule VIIIa
label =  cb_data['parcellation']['fine labels'][714]
active_verts = np.where(np.isin(cb_data['dw_data'][cerebellum_subsampling], label.vertices))[0]
active_verts = np.where(np.isin(fwd['src'][1]['vertno'], active_verts))[0]
act_cerb = np.zeros((fwd['src'][1]['nuse']))
act_cerb[active_verts] = 1

# Plot activated patch
plot_cerebellum_data(act_cerb, fwd['src'], src_whole, cb_data, cort_data=np.zeros(fwd['src'][0]['nuse']), flatmap_cmap='bwr', mayavi_cmap='OrRd',
                     smoothing_steps=0, view='all', sub_sampling=cerebellum_subsampling, cmap_lims=[0,100])

evo = mne.read_evokeds(evo_fname)[0]
sens = np.zeros(evo.info['nchan'])
all_chs = mne.pick_types(evo.info, meg=True, eeg=True, exclude=[])
sens[all_chs] = np.sum(fwd['sol']['data'][:, fwd['src'][0]['nuse']+active_verts]*10**-7,axis=1)
evo._data[all_chs] = np.repeat(sens[all_chs].reshape((len(all_chs),1)), repeats=evo._data.shape[1], axis=1)
estimate = mne.minimum_norm.apply_inverse(evo, inverse_operator, 1/9, 'sLORETA', verbose='WARNING')
estimate_cerb = np.linalg.norm(estimate.data[fwd['src'][0]['nuse']:estimate.shape[0], :], axis=1)
cort_data = np.linalg.norm(estimate.data[:fwd['src'][0]['nuse'], :], axis=1)
plot_cerebellum_data(estimate_cerb, fwd['src'], src_whole, cb_data, cort_data=cort_data, flatmap_cmap='bwr',
                         mayavi_cmap='OrRd', smoothing_steps=0, view='all', sub_sampling=cerebellum_subsampling,
                         cmap_lims=[25,75])

