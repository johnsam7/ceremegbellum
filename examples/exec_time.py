import os.path as op
import mne
import pickle
import time
from mne.datasets import sample
import numpy as np

from cmb import get_cerebellum_data, setup_full_source_space, plot_cerebellum_data
from cmb.segmentation import segment_cerebellum
from cmb.visualization import (MLabEmulator, one_pass_cortex_smoothing,
                               one_pass_cerebellum_smoothing, plot_normal,
                               plot_inflated, plot_flatmap)


class Timer:
    def __init__(self):
        self.timing_events = list()
        self.running = False
        self.in_sec = False

    def __str__(self):
        ret_str = ''
        for entry in self.timing_events:
            if entry[1] is not None and entry[2] is not None:
                ret_str += f'{entry[0]}: {entry[2] - entry[1]:.1f}s\n'
            else:
                ret_str += f'{entry[0]: INCOMPLETE}\n'
        return ret_str

    def start(self):
        if not self.running:
            self.timing_events = list()
            self.running = True
            self.timing_events.append(('Total time', time.time(), None))

    def stop(self):
        if self.running:
            self.runnig = False
            self.timing_events[0] = (self.timing_events[0][0],
                                     self.timing_events[0][1],
                                     time.time())

    def start_section(self, section_name):
        if self.running and not self.in_sec:
            self.timing_events.append((section_name, time.time(), None))
            self.in_sec = True

    def stop_section(self):
        if self.running and self.in_sec:
            self.timing_events[-1] = (self.timing_events[-1][0],
                                      self.timing_events[-1][1],
                                      time.time())
            self.in_sec = False


def time_computation(cerebellum_subsampling):
    data_path = sample.data_path()

    # Paths to subject data
    cmb_path = '/autofs/cluster/fusion/gbm6/Projects/cmb/cmb_data/'
    sample_dir = op.join(data_path, 'MEG', 'sample')
    raw_fname = op.join(sample_dir, 'sample_audvis_raw.fif')
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    trans = op.join(sample_dir, 'sample_audvis_raw-trans.fif')
    fname_cov = op.join(sample_dir, 'sample_audvis-cov.fif')
    evo_fname = op.join(sample_dir, 'sample_audvis-ave.fif')

    get_cerebellum_data(cmb_path)

    timer = Timer()
    timer.start()

    cb_data = pickle.load(open(op.join(cmb_path,'data','cerebellum_geo'), 'rb'))
    spacing = 2 # Use spacing 2 to get an approximately equal grid density in cerebral and cerebellar cortices

    timer.start_section('Segmenting Cerebellum')
    segment_cerebellum(subjects_dir, subject, cmb_path, debug_mode=False, force_segmentation=True)
    timer.stop_section()

    # Setup source space using the segmented data

    timer.start_section('Setup source space')
    src_cort = mne.setup_source_space(subject=subject,
                                      subjects_dir=subjects_dir,
                                      spacing=spacing, add_dist=False)
    timer.stop_section()

    timer.start_section('Setup full source space')
    src_whole = setup_full_source_space(subject, subjects_dir,
                                        cmb_path, cerebellum_subsampling,
                                        plot_cerebellum=False, spacing=spacing,
                                        debug_mode=False)
    timer.stop_section()

    # Compute forward and inverse operators
    conductivity = (0.3, 0.006, 0.3)
    mindist = 3.0 # important not to use too large mindist because the cerebellar cortex and inner skull boundary are usually within 5 mm

    timer.start_section('Make bem model')
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    timer.stop_section()

    timer.start_section('Make bem solution')
    bem = mne.make_bem_solution(model)
    timer.stop_section()

    info = mne.io.read_info(raw_fname)

    timer.start_section('Make fwd solution')
    fwd = mne.make_forward_solution(info, trans, src_whole, bem=bem,
                                    mindist=mindist, eeg=True, n_jobs=1)
    timer.stop_section()

    timer.start_section('Convert fwd soluton')
    fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=True, copy=True)
    timer.stop_section()

    noise_cov = mne.read_cov(fname_cov)

    timer.start_section('Make inverse operator')
    inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov,
                                                              depth=None, fixed=True)
    timer.stop_section()

    timer.stop()

    return timer

def time_vizualization():
    save_src_dir="/autofs/cluster/fusion/gbm6/"
    cmb_path="/autofs/cluster/fusion/gbm6/Projects/cmb/cmb_data/"
    cb_data = pickle.load(open(cmb_path+'data/cerebellum_geo', 'rb'))
    subject="Pilot_CB_001"

    src_whole = mne.read_source_spaces(save_src_dir+subject+"_src_whole-src.fif")
    fwd_type = "free"
    fwd = mne.read_forward_solution(save_src_dir+subject+"oneLayer"+"-fwd.fif")
    fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=False, copy=True)

    LOOSE = "2"
    fname = 'deg15_Fix_Left_Ctssssconcatelong'
    method = "dSPM"

    estimate = mne.read_source_estimate(save_src_dir + fname + LOOSE + "estimate_erm"+method+".stc")

    time_want = -0.651
    wh = np.where(estimate.times < time_want)[0][-1]

    estimate_cerb = estimate.data[fwd['src'][0]['nuse']:estimate.shape[0], wh]
    cort_data = estimate.data[:fwd['src'][0]['nuse'], wh]
    cerebellum_subsampling = 'dense'

    # plot_cerebellum_data(estimate_cerb, fwd['src'], src_whole, cb_data,
    #                  cort_data=cort_data, flatmap_cmap='OrRd',
    #                  mayavi_cmap='OrRd', smoothing_steps=0, view='all',
    #                  sub_sampling=cerebellum_subsampling, cmap_lims=[25, 80])

    # using these vars so they match the names in plot_cerebellum_data
    data  =  estimate_cerb
    fwd_src  =  fwd['src']
    org_src  =  src_whole
    cerebellum_geo  =  cb_data
    flatmap_cmap = 'OrRd'
    mayavi_cmap = 'OrRd'
    smoothing_steps = 5
    view = 'all'
    sub_sampling = cerebellum_subsampling
    cmap_lims = [25, 80]

    
    timer = Timer()
    timer.start()
    
    mlab = MLabEmulator()

    if cort_data is not None:
        assert cort_data.shape[0] == fwd_src[0]['nuse'], 'cort_data and src[0][\'nuse\'] must have the same number of elements.'

    timer.start_section("First pass cerebellum smoothing")
    src_cerb = fwd_src[1]
    estimate_smoothed = one_pass_cerebellum_smoothing(
                            data, src_cerb, cerebellum_geo, sub_sampling)
    timer.stop_section()

    timer.start_section("Cortex smoothing")
    src_cort = fwd_src[0]
    if cort_data is not None:
        cort_full_mantle, tris_frame = one_pass_cortex_smoothing(
                                            cort_data, org_src, src_cort)
    timer.stop_section()

    if mayavi_cmap is None:
        if cort_data is None:
            if np.min(estimate_smoothed) < 0:
                mayavi_cmap = 'bwr'
            else:
                mayavi_cmap = 'OrRd'
        else:
            if np.min(np.concatenate((estimate_smoothed, cort_data))) < 0:
                mayavi_cmap = 'bwr'
            else:
                mayavi_cmap = 'OrRd'

    for step in range(smoothing_steps):
        timer.start_section(f"Cerebellum extra smoothing {step}")
        print('Step '+str(step))
        for vert in range(estimate_smoothed.shape[0]):
            estimate_smoothed[vert] = np.nanmean(estimate_smoothed[cerebellum_geo['dw_data'][sub_sampling+'_vert_to_neighbor'][vert]])
        timer.stop_section()

    figures = []

    if view in ['all', 'normal']:
        timer.start_section("Plot Normal")
        figures.append(plot_normal(mlab, src_cerb, cort_data, org_src,
                                   src_cort, estimate_smoothed,
                                   cerebellum_geo, sub_sampling, mayavi_cmap,
                                   tris_frame, cort_full_mantle))
        timer.end_section()

    if view in ['all', 'inflated']:
        timer.start_section("Plot Inflated")
        figures.append(plot_inflated(mlab, estimate_smoothed, cerebellum_geo,
                                     sub_sampling, mayavi_cmap))
        timer.stop_section()

    if view in ['all', 'flatmap']:
        timer.start_section("Plot flatmap")
        figures.append(plot_flatmap(cerebellum_geo, estimate_smoothed,
                                    flatmap_cmap, cmap_lims, sub_sampling))
        timer.stop_section()

    timer.stop()
    return timer


def profile_computation():
    times = list()
    for den in ['dense', 'sparse']:
        times.append((den, time_computation(den)))

    for t in times:
        print(t[0])
        print(t[1])


def profile_vizualization():
    time = time_vizualization()
    print(time)

