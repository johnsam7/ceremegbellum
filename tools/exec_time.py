import os.path as op
import mne
import pickle
import time
from mne.datasets import sample

from cmb import get_cerebellum_data, setup_full_source_space, plot_cerebellum_data
from cmb.segmentation import segment_cerebellum


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


def profile_computation():
    times = list()
    for den in ['dense', 'sparse']:
        times.append((den, time_computation(den)))

    for t in times:
        print(t[0])
        print(t[1])


profile_computation()
