#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: May, 2020
# License: MIT
# ---------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import nibabel as nib
import pickle
import os
from scipy import signal
from .helpers import *

def get_cerebellum_data(cmb_path):
    """
    Checks if the required cerebellum data are available and download if not.
    
    Parameters
    ----------
    cmb_path : str
        Path to the ceremegbellum folder.
    """
    if os.path.exists(cmb_path + 'data/cerebellum_geo') and \
        os.path.isdir(cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task001_mask') and \
        os.path.isdir(cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task002_lh') and \
        os.path.isdir(cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_rh') and \
        os.path.isdir(cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task004_refine_lobsI_IV') and \
        os.path.exists(cmb_path + 'data/brain.nii')    :
            print('The required atlas data and segmentation models seem to be downloaded.')
    else:
        from pooch import retrieve
        import zipfile
        print('Seems like some data are missing. No problem, fetching...')
        os.system('mkdir ' + cmb_path + 'tmp')
        os.system('mkdir ' + cmb_path + 'data')
        os.system('mkdir ' + cmb_path + 'nnUNet')
        os.system('mkdir ' + cmb_path + 'nnUNet/RESULTS_FOLDER')
        os.system('mkdir ' + cmb_path + 'nnUNet/nnUNet_preprocessed')
        os.system('mkdir ' + cmb_path + 'nnUNet/nnUNet_raw_data_base')
        os.system('mkdir ' + cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet')
        os.system('mkdir ' + cmb_path + 'nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres')
        os.system('cp /autofs/cluster/fusion/Exchange/cerebellum-meeg/ceremegbellum.zip ' + \
                  cmb_path + 'tmp')
#        retrieve(url='https://osf.io/sdn9h/download',
#                 known_hash=None, fname='ceremegbellum',
#                 path=cmb_path + 'tmp') # UNTIL THE REPO IS PUBLIC, YOU NEED TO DO THIS STEP MANUALLY
        with zipfile.ZipFile(cmb_path + 'tmp/' + 'ceremegbellum.zip', 'r') as zip_ref:
            zip_ref.extractall(cmb_path + 'tmp')
        os.system('mv ' + cmb_path + 'tmp/osf_data/cerebellum_geo ' + cmb_path + \
                  'data/cerebellum_geo')
        os.system('mv ' + cmb_path + 'tmp/osf_data/brain.nii ' + cmb_path + \
                  'data/brain.nii')
        os.system('mv ' + cmb_path + 'tmp/osf_data/Task* ' + cmb_path + 'nnUNet/RESULTS_FOLDER' + \
                  '/nnUNet/3d_fullres/')
        os.system('rm -r ' + cmb_path + 'tmp') # clean up
        print('Done.')
    return


def change_labels(vol, old_labels, new_labels):
    new_vol = vol.copy()
    for c, old_label in enumerate(old_labels):
        mask_inds = np.where(vol == old_label)
        new_vol[mask_inds[0], mask_inds[1], mask_inds[2]] = new_labels[c]
    return new_vol

def parse_patch(filename, **kargs):
    import struct
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data


def cartesian2vols(rr, scale, center):
    rr = rr * scale
    rr = rr + (center - np.mean(rr, axis=0))
    return rr

def transform_to_vol_space(rr_high, cx_subj_vols):
    bound_box_vol = (np.max(cx_subj_vols[:, 0]) - np.min(cx_subj_vols[:, 0])) \
                    * (np.max(cx_subj_vols[:, 1]) - np.min(cx_subj_vols[:, 1])) \
                    * (np.max(cx_subj_vols[:, 2]) - np.min(cx_subj_vols[:, 2]))
    bound_box_vol_high = (np.max(rr_high[:, 0]) - np.min(rr_high[:, 0])) \
                         * (np.max(rr_high[:, 1]) - np.min(rr_high[:, 1])) \
                         * (np.max(rr_high[:, 2]) - np.min(rr_high[:, 2]))
    center = np.mean(cx_subj_vols, axis=0)
    rr_0 = cartesian2vols(rr_high, (bound_box_vol / bound_box_vol_high)**(1/3), 
                                   center)
    return rr_0

def contrast_fit(cx_subj_vols, cerb_wm_vols, high_vols_cx, high_vols_wm, weights):
    # Create fake contrasts in rgb, then weight by weights arg in cost function
    subj_vols = np.zeros((256, 256, 256, 3))
    subj_vols[:, :, :, :] = np.array([0, 0, 1])    
    subj_vols[cx_subj_vols[:, 0], cx_subj_vols[:, 1], cx_subj_vols[:, 2], :] = np.array([1, 0, 0])
    subj_vols[cerb_wm_vols[:, 0], cerb_wm_vols[:, 1], cerb_wm_vols[:, 2], :] = np.array([0, 1, 0])

    c_0 = np.array([1.])
    r_0 = np.random.rand(3)
    R_0 = 0.1*np.random.rand(3)
    res_nonlinear = basinhopping(cost_contrast, np.hstack((np.hstack((c_0, r_0)), R_0)), 
                                 minimizer_kwargs = {'args' : (weights,
                                                               high_vols_cx, 
                                                               high_vols_wm, 
                                                               subj_vols)})
    para = res_nonlinear['x']
    rr_fitted = np.rint(solid_body_transform(para[0:3], para[3:6], 
                                                       high_vols_cx)).astype(int)
    return rr_fitted, res_nonlinear

def vol2int(vol, max_vol):
    max_vol = max_vol*2 # Multiply by 2 to handle negative integers
    return max_vol**0+vol[0]+max_vol**1*vol[1]+max_vol**2*vol[2]



def convert_to_lia_coords(vol, aseg, hemi, crop_pad):

    if hemi=='lh':
        origo = np.min(np.array(np.where(np.isin(aseg, [7, 8]))).T, axis=0)
    if hemi=='rh':
        origo = np.min(np.array(np.where(np.isin(aseg, [46, 47]))).T, axis=0)
    
    foreground_coords = np.array(np.nonzero(vol)).T
    lia_coords = foreground_coords + origo - crop_pad

    vol_lia_frame = np.zeros((256, 256, 256))
    vol_lia_frame[lia_coords[:, 0], lia_coords[:, 1], lia_coords[:, 2]] = vol[np.nonzero(vol)]
    
    return vol_lia_frame

def volumetric_segmentation(rr, cx_subj_vols, scale_factor=1):
    # Scale up to make sure no part of the cerebellum is "closed"
    rr_0 = transform_to_vol_space(rr, cx_subj_vols)
    rr_0 = affine_transform(1, np.array([0,0,0]), [np.pi/2, 0, 0], rr_0)
    cortex_vols = np.unique(np.rint(rr_0*scale_factor), axis=0).astype(int)
    max_vol = np.max(cortex_vols)+1
    start_vol = (np.mean(cortex_vols, axis=0)+[20*scale_factor,0,0]).astype(int)
    front_line_vols = start_vol.reshape((1,3))
    saved_vols = start_vol.reshape((1,3))
    black_vols = [vol2int(vol, max_vol) for vol in cortex_vols]
    black_vols.append(vol2int(start_vol, max_vol))
    delta = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], 
                      [0, 0, 1], [0, 0, -1]])
    
    mins = np.min(cortex_vols,axis=0)-1
    maxs = np.max(cortex_vols,axis=0)+1
    X, Y, Z = np.mgrid[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] 
    all_voxels = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    f_vox2int = {}
    for c, vox in enumerate(all_voxels):
        ind = vol2int(vox, max_vol=max_vol)
        f_vox2int.update({tuple(vox) : ind})

    while len(front_line_vols) > 0:
        for vol in front_line_vols:
            neighbors = vol + delta
            for neighbor in neighbors:
#                if not vol2int(neighbor, max_vol) in black_vols:
                if not f_vox2int[tuple(neighbor)] in black_vols:
                    front_line_vols = np.vstack((front_line_vols, neighbor))
                    saved_vols = np.vstack((saved_vols, neighbor))
#                    black_vols.append(vol2int(neighbor, max_vol))
                    black_vols.append(f_vox2int[tuple(neighbor)])
            front_line_vols = front_line_vols[1: front_line_vols.shape[0], :]
        print(len(front_line_vols))
        
    # Scale back down and return mixed voxels and enclosed voxels
    saved_vols_org = np.unique(np.rint(saved_vols/scale_factor).astype(int),axis=0)
    cortex_vols_org = np.unique(np.rint(cortex_vols/scale_factor).astype(int), axis=0)
    enclosed_vols = []
    mixed_vols = []
    cx_ints = []
    for cx_vol in cortex_vols_org:
        cx_ints.append(vol2int(cx_vol, max_vol))
    cx_ints = np.array(cx_ints).astype(int)
    for vol in saved_vols_org:
        if vol2int(vol, max_vol) in cx_ints:
            mixed_vols.append(vol)
        else:
            enclosed_vols.append(vol)
    
    return np.array(enclosed_vols), np.array(mixed_vols), cortex_vols_org

def get_average_normals(nn, rr_vol, plot=False):
    if nn.shape[0] != rr_vol.shape[0]:
        raise Exception('Number of normals must be the same as number of voxels.')
    vols_unique, inds, counts = np.unique(rr_vol, axis=0, return_inverse=True,
                                          return_counts=True)
    mapp = []
    for ind in range(len(vols_unique)):
        mapp.append(np.where(ind == inds)[0])
    
    nn_ave = []
    for c, vol in enumerate(vols_unique):
        nn_ave.append(np.mean(nn[mapp[c], :], axis=0))
    
    if plot:
        norms = np.linalg.norm(np.array(nn_ave), axis=1)
        plt.figure('cumulative')
        y = plt.hist(norms, bins=100, cumulative=True)
        plt.close('cumulative')
        plt.plot(y[1][1:len(y[1])],y[0]/len(norms))
        plt.xlabel('Norm of vector average')
        plt.ylabel('Voxels (% cumulative)')
    
    return np.array(nn_ave)

def print_mgz(mgz, orig_vols, rr_vols, contrasts, data_dir, subject):
    """
    Print mgz giving rr_fitted voxels 0 contrast in the vol data.
    """
    if len(rr_vols) != len(contrasts):
        raise Exception('rr_vols and contrasts must be same length')
    vol_mod = mgz.get_data().copy()
    vol_mod[orig_vols[:, 0], orig_vols[:, 1], orig_vols[:, 2]] = 0
    for c, rr_vol_set in enumerate(rr_vols):
        vol_mod[rr_vol_set[:, 0], rr_vol_set[:, 1], rr_vol_set[:, 2]] = contrasts[c]
    mgz_mod = nib.Nifti1Image(vol_mod, mgz.affine, mgz.header, 
                              mgz.extra, mgz.file_map)

    nib.save(mgz_mod, data_dir + subject + '_tf.mgz')
    print('volume data saved to ' + data_dir + subject + '_tf.mgz')
    return mgz_mod

def mean_cont2tissue_cont(hr_vol, subj_data, cx_subj_vols, wm_subj_vols):
    cx_contrast = np.mean(subj_data[cx_subj_vols[:, 0], cx_subj_vols[:, 1], cx_subj_vols[:, 2]])
    wm_contrast = np.mean(subj_data[wm_subj_vols[:, 0], wm_subj_vols[:, 1], wm_subj_vols[:, 2]])
    tissue_contrast = 2*cx_contrast*hr_vol*np.heaviside(0.5-hr_vol, 0.5) + \
                       (2*(wm_contrast-cx_contrast)*hr_vol + 2*cx_contrast-wm_contrast) \
                       *np.heaviside(hr_vol-0.5, 0.5)
    return tissue_contrast

def hr_int2mean_cont(hr_vol_int):
    unit = 1/np.max(hr_vol_int)
    hr_vol = hr_vol_int*unit    
    return hr_vol

def space_grid(mins, maxs, steps):
    X, Y, Z = np.meshgrid(np.linspace(mins[0], maxs[0], num=steps[0]), 
                          np.linspace(mins[1], maxs[1], num=steps[1]), 
                          np.linspace(mins[2], maxs[2], num=steps[2]))
    return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

def blurring_vol(vol, blurring_steps=3):
    hr_vol_blurred = vol
    voxels_coo = space_grid(mins=(1, 1, 1), maxs=vol.shape-np.array((2, 2, 2)), steps=vol.shape-np.array((2, 2, 2))).astype(int)
    neighbor_mesh = space_grid([-1, -1, -1], [1, 1, 1], steps=[3, 3, 3]).astype(int) 
    for step in range(blurring_steps):
        hr_vol_blurred_static = hr_vol_blurred.copy()
        print('\n \n blurring step ' + str(step) + '\n ')
        for c, vox in enumerate(voxels_coo):
            neighbors = neighbor_mesh + vox
            mean_contrast = np.mean([hr_vol_blurred_static[neighbor[0], neighbor[1], neighbor[2]] for neighbor in neighbors])
            hr_vol_blurred[vox[0], vox[1], vox[2]] = mean_contrast
            print(str(c/len(voxels_coo)*100)[0:3]+' % complete         \r',end='')
    return hr_vol_blurred

def extract_cerb(vol_data, cx_vols, wm_vols, pad=9):
    x = range(np.min(cx_vols[:, 0])-pad, np.max(cx_vols[:, 0])+pad)
    y = range(np.min(cx_vols[:, 1])-pad, np.max(cx_vols[:, 1])+pad)
    z = range(np.min(cx_vols[:, 2])-pad, np.max(cx_vols[:, 2])+pad)
    subj = np.zeros(vol_data.shape)
    subj[cx_vols[:, 0], cx_vols[:, 1], cx_vols[:, 2]] = \
        vol_data[cx_vols[:, 0], cx_vols[:, 1], cx_vols[:, 2]]
    subj[wm_vols[:, 0], wm_vols[:, 1], wm_vols[:, 2]] = \
        vol_data[wm_vols[:, 0], wm_vols[:, 1], wm_vols[:, 2]]
    subj = subj[x, :, :][:, y, :][:, :, z]
    
    return subj

def blurring(hr_rs, plot=False):
    """
    Blurs a volume by convolution with a Gaussian kernel.
    """

    # Create the kernel
    sigma = 3.0     # width of kernel
    x = np.arange(-5,6,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-5,6,1)
    z = np.arange(-5,6,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)  # Normalize kernel so that homogenous regions stay the same
    
    # apply to sample data
    high_res_filtered = signal.convolve(hr_rs, kernel, mode="same")
    high_res_filtered[np.where(hr_rs > 5)] = hr_rs[np.where(hr_rs > 5)]
    high_res_filtered = signal.convolve(high_res_filtered, kernel, mode="same")
    high_res_filtered[np.where(hr_rs > 5)] = hr_rs[np.where(hr_rs > 5)]
    high_res_filtered = signal.convolve(high_res_filtered, kernel, mode="same")
    high_res_filtered[np.where(hr_rs > 5)] = hr_rs[np.where(hr_rs > 5)]
    
    sigma = 1.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)  # Normalize kernel so that homogenous regions stay the same
    
    high_res_filtered = signal.convolve(high_res_filtered, kernel, mode="same", method='fft')
    if plot:
        fig, ax = plot_sagittal(high_res_filtered, title='High-res (blurred)')
        
    return high_res_filtered


def image_filter(vols, filter_type, threshold=None, plot=False):
    from scipy import signal

    # Create the kernel
    sigma = 2.0     # width of kernel
    x = np.arange(-5,6,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-5,6,1)
    z = np.arange(-5,6,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    if filter_type == 'highpass':
        kernel[5, 5, 5] = - (np.sum(kernel) - 1)
    kernel = kernel/np.sum(np.abs(kernel))  # Normalize kernel so that homogenous regions stay the same
    
    # apply to sample data
    vols_hp = signal.convolve(vols, kernel, mode="same", method='fft')    
        
    # Threshold and clean
    if not threshold is None:
        vols_hpt = np.zeros(vols_hp.shape)
        vols_hpt[np.where(vols_hp > threshold)] = 1.0
    
    if plot:
        plot_sagittal(vols_hpt, title='Warped points in subj vol') 
    
    return vols_hpt


def get_convex_hull_2d(points):
    import scipy
    cx_hull = scipy.spatial.ConvexHull(points)
    hull_points = points[cx_hull.vertices, :]
    hull_points = np.concatenate((hull_points, hull_points[0,:].reshape(1,2)), axis=0)
    return hull_points


def calculate_dice(vol, ground_truth):
    vol = vol.astype('float')
    ground_truth = ground_truth.astype('float')
    vol_zero = np.where((vol==0))
    vol[vol_zero[0], vol_zero[1], vol_zero[2]] = np.nan
    vol_zero = np.where((ground_truth==0))
    ground_truth[vol_zero[0], vol_zero[1], vol_zero[2]] = np.nan
    dice = 2 * np.sum(vol == ground_truth)/(np.sum(~np.isnan(vol)) + np.sum(~np.isnan(ground_truth)))
    if np.isnan(dice):
        dice = 0
    return dice

def calculate_dice_ind(vol, ground_truth, lob_val):
    vol_inds = np.where(vol == lob_val)
    vol = np.zeros(vol.shape)
    vol[vol_inds] = lob_val
    ground_truth_inds = np.where(ground_truth == lob_val)
    ground_truth = np.zeros(ground_truth.shape)
    ground_truth[ground_truth_inds] = lob_val
    dice = calculate_dice(vol, ground_truth)
    return dice

def rr_to_labels(rr, cb_data, subj_labels):
    vert_to_vox = np.rint(rr).astype(int)
    recon_vert_labels = subj_labels[vert_to_vox[:, 0], vert_to_vox[:, 1], vert_to_vox[:, 2]]
    neighbor_iterations = 4
    while (recon_vert_labels == 0).any():
        zero_verts = np.where(recon_vert_labels == 0)[0]
        recon_vert_labels_temp = recon_vert_labels.copy()
        for c, vert in enumerate(zero_verts):
            neighbors = cb_data['vert_to_neighbor'][vert]
            all_neighbors = neighbors.copy()
            for k in range(neighbor_iterations):
                for neighbor in all_neighbors:
                    neighbors = np.concatenate((cb_data['vert_to_neighbor'][neighbor], neighbors))
                neighbors = np.unique(neighbors)
                all_neighbors = neighbors
            counts = np.bincount(recon_vert_labels[neighbors])
            type_val = np.argmax(counts)
            if type_val == 0 and len(counts) > 1:
                type_val = np.argmax(counts[1:len(counts)])+1
            recon_vert_labels_temp[vert] = type_val
            print(str(c/len(zero_verts)*100)+' % complete', end='\r', flush=True)
        recon_vert_labels = recon_vert_labels_temp
        print('\n verts left : '+str(len((zero_verts)))+'\n')
    return recon_vert_labels

def compute_segmentation_dice(segmentations, ground_truths):
    coarse_groups = [np.arange(1,29), [2, 3, 4, 5, 6], [1]] #WM, vermis, cerebellum
    lobe_groups = [[7,8,9],[18, 19, 20],[10,11,12,13],[21,22,23,24],[14,15,16],[25,26,27],[17],[28]] #8 lobes
    vermis_groups = [[2], [3], [4], [5], [6]] # vermis
    lobule_groups = [[7], [18], [8], [19], [9], [20], [10], [21], [11], [22], [12], [23], 
                     [13], [24], [14], [25], [15], [26], [16], [27], [17], [28]] #22 hemispheric lobules
    
    vol_dice_d = {}
    hierarchies = ['Coarse Division', 'Lobes', 'Vermis', 'Hemispheric Lobules']
    groups = [['cerebellum', 'vermis', 'CM'],
               ['lh anterior', 'rh anterior', 'lh post. sup.', 'rh post. sup.',
                'lh post. inf.', 'rh post. inf.', 'lh flocculus', 'rh flocculus'],
                ['vermis lob 6', 'vermis lob 7', 'vermis lob 8',
                 'vermis lob 9', 'vermis lob 10'],
                 ['lh lob 1-3', 'rh lob 1-3', 'lh lob 4', 'rh lob 4', 'lh lob 5',
                  'rh lob 5', 'lh lob 6', 'rh lob 6', 'lh lob 7af', 'rh lob 7af',
                  'lh lob 7at', 'rh lob 7at', 'lh lob 7b', 'rh lob 7b', 'lh lob 8a',
                  'rh lob 8a', 'lh lob 8b', 'rh lob 8b', 'lh lob 9', 'rh lob 9',
                  'lh lob 10', 'rh lob 10']]
    comparing_methods_best_worst = [[[0.85, 0.95], [0.67, 0.89], [0.65, 0.89]],
                                    [[0.7, 0.86], [0.7, 0.87], [0.75, 0.9],
                                     [0.73, 0.9], [0.84, 0.9], [0.82, 0.91],
                                     [0.58, 0.73], [0.61, 0.77]],
                                    [[0.58, 0.79], [0.42, 0.78], [0.63, 0.89],
                                     [0.58, 0.86], [0.73, 0.85]],
                                    [[0, 0.75], [0, 0.63], [0.5, 0.78],
                                     [0.51, 0.75], [0.52, 0.65], [0.5, 0.65],
                                     [0.71, 0.84], [0.73, 0.85], [0.73, 0.92],
                                     [0.7, 0.9], [0.62, 0.8], [0.63, 0.85],
                                     [0.47, 0.6], [0.48, 0.7], [0.67, 0.73],
                                     [0.5, 0.7], [0.71, 0.86], [0.68, 0.82],
                                     [0.73, 0.9], [0.73, 0.9], [0.58, 0.74],
                                     [0.61, 0.73]]]
    hierarchy_dice = {}
    for b, hierarchy in enumerate([coarse_groups, lobe_groups, vermis_groups, lobule_groups]):
        group_dice = {}
        for c, group in enumerate(hierarchy):
            dices = []
            for segmentation, ground_truth in zip(segmentations, ground_truths):
                vol_1 = segmentation.copy()
                vol_2 = ground_truth.copy()
                inds_1 = np.where(np.isin(vol_1, group))
                inds_2 = np.where(np.isin(vol_2, group))
                vol_1 = np.zeros(vol_1.shape)
                vol_2 = np.zeros(vol_2.shape)
                vol_1[inds_1] = c+1
                vol_2[inds_2] = c+1
                dices.append(calculate_dice(vol_1, vol_2))
            group_dice.update({groups[b][c] : dices})   
        hierarchy_dice.update({hierarchies[b] : group_dice})    
    
    print('plotting segmentation performance...')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    plt.ylim([0, 1])
    plt.yticks(.1*np.arange(11))
    mean_performances = [[],[],[],[]]
    for c, hierarchy in enumerate(hierarchies):
        performance_data = hierarchy_dice[hierarchy]
        categories = performance_data.keys()
        axs[int(c/2)][np.mod(c,2)].set_xticks(np.arange(len((categories))))
        for d, category in enumerate(list(categories)):
            axs[int(c/2)][np.mod(c,2)].scatter(x=np.repeat(d,repeats=len(performance_data[category])),
               y=performance_data[category])
            mean_performance = np.mean(performance_data[category])
            mean_performances[c].append(mean_performance)
            axs[int(c/2)][np.mod(c,2)].scatter(x=d, y=mean_performance, marker='X', color='k', s=120)
            axs[int(c/2)][np.mod(c,2)].scatter(x=[d, d], y=comparing_methods_best_worst[c][d], marker='x', color='r', s=60)
        axs[int(c/2)][np.mod(c,2)].set_xticklabels(list(categories), rotation=90)
    #    axs[int(c/2)][np.mod(c,2)].set_xlabel(hierarchy)
        axs[int(c/2)][np.mod(c,2)].title.set_text(hierarchy)
        axs[int(c/2)][np.mod(c,2)].grid('on',linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    print('mean performane (best comparing mean performance): \n'+
          'coarse: '+str(np.mean(mean_performances[0]))+' (0.912) \n'+
          'lobe: '+str(np.mean(mean_performances[1]))+' (0.839) \n'+
          'vermis: '+str(np.mean(mean_performances[2]))+' (0.830) \n'+
          'lobule: '+str(np.mean(mean_performances[3]))+' (0.766) \n')

    return hierarchy_dice

def print_parcellation(rr_labels, rr, cb_data, fname, labels, RH_factor=0.75, el_face=None):
    if labels == 'atlas':
        lh_vals = [13, 33, 43, 53, 63, 77, 78, 79, 83, 84, 99, 103]
        rh_vals = [16, 36, 46, 56, 66, 71, 72, 73, 86, 87, 96, 106]
        hemis = lh_vals + rh_vals
        vermis = [6, 7, 8, 9, 10, 12]
    if labels == 'challenge':
        lh_vals = [33, 43, 53, 63, 73, 74, 75, 83, 84, 93, 103]
        rh_vals = [36, 46, 56, 66, 76, 77, 78, 86, 87, 96, 106]
        hemis = lh_vals + rh_vals
        vermis = [60, 70, 80, 90, 100, 12]

    array_list = []
    # Color and prepare vertices
    for c, src_point in enumerate(rr_labels):
        if src_point == 0:
            color = (0., 0., 0., 1.)
        elif src_point in rh_vals:
            ind = np.where(np.isin(rh_vals, src_point))[0][0]
            ind_sc = ind/10
            if ind_sc < 1.:
                color = plt.cm.tab10(ind_sc)
            else:
                if ind_sc == 1.1:
                    color = (0.2, 0.2, 0.2, 1.)
                else:
                    color = (1., 1., 0., 1.)
        elif src_point in lh_vals:
            ind = np.where(np.isin(lh_vals, src_point))[0][0]
            ind_sc = ind/10
            if ind_sc < 1.:
                color = plt.cm.tab10(ind_sc)
            else:
                if ind_sc == 1.1:
                    color = (0.2, 0.2, 0.2, 1.)
                else:
                    color = (1., 1., 0., 1.)
        elif src_point in vermis:
            ind = np.where(np.isin(vermis, src_point))[0][0]
            color = plt.cm.Set3(ind/6)
        if src_point in rh_vals:
            color = [int(255.99*val*RH_factor) for val in color]
        else:
            color = [int(255.99*val) for val in color]
        data_tup = (rr[c, 0], rr[c, 1], rr[c, 2], color[0], color[1], color[2], 255)
        array_list.append(data_tup)
        print(str(c/4573612*100) + ' % complete', end='\r', flush=True)

    # Prepare faces
    if type(el_face) == type(None):
        el_face = prepare_faces(cb_data['faces'])
    else:
        el_face = prepare_faces(el_face)
    vertex = np.array(array_list,dtype=[('x', 'float64'), ('y', 'float64'), ('z', 'float64'),
                                         ('red', 'int32'), ('green', 'int32'), ('blue', 'int32'), ('alpha', 'int32')])
    el_vert = PlyElement.describe(vertex,'vertex')
    PlyData([el_vert, el_face], text=True).write(fname)
    return el_vert, el_face

def prepare_faces(faces):
    face_list = []
    for c,x in enumerate(faces):
        data_tup = ([x[0], x[1], x[2]], 1., 1., 1.)        
        face_list.append(data_tup)
        print(str(c/9163916*100) + ' % complete', end='\r', flush=True)
    
    faces = np.array(face_list, dtype=[('vertex_indices', 'int32', (3,)),
                                       ('red', 'int32'), ('green', 'int32'), ('blue', 'int32')])
    el_face = PlyElement.describe(faces,'face')
    return el_face

def remove_verts_from_surface(rr, tris, points_to_keep):
    rr_cropped = rr[points_to_keep, :] # Remove the points outside the volume
    tris_new = tris[np.isin(tris, points_to_keep).all(axis=1), :]
    old_to_new = np.zeros(len(rr))
    old_to_new[points_to_keep] = np.arange(len(points_to_keep))
    tris_new = old_to_new[tris_new].astype(int)
    return rr_cropped, tris_new

def mask_cerb(subjects_dir, subject, vol, hemi='both', pad=0):
    aseg = np.asanyarray(nib.load(subjects_dir+subject+'/mri/'+'/aseg.mgz').dataobj)
    if hemi == 'both':
        aseg_inds = [7, 8, 46, 47]
    elif hemi == 'lh':
        aseg_inds = [7, 8]
    elif hemi == 'rh':
        aseg_inds = [46, 47]
    cerb_coords = np.array(np.where(np.isin(aseg, aseg_inds))).T
    cerb_mask = np.zeros(vol.shape)
    cerb_mask[cerb_coords[:, 0], cerb_coords[:, 1], cerb_coords[:, 2]] = \
            vol[cerb_coords[:, 0], cerb_coords[:, 1], cerb_coords[:, 2]]
    cb_range = np.concatenate((np.min(cerb_coords, axis=0), np.max(cerb_coords, axis=0)))
    cerb_mask = cerb_mask[cb_range[0]-pad:cb_range[3]+pad, :, :][:, cb_range[1]-pad:cb_range[4]+pad, :][:, :, cb_range[2]-pad:cb_range[5]+pad]
    return cerb_mask

def print_cerebellum(subjects_dir, subject, fname, hemi='both', pad=0, convert_to_coords = False, crop=False):
    orig_nib = nib.load(subjects_dir+subject+'/mri/'+'/orig.mgz')
    orig = np.asanyarray(orig_nib.dataobj)
    cerebellum = mask_cerb(subjects_dir, subject, orig, hemi=hemi, pad=pad)
    save_nifti_from_3darray(cerebellum, fname+'.nii.gz', rotate=False, affine=orig_nib.affine)
    if type(convert_to_coords) == str:
        os.system('mri_convert --out_orientation '+convert_to_coords+' '+fname+'.nii.gz '+fname+'.nii.gz')
    return cerebellum


def create_label_verts(labels, fwd):
    label_verts = {}
    num = 0
    for label in labels:
        if label.hemi == 'lh':
            hemi_ind = 0
            vert_offset = 0
        if label.hemi == 'rh':
            hemi_ind = 1     
            vert_offset = fwd['src'][0]['nuse']
        verts_lab = label.vertices
        verts_in_src_space = verts_lab[np.isin(verts_lab,fwd['src'][hemi_ind]['vertno'])]
        inds = np.where(np.in1d(fwd['src'][hemi_ind]['vertno'],verts_in_src_space))[0]+vert_offset
        if len(inds) == 0:
            warnings.warn(label.name + ' label has no active source.')
            num = num+1
        label_verts.update({label.name : inds})    
    return label_verts

