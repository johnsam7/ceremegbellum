#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:31:57 2020

@author: ju357
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import nibabel as nib
import pickle
import os
import evaler
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
        os.path.isdir(os.environ['RESULTS_FOLDER'] + '/nnUNet/3d_fullres/Task001_mask_cerebellum') and \
        os.path.isdir(os.environ['RESULTS_FOLDER'] + '/nnUNet/3d_fullres/Task001_mask_cerebellum') and \
        os.path.isdir(os.environ['RESULTS_FOLDER'] + '/nnUNet/3d_fullres/Task001_mask_cerebellum'):
            print('The required atlas data and segmentation models seem to be downloaded.')
    else:
        from pooch import retrieve
        import zipfile
        print('Seems like some data are missing. No problem, fetching...')
        os.system('mkdir ' + cmb_path + 'tmp')
        retrieve(url='https://osf.io/x5ryb/download',
                 known_hash=None, fname='ceremegbellum',
                 path=cmb_path + 'tmp') # UNTIL THE REPO IS PUBLIC, YOU NEED TO DO THIS STEP MANUALLY
        with zipfile.ZipFile(cmb_path + 'tmp/' + 'ceremegbellum.zip', 'r') as zip_ref:
            zip_ref.extractall(cmb_path + 'tmp')
        os.system('mv ' + cmb_path + 'tmp/ceremegbellum/cerebellum_geo ' + cmb_path + \
                  'data/cerebellum_geo')
        os.system('mv ' + cmb_path + 'tmp/ceremegbellum/Task* ' + os.environ['RESULTS_FOLDER'] + \
                  '/nnUNet/3d_fullres/')
        os.system('rm -r ' + cmb_path + 'tmp') # clean up
        print('Done.')
    return

def save_nifti_from_3darray(vol, fname, rotate=False, affine=None):
    if rotate:
        vol = vol[:, ::-1, ::-1]
        vol = np.transpose(vol, axes=(0, 2, 1))
    mgz = nib.Nifti1Image(vol, affine=affine)
    nib.save(mgz, fname)
    print('saved to '+fname)
    return mgz

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


def keep_only_biggest_region(vol, region_removal_limit=0.2, print_progress=False):
    """
    Keeps only the biggest connected region of each unique value in vol. 
    If one smaller region is greater than region_removal_limit*len(biggest_region),
    then do not remove it (set this to >1 if you want to definitely remove all 
    smaller regions). The removed regions are interpolated by typevalue of
    neighbors (spreading). Value 0 is considered background and not examined.
    """
    connected_regions = find_connected_regions(vol, print_progress=False)
    neighbors = np.array([[[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)] for z in np.arange(-1, 2)]).reshape(27,3)
    if print_progress:
        for label in list(connected_regions.keys()):
            print('label '+str(label)+':')
            for k in range(len(connected_regions[label])):
                print(len(connected_regions[label][k]))
    for label in list(connected_regions.keys()):
        biggest_region = np.argmax([len(connected_regions[label][k]) for k in range(len(connected_regions[label]))])
        smaller_regions = list(range(len(connected_regions[label])))
        smaller_regions.remove(biggest_region)
        for smaller_region in smaller_regions:
            if len(connected_regions[label][smaller_region]) > region_removal_limit*len(connected_regions[label][biggest_region]):
                smaller_regions.remove(smaller_region)
                print('Found a separate region for label '+str(label)+' that is > '+str(region_removal_limit*100)+'% of biggest region. Skipping.')
        voxels_in_smaller_regions = np.array([[]]).reshape((0,3))
        for k in smaller_regions:
            voxels_in_smaller_regions = np.concatenate((voxels_in_smaller_regions, connected_regions[label][k]), axis=0).astype(int)
        while len(voxels_in_smaller_regions)>0:
            for vox in voxels_in_smaller_regions:
                all_neighbors = neighbors+vox
                all_neighbors = all_neighbors[np.concatenate((all_neighbors < np.array(vol.shape), all_neighbors > np.array([-1, -1, -1])), axis=1).all(axis=1)]
                val_neighbors = vol[all_neighbors[:, 0], all_neighbors[:, 1], all_neighbors[:, 2]]
                val_neighbors = val_neighbors[~(val_neighbors == vol[vox[0], vox[1], vox[2]])] # Remove label val to make sure it changes label
                if len(val_neighbors) > 0:
                    counts = np.bincount(val_neighbors)
                    type_val = np.argmax(counts)
                    if print_progress:
                        print('Replacing '+str(vox)+', old val: '+str(vol[vox[0], vox[1], vox[2]])+' new val: '+str(type_val))
                    vol[vox[0], vox[1], vox[2]] = type_val
                    voxels_in_smaller_regions = voxels_in_smaller_regions[~np.stack([voxels_in_smaller_regions[:,0]==vox[0],
                                                                                     voxels_in_smaller_regions[:,0]==vox[0], 
                                                                                     voxels_in_smaller_regions[:,0]==vox[0]]).all(axis=0)]
    return vol


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


def split_cerebellar_hemis(subjects_dir, subject, output_folder):
    aseg_nib = nib.load(subjects_dir+subject+'/mri/aseg.mgz')
    aseg = np.asanyarray(aseg_nib.dataobj)
    image = nib.load(subjects_dir+subject+'/mri/orig.mgz')
    image_vol = np.asanyarray(image.dataobj)
    mask_vol = np.asanyarray(nib.load((output_folder+'/mask/output/cerebellum_001.nii.gz')).dataobj)
    pads = ((np.array(aseg.shape)-np.array(mask_vol.shape))/2).astype(int)
    mask_aligned = np.zeros((256, 256, 256))
    mask_aligned[pads[0]:256-pads[0], pads[1]:256-pads[1], pads[2]:256-pads[2]] = mask_vol

    mask = np.array(np.nonzero(mask_aligned)).T
    lh = np.where(np.isin(aseg, [7, 8]))
    rh = np.where(np.isin(aseg, [46, 47]))
    lh_rh_vol = np.zeros(aseg.shape).astype(int)
    lh_rh_vol[lh] = 1
    lh_rh_vol[rh] = 2
    aseg_cerb = np.concatenate((np.array(lh).T, np.array(rh).T), axis=0)
    all_voxels = np.unique(np.concatenate((mask, aseg_cerb), axis=0), axis=0)
    aseg_ints = np.dot(aseg_cerb, np.array([1, 256, 256**2]))
    mask_ints = np.dot(mask, np.array([1, 256, 256**2]))
    unsigned_voxels = mask[~(np.isin(mask_ints, aseg_ints))]
    neighbors = np.array([[[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)] for z in np.arange(-1, 2)]).reshape(27,3)
    
    while len(unsigned_voxels)>0:
        assigned = np.zeros(len(unsigned_voxels))
        type_vals = []
        for c, vox in enumerate(unsigned_voxels):
            all_neighbors = neighbors+vox
            all_neighbors = all_neighbors[np.concatenate((all_neighbors < np.array(lh_rh_vol.shape), all_neighbors > np.array([-1, -1, -1])), axis=1).all(axis=1)]
            val_neighbors = lh_rh_vol[all_neighbors[:, 0], all_neighbors[:, 1], all_neighbors[:, 2]]
            val_neighbors = val_neighbors[~(val_neighbors == 0)] # Remove background
            if len(val_neighbors) > 0:
                counts = np.bincount(val_neighbors)
                type_val = np.argmax(counts)
                type_vals.append(type_val)
                assigned[c] = 1
        vox_to_assign = unsigned_voxels[np.nonzero(assigned)]
        lh_rh_vol[vox_to_assign[:,0], vox_to_assign[:,1], vox_to_assign[:,2]] = type_vals
        unsigned_voxels = unsigned_voxels[np.where(assigned==0)]

    lh_split = np.zeros((256, 256, 256))
    lh_split[np.where(lh_rh_vol == 1)] = image_vol[np.where(lh_rh_vol == 1)]
    rh_split = np.zeros((256, 256, 256))
    rh_split[np.where(lh_rh_vol == 2)] = image_vol[np.where(lh_rh_vol == 2)]
    mask = np.zeros((256, 256, 256)).astype(int)
    mask[np.where(lh_rh_vol == 2)] = 1
    mask[np.where(lh_rh_vol == 1)] = 1

    save_nifti_from_3darray(mask, output_folder+'/../'+subject+'_mask.nii.gz', rotate=False, affine=image.affine)
    save_nifti_from_3darray(lh_split, output_folder+'/lh/cerebellum_001_0000.nii.gz', rotate=False, affine=image.affine)
    save_nifti_from_3darray(rh_split, output_folder+'/rh/cerebellum_001_0000.nii.gz', rotate=False, affine=image.affine)

    return


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

def print_fs_surf(rr, tris, fname, mirror=False):
    """
    Convert to RAS coords and print surface to be plotted with Freeview
    """
    fsVox2RAS = np.array([[-1, 0, 0, 128], [0, 0,  1, -128],
                            [0, -1, 0, 128]]).T
#    fsVox2RAS = np.array([[1, 0, 0, 128], [0, 0,  1, -128],
#                            [0, -1, 0, 128]]).T
#    fsVox2RAS = np.array([[-0.25, 0, 0, 80], [0, 0,  0.25, -110],
#                            [0, -0.25, 0, 110]]).T

    fs_vox = np.hstack((rr, np.ones((len(rr),1))))
    ras = np.dot(fs_vox, fsVox2RAS)
    if mirror:
        ras[:,0] =  -ras[:,0] # Note this is for MNE which mirrors the source space - remove this for alignment with RAS freeview
    nib.freesurfer.io.write_geometry(fname, ras, tris)

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

def plot_sagittal(vol, only_show_midline=False, **kwargs):
    sag_ind = kwargs.get('sag_ind')
    title = kwargs.get('title')
    rr = kwargs.get('rr')
    nn = kwargs.get('nn')
    tris = kwargs.get('tris')
    cmap = kwargs.get('cmap')
    linewidth = kwargs.get('linewidth')
    if type(cmap) == type(None):
        cmap = 'gray_r'
    if type(linewidth) == type(None):
        linewidth = 1.
    fig, ax = plt.subplots(3, 2)
    fig.suptitle(title)

    if type(sag_ind) == type(None):
        x_width = vol.shape[0]
        sag_ind = np.linspace(int(x_width*0.1), int(x_width*0.9), 6).astype(int)

    if only_show_midline:
        sag_ind = [sag_ind[3]]

    for c, slice_ind in enumerate(sag_ind):
        image = vol[slice_ind, :, :]
        plt.subplot(3, 2, c+1)
        plt.imshow(image, cmap=cmap)

        if not type(tris) == type(None):
            z_0 = slice_ind
            cart_ind = 0
            xy = [x for x in range(3) if not x==cart_ind] 
            intersecting_tris = []
            for tri in tris:
                rr_0 = rr[tri[0], :]
                rr_1 = rr[tri[1], :]
                rr_2 = rr[tri[2], :]
                if (np.array([np.sign((rr_0[cart_ind]-z_0)*(rr_1[cart_ind]-z_0)), 
                              np.sign((rr_0[cart_ind]-z_0)*(rr_2[cart_ind]-z_0)), 
                              np.sign((rr_1[cart_ind]-z_0)*(rr_2[cart_ind]-z_0))]) == -1).any():
                    intersecting_tris.append(tri)
            intersecting_tris=np.array(intersecting_tris)
            for int_tri in intersecting_tris:
                rr_0 = rr[int_tri[0], :]
                rr_1 = rr[int_tri[1], :]
                rr_2 = rr[int_tri[2], :]
                t_0 = (z_0 - rr_0[cart_ind])/(rr_1[cart_ind] - rr_0[cart_ind])
                t_1 = (z_0 - rr_0[cart_ind])/(rr_2[cart_ind] - rr_0[cart_ind])
                t_2 = (z_0 - rr_1[cart_ind])/(rr_2[cart_ind] - rr_1[cart_ind])
                xy_points = []
                if t_0 > 0 and t_0 < 1:
                    xy_points.append(t_0*rr_1[xy] + (1-t_0)*rr_0[xy])
                if t_1 > 0 and t_1 < 1:
                    xy_points.append(t_1*rr_2[xy] + (1-t_1)*rr_0[xy])
                if t_2 > 0 and t_2 < 1:
                    xy_points.append(t_2*rr_2[xy] + (1-t_2)*rr_1[xy])
                xy_points = np.array(xy_points)
                plt.plot(xy_points[:,1], xy_points[:,0], color='red', linewidth=linewidth)


        if not type(nn) == type(None):
            ptsp = np.where(np.abs(rr[:,0]-(slice_ind-0.5)) < 1.0)[0]
            x_tp = rr[ptsp,2]
            y_tp = rr[ptsp,1]
#            plt.scatter(x_tp, y_tp, color='r', s=0.1)
            plt.quiver(x_tp, y_tp, nn[ptsp,2], -nn[ptsp,1], scale=1, scale_units='inches')

    return fig, ax

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


def highpass_filter(vols, threshold, plot=False):
    from scipy import signal

    # Create the kernel
    sigma = 2.0     # width of kernel
    x = np.arange(-5,6,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-5,6,1)
    z = np.arange(-5,6,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel[5, 5, 5] = - (np.sum(kernel) - 1)
    kernel = kernel/np.sum(np.abs(kernel))  # Normalize kernel so that homogenous regions stay the same
    
    # apply to sample data
    vols_hp = signal.convolve(vols, kernel, mode="same", method='fft')    
        
    # Threshold and clean
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
    

def setup_cerebellum_source_space(subjects_dir, subject, cerb_dir, cerebellum_subsampling='sparse',
                                  calc_nn=True, print_fs=False, plot=False, mirror=False,
                                  post_process=False, debug_mode=False):
    """Sets up the cerebellar surface source space. Requires cerebellum geometry file
    to be downloaded.
    
    Parameters
    ----------
    subjects_dir : str
        Subjects directory.
    subject : str
        Subject name.
    cerb_dir : str
        Path to cerebellum folder.
    calc_nn: Boolean
        If True, it will calculate the normals of the cerebellum source space.
    print_fs : Boolean
        If True, it will print an fs file of the cerebellar source space that can be viewed with e.g. freeview.
    plot : Boolean
        If True, will plot sagittal cross-sectional plots of the cerebellar source space suposed on subject MR data.
        
    Returns
    -------
    subj_cerb: dictionary
        Dictionary containing geometry data: vertex positions (rr), faces (tris) and normals (nn, if calc_nn is True).
    
    """
    
    from scipy import signal
    import ants
    import pandas as pd
    import evaler

    subjects_dir = subjects_dir + '/'
    print('starting subject '+subject+'...')
    # Load data
    subj_cerb = {}
    data_dir = cerb_dir + 'data/'    
    cb_data = pickle.load(open(data_dir+'cerebellum_geo','rb'))
    if cerebellum_subsampling == 'full':
        rr = cb_data['verts_normal']
        tris = cb_data['faces']
    else:
        rr = cb_data['dw_data'][cerebellum_subsampling+'_verts']
        tris = cb_data['dw_data'][cerebellum_subsampling+'_tris']
        rr = affine_transform(1, np.array([0,0,0]), [np.pi/2, 0, 0], rr)

    hr_vol = cb_data['hr_vol']
    hr_segm = cb_data['parcellation']['volume'].copy()
    old_labels = [12,  33,  36,  43,  46,  53,  56,  60,  63,  66,  70,  73, 74,  75,
                  76,  77,  78,  80,  83,  84,  86,  87,  90,  93,  96, 100, 103, 106]
    hr_segm = change_labels(hr_segm, old_labels=old_labels, new_labels=np.arange(29)[1:])
    
    # Get subject segmentation
    print('Doing segmentation...')
    subj_segm = np.asanyarray(get_segmentation(subjects_dir, subject, data_dir+'segm_folder',
                                               post_process=post_process, debug_mode=debug_mode).dataobj)
    subj_mask = np.asanyarray(nib.load(data_dir+'segm_folder/'+subject+'_mask.nii.gz').dataobj)
    subj = np.asanyarray(nib.load(subjects_dir+subject+'/mri/orig.mgz').dataobj)

    # Mask cerebellum
    pad = 3
    cerb_coords = np.nonzero(subj_mask)
    cb_range = [[np.min(np.nonzero(subj_mask)[x])-pad for x in range(3)],
                 [np.max(np.nonzero(subj_mask)[x])+pad for x in range(3)]]
    subj_segm = subj_segm[cb_range[0][0] : cb_range[1][0],
                              cb_range[0][1] : cb_range[1][1],
                              cb_range[0][2] : cb_range[1][2]]
    subj_contrast = np.zeros(subj.shape)
    subj_contrast[cerb_coords] = subj[cerb_coords]
    subj_contrast = subj_contrast[cb_range[0][0] : cb_range[1][0],
                                  cb_range[0][1] : cb_range[1][1],
                                  cb_range[0][2] : cb_range[1][2]]

    print('Setting up adaptation to subject... ', end='', flush=True)
    hr_vol_scaled = hr_vol
    for axis in range(0,3):
        hr_vol_scaled = signal.resample(hr_vol_scaled, num=subj_segm.shape[axis], axis=axis)
    scf = np.array(hr_vol_scaled.shape) / np.array(hr_vol.shape)
    for x in range(3): rr[:, x] = rr[:, x] * scf[x]
    hr_rs = np.zeros(hr_vol_scaled.shape)
    non_zero_coo_50 = np.array([np.where(hr_vol_scaled > 50)[x] for x in range(3)]).T    
    non_zero_coo = np.array([np.where(hr_vol_scaled > 10)[x] for x in range(3)]).T
    hr_rs[non_zero_coo[:, 0], non_zero_coo[:, 1], non_zero_coo[:, 2]] = \
        hr_vol_scaled[non_zero_coo[:, 0], non_zero_coo[:, 1], non_zero_coo[:, 2]]
    
    # scale labels matrix (by type value vote)
    hr_label_scaled = np.zeros((subj_segm.shape[0], subj_segm.shape[1], subj_segm.shape[2]))
    count_matrix = np.zeros((subj_segm.shape[0], subj_segm.shape[1], subj_segm.shape[2], 100))
    count_matrix[:] = np.nan
    for x in range(hr_segm.shape[0]):
        for y in range(hr_segm.shape[1]):
            for z in range(hr_segm.shape[2]):
                target_vox = (scf*(x,y,z)).astype(int)
                ind = np.min(np.where(np.isnan(count_matrix[target_vox[0], target_vox[1], target_vox[2], :])))
                count_matrix[target_vox[0], target_vox[1], target_vox[2], ind] = hr_segm[x, y, z]
    for x in range(subj_segm.shape[0]):
        for y in range(subj_segm.shape[1]):
            for z in range(subj_segm.shape[2]):
                votes = count_matrix[x, y, z, :]
                votes = votes[~np.isnan(votes)]
                hr_label_scaled[x, y, z] = np.bincount(votes.astype(int)).argmax()
    
    # Correct verts by co-registering lower left posterior and upper right anterior corners
    correction_vector_2 = np.mean([np.min(non_zero_coo_50, axis=0) - np.min(rr, axis=0), 
                                   np.max(non_zero_coo_50, axis=0) - np.max(rr, axis=0)], axis=0)
    rr = rr + correction_vector_2
    print('Done.')

    # Register
    print('Fitting... ', end='', flush=True)
    subj_vec = subj_segm
    hr_vec = hr_label_scaled
    
    print('Fitting labels... ', end='', flush=True)
    subj_label_ants = ants.from_numpy(subj_vec.astype(float))
    hr_label_ants = ants.from_numpy(hr_label_scaled.astype(float))
    reg = ants.registration(fixed=subj_label_ants, moving=hr_label_ants, type_of_transform='SyNCC')
    def_hr_label = ants.apply_transforms(fixed=subj_label_ants, moving=hr_label_ants,
                                     transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    vox_dir = {'x' : list(rr[:, 0]), 'y' : list(rr[:, 1]), 'z' : list(rr[:, 2])}
    pts = pd.DataFrame(data=vox_dir)
    rrw_0 = np.array(ants.apply_transforms_to_points( 3, pts, reg['invtransforms']))
    
    print('Fitting contrast... ')
    subj_contrast = subj_contrast/np.max(subj_contrast)
    hr_rs = hr_rs/np.max(hr_rs)
    subj_ants = ants.from_numpy(subj_contrast)
    hr_rs_ants = ants.from_numpy(hr_rs)
    hr_ants = ants.apply_transforms(fixed=subj_ants, moving=hr_rs_ants,
                                     transformlist=reg['fwdtransforms'])
    reg = ants.registration(fixed=subj_ants, moving=hr_ants, type_of_transform='SyNCC')
    vox_dir = {'x' : list(rrw_0[:, 0]), 'y' : list(rrw_0[:, 1]), 'z' : list(rrw_0[:, 2])}
    pts = pd.DataFrame(data=vox_dir)
    rrw_1 = np.array(ants.apply_transforms_to_points( 3, pts, reg['invtransforms']))
    hr_label_final = ants.apply_transforms(fixed=subj_ants, moving=def_hr_label,
                                     transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
    
    rr_p = rrw_1+cb_range[0]
    subj_cerb.update({'rr' : rr_p})
    subj_cerb.update({'tris' : tris})
    print('Done.')
    
    if calc_nn:
        print('Calculating normals on deformed surface...', end='', flush=True)
        (nn_def, area, area_list, nan_vertices) = evaler.calculate_normals(rr_p, tris, print_info=False)
        subj_cerb.update({'nn' : nn_def})
        subj_cerb.update({'nan_nn' : nan_vertices})
        print('Done.')
    
    # Visualize results as sagittal (x=const) cross-sections
    if plot:
        fig, ax = plot_sagittal(subj, title='Warped points in subj vol', rr=rr_p, tris=tris)
    
    if print_fs:
        print('Saving cerebellar surface as fs files...')
        rr_def = rr_p.copy()
        for x in range(3): rr_def[:, x] = rr_p[:, x]
        print_fs_surf(rr_def, tris, data_dir + subject + '_cerb_cxw.fs', mirror)
        print('Saved to ' + data_dir + subject + '_cerb_cxw.fs')
        
    return subj_cerb


def setup_full_source_space(subject, subjects_dir, cerb_dir, cerb_subsampling='sparse', spacing='oct6',
                            plot_cerebellum=False, debug_mode=False,):
    """Sets up a full surface source space where the first element in the list 
    is the combined cerebral hemishperic source space and the second element
    is the cerebellar source space.
    
    Parameters
    ----------
    subject : str
        Subject name.
    subjects_dir : str
        Subjects directory.
    cerb_dir : str
        Path to cerebellum folder.
    plot_cerebellum : Boolean
        If True, will plot sagittal cross-sectional plots of the cerebellar
        source space superposed on subject MR data.
    spacing : str
        The spacing to use for cortex. Can be ``'ico#'`` for a recursively subdivided
        icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
        or ``'all'`` for all points.
    cerb_subsampling : 'full' | 'sparse' | 'dense'
        The spacing to use for the cerebellum. Can be either full, sparse or dense.

        
    Returns
    -------
    src_whole: list
        List containing two source space elements: the cerebral cortex and the 
        cerebellar cortex.
    
    """
    import mne
#    from evaler import join_source_spaces

    assert cerb_subsampling in ['full', 'sparse', 'dense'], "cerb_subsampling must be either \'full\', \'sparse\' or \'dense\'"
    src_cort = mne.setup_source_space(subject=subject, subjects_dir=subjects_dir, spacing=spacing, add_dist=False)
    if spacing == 'all':
        src_cort[0]['use_tris'] = src_cort[0]['tris']
        src_cort[1]['use_tris'] = src_cort[1]['tris']
    cerb_subj_data = setup_cerebellum_source_space(subjects_dir, subject, cerb_dir, calc_nn=True, cerebellum_subsampling=cerb_subsampling,
                                                   print_fs=True, plot=plot_cerebellum, mirror=False, post_process=True, debug_mode=debug_mode)
    cb_data = pickle.load(open(cerb_dir+'data/cerebellum_geo', 'rb'))
    rr = mne.read_surface(cerb_dir + 'data/' + subject + '_cerb_cxw.fs')[0]/1000
    src_whole = src_cort.copy() 
    hemi_src = join_source_spaces(src_cort)
    src_whole[0] = hemi_src
    src_whole[1]['rr'] = rr
    src_whole[1]['tris'] = cerb_subj_data['tris']
    src_whole[1]['nn'] = cerb_subj_data['nn']
    src_whole[1]['ntri'] = src_whole[1]['tris'].shape[0]
    src_whole[1]['use_tris'] = cerb_subj_data['tris']
    in_use = np.ones(rr.shape[0]).astype(int)
    in_use[cerb_subj_data['nan_nn']] = 0
#    in_use = np.zeros(rr.shape[0])
#    in_use[cb_data['dw_data'][cerb_spacing]] = 1
    src_whole[1]['inuse'] = in_use
    if cerb_subsampling == 'full':
        src_whole[1]['nuse'] = int(np.sum(src_whole[1]['inuse']))
    else:
        src_whole[1]['nuse'] = int(np.sum(src_whole[1]['inuse']))
    src_whole[1]['vertno'] = np.nonzero(src_whole[1]['inuse'])[0]
    src_whole[1]['np'] = src_whole[1]['rr'].shape[0]
    
    return src_whole

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

def get_segmentation(subjects_dir, subject, data_dir, region_removal_limit=0.2,
                     post_process=True, print_progress=False, debug_mode=False):
    import warnings
    import subprocess
    subjects_dir = subjects_dir #+ '/'

    if not os.path.exists(data_dir):
        os.system('mkdir '+data_dir)

    # Check that all prerequisite programs are ready 
    if not os.system('mri_convert --help >/dev/null 2>&1') == 0:
        warnings.warn('mri_convert not found. FreeSurfer has to be compiled for segmentation to work.')
    if not os.path.exists(subjects_dir+subject+'/mri/orig.mgz'):
        raise FileNotFoundError('Could not locate subject MRI at '+subjects_dir+subject+'/mri/orig.mgz')
    if not os.system('nnUNet_predict --help >/dev/null 2>&1') == 0:
        raise OSError('nnUNet_predict not found. Please make sure nnUNet is installed and its environment activated and try again.')
        
    if os.path.exists(data_dir+subject+'.nii.gz') and os.path.exists(data_dir+subject+'_mask.nii.gz'): # check if segmentation exists
        print('Previous segmentation found on subject '+subject+'. Returning old segmentation.')
        return nib.load(data_dir+subject+'.nii.gz') # If yes, return
    else: # If not, make segmentation with trained nnUnet model
        rel_paths = ['/tmp/', '/tmp/lh/', '/tmp/rh/', '/tmp/lh/segmentations/', 
                     '/tmp/rh/segmentations/', '/tmp/lh/segmentations/postprocessed/',
                     '/tmp/rh/segmentations/postprocessed/', '/tmp/mask/', '/tmp/mask/output/']
        for dirs in [data_dir+rel_path for rel_path in rel_paths]:
            if not os.path.exists(dirs):
                os.system('mkdir '+dirs)
        output_folder = data_dir+'/tmp/'
        orig_fname = subjects_dir+subject+'/mri/orig.mgz '
        os.system('cp '+orig_fname+output_folder+'mask/')
        current_ori = str(subprocess.check_output('mri_info --orientation '+orig_fname, shell=True))[-6:-3]
        os.system('mri_convert --in_orientation '+current_ori+' --out_orientation LIA '+
                  output_folder+'mask/orig.mgz '+output_folder+'mask/'+'cerebellum_001_0000.nii.gz')
        # os.system('mri_convert --in_orientation LIA --out_orientation LIA '+output_folder+
        #           '/mask/orig.mgz '+output_folder+'/mask/'+'cerebellum_001_0000.nii.gz')
        os.system('nnUNet_predict -i '+output_folder+'mask/ -o '+output_folder+'mask/output/ -t 1 -m 3d_fullres -tr nnUNetTrainerV2')#' >/dev/null 2>&1')
        split_cerebellar_hemis(subjects_dir, subject, output_folder=output_folder)
        os.system('nnUNet_predict -i '+output_folder+'lh/ -o '+output_folder+'lh/segmentations/ -t 2 -m 3d_fullres -tr nnUNetTrainerV2')# >/dev/null 2>&1')
        os.system('nnUNet_predict -i '+output_folder+'rh/ -o '+output_folder+'rh/segmentations/ -t 3 -m 3d_fullres -tr nnUNetTrainerV2')# >/dev/null 2>&1')
        
        if post_process:
            for input_folder in [output_folder+'rh/segmentations/', output_folder+'lh/segmentations/']:
                postprocessed_folder = input_folder+'postprocessed/'
                nib_in = nib.load(input_folder+'cerebellum_001.nii.gz')
                vol = np.array(nib_in.dataobj).astype('uint8')
                vol = keep_only_biggest_region(vol, region_removal_limit=region_removal_limit, print_progress=print_progress)
                save_nifti_from_3darray(vol, postprocessed_folder+'cerebellum_001.nii.gz', rotate=False, affine=nib_in.affine)
                
        old_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        new_labels_lh = [0, 1, 3, 5, 7, 9, 13, 14, 15, 18, 19, 21, 23, 24, 25, 26, 27, 28]
        new_labels_rh = [0, 1, 2, 4, 6, 8, 10, 11, 12, 16, 17, 20, 22, 24, 25, 26, 27, 28]
#        aseg_nib = nib.load(subjects_dir+subject+'/mri/aseg.mgz')
#        aseg = np.asanyarray(aseg_nib.dataobj)
        
        if post_process:
            rh_seg = np.asanyarray(nib.load(output_folder+'rh/segmentations/postprocessed/cerebellum_001.nii.gz').dataobj)
            lh_seg = np.asanyarray(nib.load(output_folder+'lh/segmentations/postprocessed/cerebellum_001.nii.gz').dataobj)
        else:
            rh_seg = np.asanyarray(nib.load(output_folder+'rh/segmentations/'+subject+'.nii.gz').dataobj)
            lh_seg = np.asanyarray(nib.load(output_folder+'lh/segmentations/'+subject+'.nii.gz').dataobj)
            
        rh_seg = change_labels(vol=rh_seg, old_labels=old_labels, new_labels=new_labels_rh)
        lh_seg = change_labels(vol=lh_seg, old_labels=old_labels, new_labels=new_labels_lh)
#        rh_seg_lia = convert_to_lia_coords(rh_seg, aseg, hemi='rh', crop_pad=crop_pad)
#        lh_seg_lia = convert_to_lia_coords(lh_seg, aseg, hemi='lh', crop_pad=crop_pad)
        seg_lia = np.zeros((256, 256, 256), dtype='uint8')
        seg_lia[np.nonzero(rh_seg)] = rh_seg[np.nonzero(rh_seg)]
        seg_lia[np.nonzero(lh_seg)] = lh_seg[np.nonzero(lh_seg)]
    
        save_nifti_from_3darray(seg_lia, data_dir+subject+'.nii.gz', rotate=False, affine=nib_in.affine)

        if not debug_mode:
            for rel_path in rel_paths:
                os.system('rm '+data_dir+rel_path+'*.nii.gz >/dev/null 2>&1') # Clean up the tmp folder
        return nib.load(data_dir+subject+'.nii.gz')

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
   
def join_source_spaces(src_orig):
    if len(src_orig)!=2:
        raise ValueError('Input must be two source spaces')
        
    src_joined=src_orig.copy()
    src_joined=src_joined[0]
    src_joined['inuse'] = np.concatenate((src_orig[0]['inuse'],src_orig[1]['inuse']))
    src_joined['nn'] = np.concatenate((src_orig[0]['nn'],src_orig[1]['nn']),axis=0)
    src_joined['np'] = src_orig[0]['np'] + src_orig[1]['np']
    src_joined['ntri'] = src_orig[0]['ntri'] + src_orig[1]['ntri']
    src_joined['nuse'] = src_orig[0]['nuse'] + src_orig[1]['nuse']
    src_joined['nuse_tri'] = src_orig[0]['nuse_tri'] + src_orig[1]['nuse_tri']
    src_joined['rr'] = np.concatenate((src_orig[0]['rr'],src_orig[1]['rr']),axis=0)
    src_joined['tris'] = np.concatenate((src_orig[0]['tris'],src_orig[1]['tris']+src_orig[0]['np']),axis=0)
#    src_joined['use_tris'] = np.concatenate((src_orig[0]['use_tris'],src_orig[1]['use_tris']+src_orig[0]['np']),axis=0)
    try:
        src_joined['use_tris'] = np.concatenate((src_orig[0]['use_tris'],src_orig[1]['use_tris']+src_orig[0]['np']),axis=0)
    except:
        Warning('Failed to concatenate use_tris, use_tris will be put to None. This means you will not be able to visualize'+
                ' the cortex in 3d but can still do all computational operations.')
        src_joined['use_tris'] = None
    src_joined['vertno'] = np.nonzero(src_joined['inuse'])[0]

    return src_joined   

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

