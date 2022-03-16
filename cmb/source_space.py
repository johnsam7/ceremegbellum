#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: November, 2021
# License: MIT
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
import os

from .helpers import *
from .visualization import plot_sagittal


def save_nifti_from_3darray(vol, fname, rotate=False, affine=None):
    if rotate:
        vol = vol[:, ::-1, ::-1]
        vol = np.transpose(vol, axes=(0, 2, 1))
    mgz = nib.Nifti1Image(vol, affine=affine)
    nib.save(mgz, fname)
    print('saved to '+fname)
    return mgz


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
