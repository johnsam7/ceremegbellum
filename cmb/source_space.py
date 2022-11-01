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


def setup_cerebellum_source_space(subjects_dir, subject, cmb_path, cerebellum_subsampling='sparse',
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
#    import evaler

    subjects_dir = subjects_dir + '/'
    print('starting subject '+subject+'...')
    # Load data
    subj_cerb = {}
    data_dir = cmb_path + 'data/'    
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
    subj_segm = np.asanyarray(get_segmentation(subjects_dir, subject, cmb_path,
                                               post_process=post_process, debug_mode=debug_mode).dataobj)
    subj = np.asanyarray(nib.load(subjects_dir+subject+'/mri/orig.mgz').dataobj)

    # Mask cerebellum
    pad = 3
    cerb_coords = np.nonzero(subj_segm)
    cb_range = [[np.min(np.nonzero(subj_segm)[x])-pad for x in range(3)],
                 [np.max(np.nonzero(subj_segm)[x])+pad for x in range(3)]]
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
                                     transformlist=reg['fwdtransforms'], interpolator='genericLabel')
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
                                     transformlist=reg['fwdtransforms'], interpolator='genericLabel')
    
    rr_p = rrw_1+cb_range[0]
    subj_cerb.update({'rr' : rr_p})
    subj_cerb.update({'tris' : tris})
    print('Done.')
    
    if calc_nn:
        print('Calculating normals on deformed surface...', end='', flush=True)
        (nn_def, area, area_list, nan_vertices) = calculate_normals(rr_p, tris, print_info=False)
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


def calculate_normals(rr, tris, solid_angle_calc=False, obs_point=np.zeros(3), print_info=True):
    """Takes rr - an array of position of vertices and tris - indices of vertices that deliniates
    triangle face and returns vertex normals based on an (unweighted) average of neighboring face normals."""
    A = []
    area_list = []
    area = 0.0
    count=0
    nan_vertices = []
    for x in range(len(rr)):
        A.append([])
    solid_angle = 0
    for row in tris:
        v1=rr[row[1],:]-rr[row[0],:]
        v2=rr[row[2],:]-rr[row[0],:]
        nml = np.cross(v1,v2)
        area = area + np.linalg.norm(nml)/2.0
        area_list.append(area)
        nn_fc = nml/np.linalg.norm(nml)
        A[row[0]].append(nn_fc)
        A[row[1]].append(nn_fc)
        A[row[2]].append(nn_fc)
        if solid_angle_calc==True:
            R1 = rr[row[0]] - obs_point
            R2 = rr[row[1]] - obs_point
            R3 = rr[row[2]] - obs_point
            
            solid_angle = solid_angle + 2*np.arctan(np.dot(R1,np.cross(R2,R3))/ \
                (np.linalg.norm(R1)*np.linalg.norm(R2)*np.linalg.norm(R3) + np.dot(R1,R2)*np.linalg.norm(R3) + \
                np.dot(R1,R3)*np.linalg.norm(R2) + np.dot(R2,R3)*np.linalg.norm(R1)))  
    
    if solid_angle_calc==True:
        print('solid_angle at the point of observation estimated to:')
        print(solid_angle)    
        
    nn = np.zeros((len(rr),3))
    for c, ele in enumerate(A):
        vert_norm = np.zeros(3)
        for vec in ele:
            vert_norm = vert_norm + vec
        vert_norm = vert_norm/np.linalg.norm(vert_norm)
        nn[c,:]=vert_norm
        
    for c, ele in enumerate(A):
        if np.isnan(nn[c,:]).any(): #np.linalg.norm(nn[c,:]) == 0:
            neighbor_rows = np.where(tris==c)[0]
            neighbors = np.unique(tris[neighbor_rows])
            neighbors = neighbors[np.where(neighbors!=c)]
            normal = np.mean(nn[neighbors,:],axis=0)
            nn[c,:] = normal/np.linalg.norm(normal)
            count = count+1
            
            if np.isnan(nn[c,:]).any():
                nan_vertices.append(c)
            
    if print_info:
        print('number of nan normals that have been smoothed = ' + str(count))
        print('Remaining NAN normals = ' + str(len(nan_vertices)))                
        print('Total surface area: ' + np.str(area))
    
    return (nn,area,area_list,nan_vertices)



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


def get_segmentation(subjects_dir, subject, cmb_path, region_removal_limit=0.2,
                     post_process=True, print_progress=False, debug_mode=False):
    import warnings
    import subprocess
    import ants
    
    set_nnunet_paths()

    if not subjects_dir[-1] == '/':
        subjects_dir = subjects_dir +'/'

    data_dir = cmb_path + 'data/segm_folder/'
    if not os.path.exists(data_dir):
        os.system('mkdir '+data_dir)

    # Check that all prerequisite programs are ready 
    if not os.system('mri_convert --help >/dev/null 2>&1') == 0:
        warnings.warn('WARNING: mri_convert not found. FreeSurfer has to be compiled for segmentation to work.')
    if not os.path.exists(subjects_dir+subject+'/mri/orig.mgz'):
        raise FileNotFoundError('Could not locate subject MRI at '+subjects_dir+subject+'/mri/orig.mgz')
    if not os.system('nnUNet_predict --help >/dev/null 2>&1') == 0:
        raise OSError('nnUNet_predict not found. Please make sure nnUNet is installed and its environment activated and try again.')
        
    if os.path.exists(data_dir+subject+'.nii.gz'): # check if segmentation exists
        print('Previous segmentation found on subject '+subject+'. Returning old segmentation.')
        return nib.load(data_dir+subject+'.nii.gz') # If yes, return

    else: # If not, make segmentation with trained nnUnet model
        rel_paths = ['/tmp/', '/tmp/registered/', '/tmp/registered/whole',
                     '/tmp/registered/lh', '/tmp/registered/rh', '/tmp/registered/mask',
                     '/tmp/registered/lh_segmented', '/tmp/registered/rh_segmented',
                     '/tmp/registered/lob_I_IV', '/tmp/registered/lob_I_IV_segmented',
                     '/tmp/registered/mask_divide']
        for dirs in [data_dir+rel_path for rel_path in rel_paths]:
            if not os.path.exists(dirs):
                os.system('mkdir '+dirs)

        # Load brain template to get a common space
        brain_template_nib = nib.load(cmb_path + 'data/brain.nii')
        brain_template = np.asanyarray(brain_template_nib.dataobj)
        brain_template = brain_template/np.max(brain_template)
        template_ants = ants.from_numpy(brain_template)

        # Register
        output_folder = data_dir+'/tmp/'
        orig_fname = subjects_dir+subject+'/mri/brain.mgz'

        subject_mri = nib.load(orig_fname)
        subj_brain = np.asanyarray(subject_mri.dataobj)
        subj_brain = subj_brain/np.max(subj_brain)
        
        # Find registration from subject to common space
        subj_ants = ants.from_numpy(subj_brain)
        
        # Calculate registration 
        reg = ants.registration(fixed=template_ants, moving=subj_ants, type_of_transform='SyNCC')

        # Prepare for masking
        subj_reg_ants = ants.apply_transforms(fixed=template_ants, moving=subj_ants,
                                              transformlist=reg['fwdtransforms'],
                                              interpolator='nearestNeighbor')
        subj_reg = subj_reg_ants.numpy()
        save_nifti_from_3darray(subj_reg, output_folder + 'registered/whole/' + subject + '_0000.nii.gz',
                                affine=brain_template_nib.affine)
        
        # Mask
        os.system('nnUNet_predict -i ' + output_folder + 'registered/whole/ -o ' + output_folder + 
                  'registered/mask/ -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t 001')
        
        # Split into LH and RH using ASEG
        aseg = np.asanyarray(nib.load(subjects_dir + subject + '/mri/aseg.mgz').dataobj).astype('uint8')
        aseg = ants.from_numpy(aseg)
        aseg_reg = ants.apply_transforms(fixed=template_ants, moving=aseg, transformlist=reg['fwdtransforms'],
                                         interpolator='genericLabel').numpy()

        mask = np.asanyarray(nib.load(output_folder + 'registered/mask/' + subject + '.nii.gz').dataobj)
        split_cerebellar_hemis_aseg(aseg_reg, subj_reg, mask, subject, output_folder + 'registered/',
                                    brain_template_nib.affine)
        
        # Predict LH and RH
        os.system('nnUNet_predict -i ' + output_folder + '/registered/lh/ -o ' + output_folder + 
                  '/registered/lh_segmented/ -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t 002')
        os.system('nnUNet_predict -i ' + output_folder + '/registered/rh/ -o ' + output_folder + 
                  '/registered/rh_segmented/ -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t 003')
        
        # Refine lob I-IV into lobs I-III and IV
        pred_nib = nib.load(output_folder + '/registered/lh_segmented/' + subject + '.nii.gz')
        vol = np.asanyarray(pred_nib.dataobj)
        image = np.asanyarray(nib.load(output_folder + '/registered/lh/' + subject + '_0000.nii.gz').dataobj)
        lobI_IV = np.zeros(vol.shape)
        lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
        pred_nib = nib.load(output_folder + '/registered/rh_segmented/' + subject + '.nii.gz')
        vol = np.asanyarray(pred_nib.dataobj)
        image = np.asanyarray(nib.load(output_folder + '/registered/rh/' + subject + '_0000.nii.gz').dataobj)
        lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
        save_nifti_from_3darray(lobI_IV, output_folder + '/registered/lob_I_IV/' + subject + '_0000.nii.gz',
                                rotate=False, affine=pred_nib.affine)
        
        os.system('nnUNet_predict -i ' + output_folder + '/registered/lob_I_IV/ -o ' + output_folder + 
                  '/registered/lob_I_IV_segmented/ -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t 004')
        
        # Correct labels
        old_labels_ant = [1, 2, 3, 4]
        new_labels_ant = [33, 43, 36, 46]
        old_labels_hemi = np.arange(1, 17)
        new_labels_lh = [12, 43, 53, 63, 73, 74, 75, 83, 84, 93, 103, 60, 70, 80, 90, 100]
        new_labels_rh = [12, 46, 56, 66, 76, 77, 78, 86, 87, 96, 106, 60, 70, 80, 90, 100]
        
        # Assemble segmentations into one image
        seg = np.asanyarray(nib.load(output_folder + '/registered/lh_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
        seg_lh = change_labels(seg, old_labels_hemi, new_labels_lh)
        seg = np.asanyarray(nib.load(output_folder + '/registered/rh_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
        seg_rh = change_labels(seg, old_labels_hemi, new_labels_rh)
        seg = np.asanyarray(nib.load(output_folder + '/registered/lob_I_IV_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
        seg_ant = change_labels(seg, old_labels_ant, new_labels_ant)
        seg_complete = np.zeros(seg.shape)
        seg_complete[np.nonzero(seg_lh)] = seg_lh[np.nonzero(seg_lh)]
        seg_complete[np.nonzero(seg_rh)] = seg_rh[np.nonzero(seg_rh)]
        seg_complete[np.nonzero(seg_ant)] = seg_ant[np.nonzero(seg_ant)]
        seg_ants = ants.from_numpy(seg_complete)
    
        # Go back to subject space
        seg_reg = ants.apply_transforms(fixed=template_ants, moving=seg_ants, transformlist=reg['invtransforms'],
                                         interpolator='genericLabel').numpy()
        
        save_nifti_from_3darray(seg_reg, data_dir + subject + '.nii.gz',
                                rotate=False, affine=subject_mri.affine)

        if not debug_mode:
            for rel_path in rel_paths:
                os.system('rm '+data_dir+rel_path+'/*.nii.gz >/dev/null 2>&1') # Clean up the tmp folder
                os.system('rm '+data_dir+rel_path+'/plans.pkl >/dev/null 2>&1') # Clean up the tmp folder
                os.system('rm '+data_dir+rel_path+'/postprocessing.json >/dev/null 2>&1') # Clean up the tmp folder
        return nib.load(data_dir+subject+'.nii.gz')
    
def split_cerebellar_hemis_aseg(aseg, brain, mask, subject, output_folder, affine):
    mask_org = mask.copy()
    if not aseg.shape == mask.shape:
        pads = ((np.array(aseg.shape)-np.array(mask.shape))/2).astype(int)
        mask_aligned = np.zeros(aseg.shape)
        mask_aligned[pads[0]:aseg.shape[0]-pads[0], pads[1]:aseg.shape[1]-pads[1],
                     pads[2]:aseg.shape[3]-pads[2]] = mask
        mask = np.array(np.nonzero(mask_aligned)).T
    else:
        mask = np.array(np.nonzero(mask)).T

    lh = np.where(np.isin(aseg, [7, 8]))
    rh = np.where(np.isin(aseg, [46, 47]))
    lh_rh_vol = np.zeros(aseg.shape).astype(int)
    lh_rh_vol[lh] = 1
    lh_rh_vol[rh] = 2
    aseg_cerb = np.concatenate((np.array(lh).T, np.array(rh).T), axis=0)
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

    final_split = np.zeros(lh_rh_vol.shape)
    final_split[np.nonzero(mask_org)] = lh_rh_vol[np.nonzero(mask_org)]
    lh_split = np.zeros(brain.shape)
    lh_split[np.where(final_split == 1)] = brain[np.where(final_split == 1)]
    rh_split = np.zeros(brain.shape)
    rh_split[np.where(final_split == 2)] = brain[np.where(final_split == 2)]
    mask = np.zeros(brain.shape)#.astype(int)
    mask[np.where(final_split == 2)] = 2
    mask[np.where(final_split == 1)] = 1

    save_nifti_from_3darray(mask, output_folder+'/mask_divide/'+subject+'_mask_lh_rh.nii.gz', rotate=False, affine=affine)
    save_nifti_from_3darray(lh_split, output_folder+'/lh/'+subject+'_0000.nii.gz', rotate=False, affine=affine)
    save_nifti_from_3darray(rh_split, output_folder+'/rh/'+subject+'_0000.nii.gz', rotate=False, affine=affine)
    
    return

    
    
