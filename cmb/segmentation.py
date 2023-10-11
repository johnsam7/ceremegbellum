import os
import os.path as op
from warnings import warn
import warnings
import subprocess
import numpy as np
import nibabel as nib
import glob

from .helpers import save_nifti_from_3darray, change_labels


def segment_cerebellum(subjects_dir, subject, cmb_path, debug_mode=False, force_segmentation=False):
    """
    Creates segmentation of the cerebellar volume using the nnUnet package. Saves a nifti file to
    the specified subjects directory within the MRI subfolder as cmbseg.nii.gz 

    Parameters
    ----------
    subjects_dir : str
        Path to Freesurfer subject directory 
    subject : str 
        Subject identifier
    cmb_path : str
        Path to the CereMegBellum module
    debug_mode : bool
        Whether to run in debug mode
    fore_segmentation : bool
        Whether to overwrite or skip when existing segmentation is found
    """

    import ants

    print('Doing segmentation...')
    cmb_fname = op.join(subjects_dir, subject, 'mri', 'cmbseg.nii.gz')
    if op.exists(cmb_fname) and not force_segmentation:
        warn(f'Segmentation file {cmb_fname} already exists.'
              ' Skipping segmentation. Please force_segmentation=True if you'
              'want to recompute the segmentation.')
        return

    cmb_path = op.join(cmb_path, '')

    subjects_dir = op.join(subjects_dir, '')

    data_dir = op.join(cmb_path,'data','segm_folder')
    os.makedirs(data_dir,exist_ok=True)

    # Check that all prerequisite programs are ready 
    nnunet_env = os.environ
    nnunet_env['nnUNet_preprocessed'] = op.join(cmb_path,'nnUNet','nnUNet_preprocessed')
    nnunet_env['RESULTS_FOLDER'] = op.join(cmb_path,'nnUNet','RESULTS_FOLDER')
    nnunet_env['nnUNet_raw_data_base'] = op.join(cmb_path,'nnUNet','nnUNet_raw_data_base')

    if not subprocess.run(["mri_convert", "--help"],env=nnunet_env,check=True,stdout = subprocess.DEVNULL):
        warnings.warn('WARNING: mri_convert not found. FreeSurfer has to be compiled for segmentation to work.')
    if not os.path.exists(op.join(subjects_dir,subject,'mri','orig.mgz')):
        raise FileNotFoundError('Could not locate subject MRI at '+subjects_dir+subject+'/mri/orig.mgz')
    if not subprocess.run(["nnUNet_predict", "--help"],env=nnunet_env,check=True,stdout = subprocess.DEVNULL):
        raise OSError('nnUNet_predict not found. Please make sure nnUNet is installed and its environment activated and try again.')
        

    rel_paths = ['whole', 'lh', 'rh', 'mask',
                    'lh_segmented', 'rh_segmented',
                    'lob_I_IV', 'lob_I_IV_segmented', 'mask_divide']
    for dir in rel_paths:
        os.makedirs(op.join(data_dir, 'tmp', 'registered', dir), exist_ok=True)
    
    output_folder = op.join(data_dir,'tmp','')
    
    orig_fname = op.join(subjects_dir,subject,'mri','brain.mgz')
    orig_registered_fname = f'{output_folder}registered/whole/{subject}_0000.nii.gz'

    # Load brain template to get a common space
    brain_template_nib = nib.load(op.join(cmb_path,'data','brain.nii'))
    brain_template = np.asanyarray(brain_template_nib.dataobj)
    brain_template = brain_template/np.max(brain_template)
    template_ants = ants.from_numpy(brain_template)

    # Register
    
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
    save_nifti_from_3darray(subj_reg, orig_registered_fname,
                            affine=brain_template_nib.affine)
    
    # Mask
    subprocess.run(["nnUNet_predict", "-i", f'{output_folder}registered/whole/', "-o", f'{output_folder}registered/mask/', "-tr", "nnUNetTrainerV2", "-ctr", "nnUNetTrainerV2CascadeFullRes", "-m", "3d_fullres", "-p", "nnUNetPlansv2.1", "-t", "001"],env=nnunet_env)

    # Split into LH and RH using ASEG
    aseg = np.asanyarray(nib.load(subjects_dir + subject + '/mri/aseg.mgz').dataobj).astype('uint8')
    aseg = ants.from_numpy(aseg)
    aseg_reg = ants.apply_transforms(fixed=template_ants, moving=aseg, transformlist=reg['fwdtransforms'],
                                        interpolator='genericLabel').numpy()

    mask = np.asanyarray(nib.load(output_folder + 'registered/mask/' + subject + '.nii.gz').dataobj)
    split_cerebellar_hemis_aseg(aseg_reg, subj_reg, mask, subject, output_folder + 'registered/',
                                brain_template_nib.affine)

    # Predict LH and RH
    subprocess.run(["nnUNet_predict", "-i", f'{output_folder}registered/lh/', "-o", f'{output_folder}registered/lh_segmented/', "-tr", "nnUNetTrainerV2", "-ctr", "nnUNetTrainerV2CascadeFullRes", "-m", "3d_fullres", "-p", "nnUNetPlansv2.1", "-t", "002"],env=nnunet_env)
    subprocess.run(["nnUNet_predict", "-i", f'{output_folder}registered/rh/', "-o", f'{output_folder}registered/rh_segmented/', "-tr", "nnUNetTrainerV2", "-ctr", "nnUNetTrainerV2CascadeFullRes", "-m", "3d_fullres", "-p", "nnUNetPlansv2.1", "-t", "003"],env=nnunet_env)
    
    # Refine lob I-IV into lobs I-III and IV
    pred_nib = nib.load(output_folder + 'registered/lh_segmented/' + subject + '.nii.gz')
    vol = np.asanyarray(pred_nib.dataobj)
    image = np.asanyarray(nib.load(output_folder + 'registered/lh/' + subject + '_0000.nii.gz').dataobj)
    lobI_IV = np.zeros(vol.shape)
    lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
    pred_nib = nib.load(output_folder + 'registered/rh_segmented/' + subject + '.nii.gz')
    vol = np.asanyarray(pred_nib.dataobj)
    image = np.asanyarray(nib.load(output_folder + 'registered/rh/' + subject + '_0000.nii.gz').dataobj)
    lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
    save_nifti_from_3darray(lobI_IV, output_folder + 'registered/lob_I_IV/' + subject + '_0000.nii.gz',
                            rotate=False, affine=pred_nib.affine)
    
    subprocess.run(["nnUNet_predict", "-i", f'{output_folder}registered/lob_I_IV/', "-o", f'{output_folder}registered/lob_I_IV_segmented/', "-tr", "nnUNetTrainerV2", "-ctr", "nnUNetTrainerV2CascadeFullRes", "-m", "3d_fullres", "-p", "nnUNetPlansv2.1", "-t", "004"],env=nnunet_env)

            
    # Correct labels
    old_labels_ant = [1, 2, 3, 4]
    new_labels_ant = [33, 43, 36, 46]
    old_labels_hemi = np.arange(1, 17)
    new_labels_lh = [12, 43, 53, 63, 73, 74, 75, 83, 84, 93, 103, 60, 70, 80, 90, 100]
    new_labels_rh = [12, 46, 56, 66, 76, 77, 78, 86, 87, 96, 106, 60, 70, 80, 90, 100]
    
    # Assemble segmentations into one image
    seg = np.asanyarray(nib.load(output_folder + 'registered/lh_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
    seg_lh = change_labels(seg, old_labels_hemi, new_labels_lh)
    seg = np.asanyarray(nib.load(output_folder + 'registered/rh_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
    seg_rh = change_labels(seg, old_labels_hemi, new_labels_rh)
    seg = np.asanyarray(nib.load(output_folder + 'registered/lob_I_IV_segmented/' + subject + '.nii.gz').dataobj).astype('uint8')
    seg_ant = change_labels(seg, old_labels_ant, new_labels_ant)
    seg_complete = np.zeros(seg.shape)
    seg_complete[np.nonzero(seg_lh)] = seg_lh[np.nonzero(seg_lh)]
    seg_complete[np.nonzero(seg_rh)] = seg_rh[np.nonzero(seg_rh)]
    seg_complete[np.nonzero(seg_ant)] = seg_ant[np.nonzero(seg_ant)]
    seg_ants = ants.from_numpy(seg_complete)

    # Go back to subject space
    seg_reg = ants.apply_transforms(fixed=template_ants, moving=seg_ants, transformlist=reg['invtransforms'],
                                        interpolator='genericLabel').numpy()
    
    save_nifti_from_3darray(seg_reg, cmb_fname,
                            rotate=False, affine=subject_mri.affine)

    if not debug_mode:
        tmp_folder = op.join(data_dir,'tmp','registered')
        file_types = ['*/*plans.pkl','*/*postprocessing.json','*/*.nii.gz']
        for file_type in file_types:
            tmp_files = glob.glob(op.join(tmp_folder,file_type))
            for tmp in tmp_files:
                os.remove(tmp)  

                           

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
