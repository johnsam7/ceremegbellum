"""Provides functions for cerebellar segmentation using nnUNet and ANTs registration."""

import os
import os.path as op
import pickle
import warnings
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from . import helpers
from .helpers import change_labels, save_nifti_from_3darray, set_nnunet_paths

if TYPE_CHECKING:
    # Import is only visible to type checkers, not at runtime.
    from ants.core.ants_image import ANTsImage


def get_segmentation(
    subjects_dir,
    subject,
    cmb_path=None,
    debug_mode=False,
) -> nib.Nifti1Image:
    """Get cerebellar segmentation for a subject.

    Parameters
    ----------
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    subject : str
        Subject identifier.
    cmb_path : str, optional
        Path to the CMB data directory. If None (default), uses the default CMB
        data directory.
    debug_mode : bool, optional
        If True, keeps intermediate files for debugging. If False (default), cleans up
        intermediate files after segmentation.

    Returns
    -------
    nibabel.Nifti1Image
        The cerebellar segmentation as a NIfTI image.
    """
    if cmb_path is None:
        from . import CMB_DATA_DIR

        cmb_path = CMB_DATA_DIR

    set_nnunet_paths(results_folder=op.join(cmb_path, "nnUNet", "RESULTS_FOLDER"))

    # Make directory for segmentation results if it doesn't exist.
    segm_data_dir = op.join(cmb_path, "data", "segm_folder")
    os.makedirs(segm_data_dir, exist_ok=True)

    mri_path = op.join(subjects_dir, subject, "mri", "orig.mgz")
    if not op.exists(mri_path):
        raise FileNotFoundError(f"Could not locate subject MRI at {mri_path}")

    if op.exists(op.join(segm_data_dir, subject + ".nii.gz")):
        print(
            "Previous segmentation found on subject "
            f"{subject}. Returning old segmentation."
        )
        return nib.Nifti1Image.from_filename(
            op.join(segm_data_dir, subject + ".nii.gz")
        )
    else:
        # No previous segmentaion found, make segmentation with trained nnUnet model.
        return _segment_cerebellum(
            subjects_dir, subject, cmb_path, debug_mode, segm_data_dir
        )


def _segment_cerebellum(subjects_dir, subject, cmb_path, debug_mode, segm_data_dir):
    import ants

    # Create temporary directories for intermediate files.
    rel_paths = [
        "tmp",
        "tmp/registered",
        "tmp/registered/whole",
        "tmp/registered/lh",
        "tmp/registered/rh",
        "tmp/registered/mask",
        "tmp/registered/lh_segmented",
        "tmp/registered/rh_segmented",
        "tmp/registered/lob_I_IV",
        "tmp/registered/lob_I_IV_segmented",
        "tmp/registered/mask_divide",
    ]
    for dirs in [op.join(segm_data_dir, rel_path) for rel_path in rel_paths]:
        os.makedirs(dirs, exist_ok=True)

    # Load brain template to get a common space.
    brain_template, brain_template_affine = helpers.load_image_volume(
        op.join(cmb_path, "data", "brain.nii")
    )
    print("Brain template data type is", brain_template.dtype)
    # Help type checkers understand that the affine is not None.
    assert brain_template_affine is not None, (
        "Brain template should have an affine matrix."
    )
    brain_template = brain_template / np.max(brain_template)
    template_ants = ants.from_numpy(brain_template)

    output_folder = op.join(segm_data_dir, "tmp")
    reg_cache_file = op.join(output_folder, "registered", subject + "_reg_cache.pkl")
    reg_whole_img_fname = op.join(
        output_folder, "registered", "whole", subject + "_0000.nii.gz"
    )

    subject_mri_fname = op.join(subjects_dir, subject, "mri", "brain.mgz")
    subject_mri, subject_affine = helpers.load_image_volume(subject_mri_fname)
    print("Subject MRI data type is", subject_mri.dtype)

    # Check if registration was already completed
    if op.exists(reg_cache_file) and op.exists(reg_whole_img_fname):
        print(
            f"Previous registration found for subject {subject}. "
            "Loading cached transforms."
        )
        with open(reg_cache_file, "rb") as f:
            registration = pickle.load(f)
        subj_registered, _ = helpers.load_image_volume(reg_whole_img_fname)
    else:
        # Register and save the result.
        registration, subj_registered = _register_subject_to_template(
            subject_mri,
            template_ants,
            reg_cache_file,
        )
        _ = save_nifti_from_3darray(
            subj_registered,
            reg_whole_img_fname,
            affine=brain_template_affine,
        )

    mask_output_fname = op.join(
        output_folder, "registered", "mask", subject + ".nii.gz"
    )
    if op.exists(mask_output_fname):
        print("Previous mask prediction found. Skipping mask step.")
    else:
        print("Running mask prediction...")
        model_folder = op.join(
            cmb_path,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task001_mask",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder=model_folder,
            input_folder=op.join(output_folder, "registered", "whole"),
            output_folder=op.join(output_folder, "registered", "mask"),
        )

    # Split into LH and RH using ASEG (the label map from FreeSurfer).
    lh_input = op.join(output_folder, "registered", "lh", subject + "_0000.nii.gz")
    rh_input = op.join(output_folder, "registered", "rh", subject + "_0000.nii.gz")
    if op.exists(lh_input) and op.exists(rh_input):
        print("Previous hemisphere split found. Skipping split step.")
    else:
        aseg_registered = _load_and_register_aseg(
            subjects_dir,
            subject,
            template_ants,
            registration,
        )
        # Get the predicted cerebellar mask.
        cerebellum_mask, _ = helpers.load_label_map(
            mask_output_fname,
            dtype=None,  # infer from file on disk
        )
        print("Cerebellum mask data type is", cerebellum_mask.dtype)
        # Split the cerebellum mask into left and right hemispheres using the
        # registered ASEG.
        _split_cerebellar_hemis_aseg(
            aseg_registered,
            subj_registered,
            cerebellum_mask,
            subject,
            op.join(output_folder, "registered"),
            brain_template_affine,
        )

    # Predict LH and RH
    lh_seg_output = op.join(
        output_folder, "registered", "lh_segmented", subject + ".nii.gz"
    )
    rh_seg_output = op.join(
        output_folder, "registered", "rh_segmented", subject + ".nii.gz"
    )
    if op.exists(lh_seg_output):
        print("Previous LH segmentation found. Skipping LH prediction.")
    else:
        print("Running LH prediction...")
        model_folder_lh = op.join(
            cmb_path,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task002_lh",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_lh,
            op.join(output_folder, "registered", "lh"),
            op.join(output_folder, "registered", "lh_segmented"),
        )
    if op.exists(rh_seg_output):
        print("Previous RH segmentation found. Skipping RH prediction.")
    else:
        print("Running RH prediction...")
        model_folder_rh = op.join(
            cmb_path,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task003_rh",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_rh,
            op.join(output_folder, "registered", "rh"),
            op.join(output_folder, "registered", "rh_segmented"),
        )

    # Refine lob I-IV into lobs I-III and IV
    lob_seg_output = op.join(
        output_folder, "registered", "lob_I_IV_segmented", subject + ".nii.gz"
    )
    if op.exists(lob_seg_output):
        print("Previous anterior lobe refinement found. Skipping refinement step.")
    else:
        # Use LH and LH predictions to extract the lob I-IV region and save it.
        lob_I_IV, lob_I_IV_affine = _extract_lob_I_IV(
            lh_input, rh_input, lh_seg_output, rh_seg_output
        )
        save_nifti_from_3darray(
            lob_I_IV,
            op.join(output_folder, "registered", "lob_I_IV", subject + "_0000.nii.gz"),
            affine=lob_I_IV_affine,
        )
        # Run the refinement model.
        print("Running anterior lobe refinement...")
        model_folder_refine = op.join(
            cmb_path,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task004_refine_lobsI_IV",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_refine,
            op.join(output_folder, "registered", "lob_I_IV"),
            op.join(output_folder, "registered", "lob_I_IV_segmented"),
        )
    # Combine the LH, RH, and anterior lobe segmentations into a single segmentation
    # and update the labels.
    seg_complete = _assemble_segmentation(
        lh_seg_fname=lh_seg_output,
        rh_seg_fname=rh_seg_output,
        anterior_seg_fname=lob_seg_output,
    )
    seg_complete_ants = ants.from_numpy(seg_complete)

    # Go back to subject space
    seg_reg = ants.apply_transforms(
        fixed=template_ants,
        moving=seg_complete_ants,
        transformlist=registration["invtransforms"],
        interpolator="genericLabel",
    ).numpy()

    final_seg_output_fname = op.join(segm_data_dir, subject + ".nii.gz")
    save_nifti_from_3darray(seg_reg, final_seg_output_fname, affine=subject_affine)

    if not debug_mode:
        for rel_path in rel_paths:
            cleanup_dir = op.join(segm_data_dir, rel_path)
            if op.exists(cleanup_dir):
                for f in os.listdir(cleanup_dir):
                    if f.endswith((".nii.gz", ".pkl", ".json")):
                        os.remove(op.join(cleanup_dir, f))

    final_seg_nifti = nib.Nifti1Image.from_filename(final_seg_output_fname)
    print("Data type of final segmentation is", final_seg_nifti.get_data_dtype())
    return final_seg_nifti


def _extract_lob_I_IV(
    lh_fname: str, rh_fname: str, lh_seg_fname: str, rh_seg_fname: str
) -> tuple[NDArray[np.float64], NDArray[np.floating]]:
    """Extract pixel intensities for lob I-IV.

    Uses the nnUNet predicted labels to extract the pixel intensities corresponding to
    lob I-IV and returns a new array containing only those intensities.

    Parameters
    ----------
    lh_fname : str
        Path to the left hemisphere image file.
    rh_fname : str
        Path to the right hemisphere image file.
    lh_seg_fname : str
        Path to the left hemisphere segmentation file.
    rh_seg_fname : str
        Path to the right hemisphere segmentation file.

    Returns
    -------
    lobI_IV : NDArray[np.float64]
        A 3D numpy array containing the pixel intensities for lob I-IV.
    affine : NDArray[np.float64]
        The affine transformation matrix associated with the lob I-IV array.
    """
    # Get predicted labels for left hemisphere.
    lh_predictions, lh_affine = helpers.load_label_map(lh_seg_fname, dtype=None)
    print("LH predictions data type is", lh_predictions.dtype)
    # Get the left hemisphere image.
    lh_image, _ = helpers.load_image_volume(lh_fname)
    print("LH image data type is", lh_image.dtype)
    assert lh_predictions.shape == lh_image.shape, (
        "LH predictions and image should have the same shape."
    )
    # Create a new array to hold the lob I-IV region, initialized to zeros.
    lobI_IV = np.zeros(lh_image.shape, dtype=np.float64)

    # Fill in the lob I-IV region based on the predictions.
    # Label 2 corresponds to lob I-IV in the nnUNet predictions.
    lh_mask = lh_predictions == 2
    lobI_IV[lh_mask] = lh_image[lh_mask]

    # Repeat the process for the right hemisphere.

    rh_predictions, rh_affine = helpers.load_label_map(rh_seg_fname)
    rh_image, _ = helpers.load_image_volume(rh_fname)
    print("RH predictions data type is", rh_predictions.dtype)
    print("RH image data type is", rh_image.dtype)
    assert rh_predictions.shape == rh_image.shape, (
        "RH predictions and image should have the same shape."
    )
    rh_mask = rh_predictions == 2
    lobI_IV[rh_mask] = rh_image[rh_mask]

    assert lh_affine is not None and rh_affine is not None, (
        "Both LH and RH predictions should have an affine matrix."
    )
    assert np.allclose(lh_affine, rh_affine), (
        "LH and RH predictions should have the same affine matrix."
    )

    return lobI_IV, lh_affine


def _load_and_register_aseg(
    subjects_dir: str,
    subject: str,
    template_ants: "ANTsImage",
    registration: dict,
) -> np.ndarray:
    """Load the FreeSurfer automatic segmentation and register it to template space."""
    import ants

    aseg, _ = helpers.load_label_map(op.join(subjects_dir, subject, "mri", "aseg.mgz"))
    aseg_ants = ants.from_numpy(aseg)

    # Register segmentation map to the template space.
    aseg_registered = ants.apply_transforms(
        fixed=template_ants,
        moving=aseg_ants,
        transformlist=registration["fwdtransforms"],
        interpolator="genericLabel",
    ).numpy()

    return aseg_registered


def _register_subject_to_template(
    subject_mri: np.ndarray,
    template_ants: "ANTsImage",
    reg_cache_file: str,
) -> tuple[dict, np.ndarray]:
    """Register the subject's MRI to the template space using ANTs registration.

    Calculates the registration transforms and applies them to the subject's MRI to
    align it with the template. The registration results are cached for future use.

    Parameters
    ----------
    subject_mri : np.ndarray
        The subject's MRI image. Does not need to be normalized, as it will be
        normalized within this function.
    template_ants : ANTsImage
        The template image in ANTs format.
    reg_cache_file : str
        Path to the file where the registration results will be cached.

    Returns
    -------
    registration : dict
        The registration results containing the forward and inverse transforms.
    subj_registered : np.ndarray
        The subject's MRI registered to the template space.
    """
    import ants

    # Make sure that normalization is applied (no effect if already normalized).
    subj_brain = subject_mri / np.max(subject_mri)
    subj_brain_ants = ants.from_numpy(subj_brain)

    # Calculate registration.
    print("Registering subject to template space...")
    registration = ants.registration(
        fixed=template_ants, moving=subj_brain_ants, type_of_transform="SyNCC"
    )

    # Save registration cache.
    with open(reg_cache_file, "wb") as f:
        pickle.dump(registration, f)

    # Apply registration.
    subj_registered_ants: ANTsImage = ants.apply_transforms(
        fixed=template_ants,
        moving=subj_brain_ants,
        transformlist=registration["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    subj_registered = subj_registered_ants.numpy()

    return registration, subj_registered


def _split_cerebellar_hemis_aseg(
    aseg: np.ndarray,
    brain: np.ndarray,
    mask: np.ndarray,
    subject: str,
    output_folder: str,
    affine: np.ndarray,
) -> None:
    """Split the whole-cerebellum mask into distinct left and right hemispheres.

    Uses the FreeSurfer `aseg` segmentation map to initialize the left and right
    cerebellar regions and then expands these regions to cover the entire cerebellum
    within the provided `mask`. Saves the resulting left and right hemisphere masks and
    the combined mask as NIfTI files in the specified output folder.

    Parameters
    ----------
    aseg : numpy.ndarray
        The FreeSurfer anatomical segmentation map, warped to the template space.
        Must contain standard FreeSurfer integer labels.
    brain : numpy.ndarray
        The structural MRI volume of the subject, registered to template space.
    mask : numpy.ndarray
        The whole-cerebellum binary or label mask (typically generated by the
        first nnUNet prediction task).
    subject : str
        The subject identifier, used for naming the output files.
    output_folder : str
        The directory path where the resulting split NIfTI files will be saved.
    affine : numpy.ndarray
        The affine transformation matrix to maintain physical spatial
        alignment when saving the NIfTI outputs.
    """
    if any(s > 256 for s in aseg.shape):
        raise ValueError(
            f"FreeSurfer aseg dimensions {aseg.shape} exceed 256. This function "
            f"assumes a maximum of 256 in each dimension."
        )
    if aseg.shape != mask.shape:
        warnings.warn(
            f"Shape mismatch between aseg {aseg.shape} and mask {mask.shape}. "
            "The internal padding logic might be unstable and cause bugs."
        )
    mask_org = mask.copy()
    if not aseg.shape == mask.shape:
        pads = ((np.array(aseg.shape) - np.array(mask.shape)) / 2).astype(int)
        mask_aligned = np.zeros(aseg.shape)
        mask_aligned[
            pads[0] : aseg.shape[0] - pads[0],
            pads[1] : aseg.shape[1] - pads[1],
            pads[2] : aseg.shape[2] - pads[2],
        ] = mask
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
    neighbors = np.array(
        [
            [[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)]
            for z in np.arange(-1, 2)
        ]
    ).reshape(27, 3)

    while len(unsigned_voxels) > 0:
        assigned = np.zeros(len(unsigned_voxels))
        type_vals = []
        for c, vox in enumerate(unsigned_voxels):
            all_neighbors = neighbors + vox
            all_neighbors = all_neighbors[
                np.concatenate(
                    (
                        all_neighbors < np.array(lh_rh_vol.shape),
                        all_neighbors > np.array([-1, -1, -1]),
                    ),
                    axis=1,
                ).all(axis=1)
            ]
            val_neighbors = lh_rh_vol[
                all_neighbors[:, 0], all_neighbors[:, 1], all_neighbors[:, 2]
            ]
            val_neighbors = val_neighbors[~(val_neighbors == 0)]  # Remove background
            if len(val_neighbors) > 0:
                counts = np.bincount(val_neighbors)
                type_val = np.argmax(counts)
                type_vals.append(type_val)
                assigned[c] = 1
        vox_to_assign = unsigned_voxels[np.nonzero(assigned)]
        lh_rh_vol[vox_to_assign[:, 0], vox_to_assign[:, 1], vox_to_assign[:, 2]] = (
            type_vals
        )
        unsigned_voxels = unsigned_voxels[np.where(assigned == 0)]

    final_split = np.zeros(lh_rh_vol.shape)
    final_split[np.nonzero(mask_org)] = lh_rh_vol[np.nonzero(mask_org)]
    lh_split = np.zeros(brain.shape)
    lh_split[np.where(final_split == 1)] = brain[np.where(final_split == 1)]
    rh_split = np.zeros(brain.shape)
    rh_split[np.where(final_split == 2)] = brain[np.where(final_split == 2)]
    mask = np.zeros(brain.shape)  # .astype(int)
    mask[np.where(final_split == 2)] = 2
    mask[np.where(final_split == 1)] = 1

    save_nifti_from_3darray(
        mask,
        op.join(output_folder, "mask_divide", subject + "_mask_lh_rh.nii.gz"),
        affine=affine,
    )
    save_nifti_from_3darray(
        lh_split,
        op.join(output_folder, "lh", subject + "_0000.nii.gz"),
        affine=affine,
    )
    save_nifti_from_3darray(
        rh_split,
        op.join(output_folder, "rh", subject + "_0000.nii.gz"),
        affine=affine,
    )

    return


def _run_nnunet_prediction(model_folder, input_folder, output_folder):
    """Run nnUNet v1 prediction via the Python API.

    Calls predict_from_folder directly and patches torch.load for
    compatibility with PyTorch >= 2.6 (which defaults to weights_only=True).
    """
    import torch

    _orig_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        from nnunet.inference.predict import predict_from_folder

        predict_from_folder(
            model_folder,
            input_folder,
            output_folder,
            folds=None,
            save_npz=False,
            num_threads_preprocessing=6,
            num_threads_nifti_save=2,
            lowres_segmentations=None,
            part_id=0,
            num_parts=1,
            tta=True,
            mixed_precision=True,
            overwrite_existing=True,
            mode="normal",
            step_size=0.5,
            checkpoint_name="model_final_checkpoint",
        )
    finally:
        torch.load = _orig_torch_load


def _assemble_segmentation(
    lh_seg_fname: str, rh_seg_fname: str, anterior_seg_fname: str
) -> np.ndarray:
    """Assemble the final segmentation by combining the individual segmentations.

    Combines the left hemisphere, right hemisphere, and anterior lobe segmentations into
    a single segmentation volume. The labels are adjusted to match the final label map.

    Parameters
    ----------
    lh_seg_fname : str
        Path to the left hemisphere segmentation file.
    rh_seg_fname : str
        Path to the right hemisphere segmentation file.
    anterior_seg_fname : str
        Path to the anterior lobe segmentation file.

    Returns
    -------
    ndarray
        3D numpy array containing the combined segmentation volume with updated labels.
    """
    # Define label mappings.
    old_labels_ant = [1, 2, 3, 4]
    new_labels_ant = [33, 43, 36, 46]
    old_labels_hemi = list(range(1, 17))
    new_labels_lh = [
        12,
        43,
        53,
        63,
        73,
        74,
        75,
        83,
        84,
        93,
        103,
        60,
        70,
        80,
        90,
        100,
    ]
    new_labels_rh = [
        12,
        46,
        56,
        66,
        76,
        77,
        78,
        86,
        87,
        96,
        106,
        60,
        70,
        80,
        90,
        100,
    ]
    # Load individual segmentations and change labels to match the final label map.
    seg_lh, _ = helpers.load_label_map(lh_seg_fname, dtype=np.uint8)
    seg_lh = change_labels(seg_lh, old_labels_hemi, new_labels_lh)

    seg_rh, _ = helpers.load_label_map(rh_seg_fname, dtype=np.uint8)
    seg_rh = change_labels(seg_rh, old_labels_hemi, new_labels_rh)

    seg_ant, _ = helpers.load_label_map(anterior_seg_fname, dtype=np.uint8)
    seg_ant = change_labels(seg_ant, old_labels_ant, new_labels_ant)

    assert seg_lh.shape == seg_rh.shape == seg_ant.shape, (
        "Shape mismatch: Segmentation pieces do not have the same dimensions."
    )
    seg_complete = np.zeros(seg_lh.shape, dtype=np.uint8)

    lh_mask = seg_lh > 0
    seg_complete[lh_mask] = seg_lh[lh_mask]

    rh_mask = seg_rh > 0
    seg_complete[rh_mask] = seg_rh[rh_mask]

    # Paint anterior last to overwrite generic labels with refined sub-lobules.
    ant_mask = seg_ant > 0
    seg_complete[ant_mask] = seg_ant[ant_mask]

    return seg_complete
