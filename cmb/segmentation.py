"""Provides pipeline for segmenting the cerebellum."""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
#          Teemu Taivainen
# Created: September, 2021 (Modified: July, 2026)
# License: MIT
# ---------------------------------------------------------------------------

import logging
import os
import os.path as op
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mne
import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from numpy.typing import NDArray

from .helpers import (
    change_labels,
    convert_to_ants_image,
    load_image_volume,
    load_label_map,
    save_nifti_from_3darray,
    set_nnunet_paths,
)

if TYPE_CHECKING:
    # Import is only visible to type checkers, not at runtime.
    from ants.core.ants_image import ANTsImage

logger = logging.getLogger(__name__)


def get_segmentation(
    subject: str,
    subjects_dir: os.PathLike[str] | str | None = None,
    cmb_dir: os.PathLike[str] | str | None = None,
    segmentation_fname: os.PathLike[str] | str | None = None,
    save_segmentation: bool = True,
    overwrite: bool = False,
    intermediate_caching: bool = False,
) -> Nifti1Image:
    """Compute or load the cerebellar segmentation for a subject.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    subjects_dir: path-like | None, optional
        The path to the directory containing the FreeSurfer subjects reconstructions.
        If None, defaults to the SUBJECTS_DIR environment variable.
    cmb_dir: path-like | None, optional
        Path to the CMB data directory. If None (default), uses the default CMB
        data directory.
    segmentation_fname : path-like | None, optional
        The path to load/save the segmentation. If None (default), uses the default path
        ``<subjects_dir>/<subject>/mri/cerebellum_segmentation.nii.gz``.
    save_segmentation : bool, optional
        If True (default), saves a newly computed segmentation to disk.
        If False, newly computed segmentations are returned without saving.
        Existing saved segmentations are still loaded unless ``overwrite=True``.
    overwrite : bool, optional
        If True, always computes the segmentation and overwrites any existing file.
        If False (default), returns the existing segmentation if it exists.
    intermediate_caching : bool, optional
        If True, reuses existing intermediate registration and nnU-Net predictions from
        ``<cmb_dir>/data/segm_folder/tmp`` and keeps newly generated intermediates
        there. If False (default), uses a temporary work directory for this run
        without touching any existing intermediate cache.

    Returns
    -------
    nibabel.Nifti1Image
        The cerebellar segmentation as a NIfTI image object.
    """
    # Handle path inputs and defaults.
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    assert subjects_dir is not None, (
        "subjects_dir returned by mne.utils.get_subjects_dir should not be None."
    )
    subjects_dir = Path(subjects_dir)  # ensure the type is Path
    if cmb_dir is None:
        from cmb import CMB_DATA_DIR

        cmb_dir = Path(CMB_DATA_DIR)
    else:
        cmb_dir = Path(cmb_dir)
    if segmentation_fname is None:
        segmentation_file = (
            subjects_dir / subject / "mri" / "cerebellum_segmentation.nii.gz"
        )
    else:
        segmentation_file = Path(segmentation_fname)

    if not overwrite and segmentation_file.exists():
        logger.info(
            "Previous segmentation found on subject %s. Returning old segmentation.",
            subject,
        )
        return nib.Nifti1Image.from_filename(segmentation_file)

    # Make segmentation with trained nnUnet model.

    if intermediate_caching:
        intermediate_files_dir = cmb_dir / "data" / "segm_folder" / "tmp"
        intermediate_files_dir.mkdir(parents=True, exist_ok=True)
        segmentation = _segment_cerebellum(
            subjects_dir, subject, cmb_dir, intermediate_files_dir
        )
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            intermediate_files_dir = Path(tmp_dir)
            segmentation = _segment_cerebellum(
                subjects_dir, subject, cmb_dir, intermediate_files_dir
            )
    if save_segmentation:
        segmentation_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving segmentation to %s", segmentation_file)
        nib.save(segmentation, segmentation_file)

    return segmentation


def _segment_cerebellum(
    subjects_dir: Path,
    subject: str,
    cmb_dir: Path,
    intermediate_files_dir: Path,
) -> Nifti1Image:
    """Run the cerebellar segmentation pipeline for a subject.

    Parameters
    ----------
    subjects_dir : Path
        Path to the FreeSurfer subjects directory.
    subject : str
        The FreeSurfer subject name.
    cmb_dir : Path
        Path to the CMB data directory.
    intermediate_files_dir : Path
        Path to the directory where intermediate files (registration and nnU-Net
        predictions) will be stored.

    Returns
    -------
    nibabel.Nifti1Image
        The cerebellar segmentation as a NIfTI image object.
    """
    import ants
    from ants.registration import apply_transforms

    set_nnunet_paths(results_folder=op.join(cmb_dir, "nnUNet", "RESULTS_FOLDER"))

    # Create temporary directories for intermediate files.
    rel_paths = [
        "registered",
        "registered/whole",
        "registered/lh",
        "registered/rh",
        "registered/mask",
        "registered/lh_segmented",
        "registered/rh_segmented",
        "registered/lob_I_IV",
        "registered/lob_I_IV_segmented",
        "registered/mask_divide",
    ]
    for dirs in [op.join(intermediate_files_dir, rel_path) for rel_path in rel_paths]:
        os.makedirs(dirs, exist_ok=True)

    # Load brain template to get a common space.
    brain_template, brain_template_affine = load_image_volume(
        op.join(cmb_dir, "data", "brain.nii")
    )
    # Help type checkers understand that the affine is not None.
    assert brain_template_affine is not None, (
        "Brain template should have an affine matrix."
    )
    template_ants = convert_to_ants_image(brain_template, normalize=True)

    # Load subject MRI.
    subject_mri, subject_affine = load_image_volume(
        op.join(subjects_dir, subject, "mri", "brain.mgz")
    )
    if subject_affine is None:
        warnings.warn(
            "Subject MRI does not have an affine matrix.", UserWarning, stacklevel=2
        )
    subject_brain_ants = convert_to_ants_image(subject_mri, normalize=True)

    reg_output_folder = op.join(intermediate_files_dir, "registered")
    reg_forward_fname = op.join(reg_output_folder, subject + "_reg_Composite.h5")
    reg_inverse_fname = op.join(reg_output_folder, subject + "_reg_InverseComposite.h5")
    reg_whole_img_fname = op.join(reg_output_folder, "whole", subject + "_0000.nii.gz")
    reg_files = [reg_forward_fname, reg_inverse_fname, reg_whole_img_fname]

    # REGISTRATION TO TEMPLATE SPACE

    # Check if registration was already completed.
    if all([op.exists(file) for file in reg_files]):
        logger.info(
            "Previous registration found for subject %s. Using cached transforms.",
            subject,
        )
        # Mimic the registration dictionary structure returned by ants.registration.
        registration = {
            "fwdtransforms": [reg_forward_fname],
            "invtransforms": [reg_inverse_fname],
        }
        subj_registered, _ = load_image_volume(reg_whole_img_fname)
    else:
        # Register and save the result.
        transform_prefix = reg_forward_fname.replace("Composite.h5", "")
        registration, subj_registered = _register_subject_to_template(
            subject_brain_ants,
            template_ants,
            transform_fname_prefix=transform_prefix,
        )
        _ = save_nifti_from_3darray(
            subj_registered,
            reg_whole_img_fname,
            affine=brain_template_affine,
        )

    # PREDICTION OF CEREBELLAR MASK

    mask_output_fname = op.join(
        intermediate_files_dir, "registered", "mask", subject + ".nii.gz"
    )
    if op.exists(mask_output_fname):
        logger.info(
            "Previous mask prediction found for subject %s. Skipping mask step.",
            subject,
        )
    else:
        logger.info("Running mask prediction for subject %s.", subject)
        model_folder = op.join(
            cmb_dir,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task001_mask",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder=model_folder,
            input_folder=op.join(intermediate_files_dir, "registered", "whole"),
            output_folder=op.join(intermediate_files_dir, "registered", "mask"),
        )

    # SPLITTING THE MASK INTO LEFT AND RIGHT HEMISPHERES

    # Split into LH and RH using ASEG (the label map from FreeSurfer).
    lh_input = op.join(
        intermediate_files_dir, "registered", "lh", subject + "_0000.nii.gz"
    )
    rh_input = op.join(
        intermediate_files_dir, "registered", "rh", subject + "_0000.nii.gz"
    )
    if op.exists(lh_input) and op.exists(rh_input):
        logger.info(
            "Previous hemisphere split found for subject %s. Skipping split step.",
            subject,
        )
    else:
        aseg_registered = _load_and_register_aseg(
            subjects_dir,
            subject,
            template_ants,
            registration,
        )
        # Get the predicted cerebellar mask.
        cerebellum_mask, _ = load_label_map(
            mask_output_fname,
            dtype=None,  # infer from file on disk
        )
        # Split the cerebellum mask into left and right hemispheres using the
        # registered ASEG.
        _split_cerebellar_hemis_aseg(
            aseg_registered,
            subj_registered,
            cerebellum_mask,
            subject,
            op.join(intermediate_files_dir, "registered"),
            brain_template_affine,
        )

    # PREDICTION OF LEFT AND RIGHT HEMISPHERES

    lh_seg_output = op.join(
        intermediate_files_dir, "registered", "lh_segmented", subject + ".nii.gz"
    )
    rh_seg_output = op.join(
        intermediate_files_dir, "registered", "rh_segmented", subject + ".nii.gz"
    )
    if op.exists(lh_seg_output):
        logger.info(
            "Previous LH segmentation found for subject %s. Skipping LH prediction.",
            subject,
        )
    else:
        logger.info("Running LH prediction for subject %s.", subject)
        model_folder_lh = op.join(
            cmb_dir,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task002_lh",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_lh,
            op.join(intermediate_files_dir, "registered", "lh"),
            op.join(intermediate_files_dir, "registered", "lh_segmented"),
        )
    if op.exists(rh_seg_output):
        logger.info(
            "Previous RH segmentation found for subject %s. Skipping RH prediction.",
            subject,
        )
    else:
        logger.info("Running RH prediction for subject %s.", subject)
        model_folder_rh = op.join(
            cmb_dir,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task003_rh",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_rh,
            op.join(intermediate_files_dir, "registered", "rh"),
            op.join(intermediate_files_dir, "registered", "rh_segmented"),
        )

    # ANTERIOR LOBE PREDICTION

    # Refine lob I-IV into lobs I-III and IV
    lob_seg_output = op.join(
        intermediate_files_dir, "registered", "lob_I_IV_segmented", subject + ".nii.gz"
    )
    if op.exists(lob_seg_output):
        logger.info(
            "Previous anterior lobe refinement found for subject %s. "
            "Skipping refinement step.",
            subject,
        )
    else:
        # Use LH and LH predictions to extract the lob I-IV region and save it.
        lob_I_IV, lob_I_IV_affine = _extract_lob_I_IV(
            lh_input, rh_input, lh_seg_output, rh_seg_output
        )
        save_nifti_from_3darray(
            lob_I_IV,
            op.join(
                intermediate_files_dir,
                "registered",
                "lob_I_IV",
                subject + "_0000.nii.gz",
            ),
            affine=lob_I_IV_affine,
        )
        # Run the refinement model.
        logger.info("Running anterior lobe refinement for subject %s.", subject)
        model_folder_refine = op.join(
            cmb_dir,
            "nnUNet",
            "RESULTS_FOLDER",
            "nnUNet",
            "3d_fullres",
            "Task004_refine_lobsI_IV",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        _run_nnunet_prediction(
            model_folder_refine,
            op.join(intermediate_files_dir, "registered", "lob_I_IV"),
            op.join(intermediate_files_dir, "registered", "lob_I_IV_segmented"),
        )
    # COMBINING THE SEGMENTATIONS

    # Combine the LH, RH, and anterior lobe segmentations into a single segmentation
    # and update the labels.
    seg_complete = _assemble_segmentation(
        lh_seg_fname=lh_seg_output,
        rh_seg_fname=rh_seg_output,
        anterior_seg_fname=lob_seg_output,
    )
    seg_complete_ants = ants.from_numpy(seg_complete)

    # Go back to subject space
    # Use cast to help type checkers understand that the result is an ANTsImage.
    seg_reg_ants = cast(
        "ANTsImage",
        apply_transforms(
            fixed=template_ants,
            moving=seg_complete_ants,
            transformlist=registration["invtransforms"],
            interpolator="genericLabel",
        ),
    )
    # numpy() returns a float32 array, convert it back to uint8.
    seg_reg = seg_reg_ants.numpy().round().astype(np.uint8)

    return nib.Nifti1Image(seg_reg, subject_affine)


def _extract_lob_I_IV(
    lh_fname: str, rh_fname: str, lh_seg_fname: str, rh_seg_fname: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    lh_predictions, lh_affine = load_label_map(lh_seg_fname, dtype=None)
    # Get the left hemisphere image.
    lh_image, _ = load_image_volume(lh_fname)
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

    rh_predictions, rh_affine = load_label_map(rh_seg_fname, dtype=None)
    rh_image, _ = load_image_volume(rh_fname)
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
    subjects_dir: Path,
    subject: str,
    template_ants: "ANTsImage",
    registration: dict,
) -> np.ndarray:
    """Load the FreeSurfer automatic segmentation and register it to template space.

    Parameters
    ----------
    subjects_dir : Path
        Path to the FreeSurfer subjects directory.
    subject : str
        The FreeSurfer subject name.
    template_ants : ANTsImage
        The template image in ANTs format.
    registration : dict
        The registration from which to apply the forward transforms to the segmentation.

    Returns
    -------
    numpy.ndarray
        The registered FreeSurfer automatic segmentation in template space.
        Data type is preserved from the original aseg.mgz file.
    """
    import ants
    from ants.registration import apply_transforms

    aseg, _ = load_label_map(op.join(subjects_dir, subject, "mri", "aseg.mgz"))
    aseg_ants = ants.from_numpy(aseg)

    # Register segmentation map to the template space.
    aseg_registered_float: NDArray[np.float32] = cast(
        "ANTsImage",
        apply_transforms(
            fixed=template_ants,
            moving=aseg_ants,
            transformlist=registration["fwdtransforms"],
            interpolator="genericLabel",
        ),
    ).numpy()
    # Convert back to original data type of aseg.
    aseg_registered = aseg_registered_float.round().astype(aseg.dtype)

    return aseg_registered


def _register_subject_to_template(
    subj_brain_ants: "ANTsImage",
    template_ants: "ANTsImage",
    transform_fname_prefix: str,
) -> tuple[dict, NDArray[np.float32]]:
    """Register the subject's MRI to the template space using ANTs registration.

    Calculates the registration transforms and applies them to the subject's MRI to
    align it with the template. The registration results are cached for future use.

    Parameters
    ----------
    subj_brain_ants : ANTsImage
        The subject's brain MRI in ANTs format.
    template_ants : ANTsImage
        The template image in ANTs format.
    transform_fname_prefix : str
        The prefix for the filenames where the registration transforms will be saved.
        ANTs saves forward transform to {transform_fname_prefix}Composite.h5 and inverse
        transform to {transform_fname_prefix}InverseComposite.h5.

    Returns
    -------
    registration : dict
        The registration results containing the forward and inverse transforms.
    subj_registered : NDArray[np.float32]
        The subject's MRI registered to the template space.
    """
    from ants.registration import apply_transforms, registration

    # Calculate registration.
    logger.info("Registering subject to template space...")
    registration = registration(
        fixed=template_ants,
        moving=subj_brain_ants,
        type_of_transform="SyNCC",
        outprefix=transform_fname_prefix,  # save transforms
        write_composite_transform=True,  # combine warp and affine into a single file
    )

    # Apply registration.
    # NOTE: Downcasts to float32.
    subj_registered_ants = cast(
        "ANTsImage",
        apply_transforms(
            fixed=template_ants,
            moving=subj_brain_ants,
            transformlist=registration["fwdtransforms"],
            interpolator="nearestNeighbor",
        ),
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
        raise ValueError(
            f"Shape mismatch: aseg shape {aseg.shape} does not match mask shape "
            f"{mask.shape}. Both images must be registered to the template space."
        )
    mask_org = mask.copy()
    mask = np.array(np.nonzero(mask)).T

    lh = np.where(np.isin(aseg, [7, 8]))
    rh = np.where(np.isin(aseg, [46, 47]))
    lh_rh_vol = np.zeros(aseg.shape, dtype=np.uint8)
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
        starting_voxel_count = len(unsigned_voxels)
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
        if len(unsigned_voxels) == starting_voxel_count:
            warnings.warn(
                f"During the splitting of the cerebellar mask into left and right "
                f"hemispheres, {len(unsigned_voxels)} voxels could not be assigned to "
                "either hemisphere. Leaving them as background (label 0).",
                RuntimeWarning,
                stacklevel=2,
            )
            break

    final_split = np.zeros(lh_rh_vol.shape, dtype=np.uint8)
    final_split[np.nonzero(mask_org)] = lh_rh_vol[np.nonzero(mask_org)]
    lh_split = np.zeros(brain.shape)
    lh_split[np.where(final_split == 1)] = brain[np.where(final_split == 1)]
    rh_split = np.zeros(brain.shape)
    rh_split[np.where(final_split == 2)] = brain[np.where(final_split == 2)]
    mask = np.zeros(brain.shape, dtype=np.uint8)
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


def _run_nnunet_prediction(
    model_folder: str, input_folder: str, output_folder: str
) -> None:
    """Run nnUNet v1 prediction via the Python API.

    Calls predict_from_folder directly and patches torch.load for
    compatibility with PyTorch >= 2.6 (which defaults to weights_only=True).

    Parameters
    ----------
    model_folder : str
        Path to the nnUNet model folder containing the trained model.
    input_folder : str
        Path to the folder containing input images for prediction.
    output_folder : str
        Path to the folder where prediction outputs will be saved.
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
            folds=None,  # pyright: ignore[reportArgumentType]
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
) -> NDArray[np.uint8]:
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
    NDArray[np.uint8]
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
    seg_lh, _ = load_label_map(lh_seg_fname, dtype=np.uint8)
    seg_lh = change_labels(seg_lh, old_labels_hemi, new_labels_lh)

    seg_rh, _ = load_label_map(rh_seg_fname, dtype=np.uint8)
    seg_rh = change_labels(seg_rh, old_labels_hemi, new_labels_rh)

    seg_ant, _ = load_label_map(anterior_seg_fname, dtype=np.uint8)
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
