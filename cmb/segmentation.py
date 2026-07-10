"""Provides functions for cerebellar segmentation using nnUNet and ANTs registration."""

import os
import os.path as op
import pickle

import nibabel as nib
import numpy as np

from .helpers import change_labels, save_nifti_from_3darray, set_nnunet_paths


def get_segmentation(
    subjects_dir,
    subject,
    cmb_path=None,
    region_removal_limit=0.2,
    post_process=True,
    print_progress=False,
    debug_mode=False,
):
    import ants

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
        return nib.load(op.join(segm_data_dir, subject + ".nii.gz"))

    else:  # If not, make segmentation with trained nnUnet model
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

        # Load brain template to get a common space
        brain_template_nib = nib.load(op.join(cmb_path, "data", "brain.nii"))
        brain_template = np.asanyarray(brain_template_nib.dataobj)
        brain_template = brain_template / np.max(brain_template)
        template_ants = ants.from_numpy(brain_template)

        output_folder = op.join(segm_data_dir, "tmp")
        reg_cache_file = op.join(
            output_folder, "registered", subject + "_reg_cache.pkl"
        )
        whole_file = op.join(
            output_folder, "registered", "whole", subject + "_0000.nii.gz"
        )

        # Check if registration was already completed
        if op.exists(reg_cache_file) and op.exists(whole_file):
            print(
                "Previous registration found for subject "
                + subject
                + ". Loading cached transforms."
            )
            with open(reg_cache_file, "rb") as f:
                reg = pickle.load(f)
            subj_reg = np.asanyarray(nib.load(whole_file).dataobj)
            subject_mri = nib.load(op.join(subjects_dir, subject, "mri", "brain.mgz"))
        else:
            # Register
            orig_fname = op.join(subjects_dir, subject, "mri", "brain.mgz")
            subject_mri = nib.load(orig_fname)
            subj_brain = np.asanyarray(subject_mri.dataobj)
            subj_brain = subj_brain / np.max(subj_brain)

            # Find registration from subject to common space
            subj_ants = ants.from_numpy(subj_brain)

            # Calculate registration
            print("Registering subject to template space...")
            reg = ants.registration(
                fixed=template_ants, moving=subj_ants, type_of_transform="SyNCC"
            )

            # Save registration cache
            with open(reg_cache_file, "wb") as f:
                pickle.dump(reg, f)

            # Prepare for masking
            subj_reg_ants = ants.apply_transforms(
                fixed=template_ants,
                moving=subj_ants,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            subj_reg = subj_reg_ants.numpy()
            save_nifti_from_3darray(
                subj_reg, whole_file, affine=brain_template_nib.affine
            )

        # Mask
        mask_output = op.join(output_folder, "registered", "mask", subject + ".nii.gz")
        if op.exists(mask_output):
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
                model_folder,
                op.join(output_folder, "registered", "whole"),
                op.join(output_folder, "registered", "mask"),
            )

        # Split into LH and RH using ASEG
        lh_input = op.join(output_folder, "registered", "lh", subject + "_0000.nii.gz")
        rh_input = op.join(output_folder, "registered", "rh", subject + "_0000.nii.gz")
        if op.exists(lh_input) and op.exists(rh_input):
            print("Previous hemisphere split found. Skipping split step.")
        else:
            aseg = np.asanyarray(
                nib.load(op.join(subjects_dir, subject, "mri", "aseg.mgz")).dataobj
            ).astype("uint8")
            aseg = ants.from_numpy(aseg)
            aseg_reg = ants.apply_transforms(
                fixed=template_ants,
                moving=aseg,
                transformlist=reg["fwdtransforms"],
                interpolator="genericLabel",
            ).numpy()
            mask = np.asanyarray(nib.load(mask_output).dataobj)
            split_cerebellar_hemis_aseg(
                aseg_reg,
                subj_reg,
                mask,
                subject,
                op.join(output_folder, "registered"),
                brain_template_nib.affine,
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
            pred_nib = nib.load(lh_seg_output)
            vol = np.asanyarray(pred_nib.dataobj)
            image = np.asanyarray(
                nib.load(
                    op.join(output_folder, "registered", "lh", subject + "_0000.nii.gz")
                ).dataobj
            )
            lobI_IV = np.zeros(vol.shape)
            lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
            pred_nib = nib.load(rh_seg_output)
            vol = np.asanyarray(pred_nib.dataobj)
            image = np.asanyarray(
                nib.load(
                    op.join(output_folder, "registered", "rh", subject + "_0000.nii.gz")
                ).dataobj
            )
            lobI_IV[np.where(vol == 2)] = image[np.where(vol == 2)]
            save_nifti_from_3darray(
                lobI_IV,
                op.join(
                    output_folder, "registered", "lob_I_IV", subject + "_0000.nii.gz"
                ),
                rotate=False,
                affine=pred_nib.affine,
            )
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

        # Correct labels
        old_labels_ant = [1, 2, 3, 4]
        new_labels_ant = [33, 43, 36, 46]
        old_labels_hemi = np.arange(1, 17)
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

        # Assemble segmentations into one image
        seg = np.asanyarray(
            nib.load(
                op.join(
                    output_folder, "registered", "lh_segmented", subject + ".nii.gz"
                )
            ).dataobj
        ).astype("uint8")
        seg_lh = change_labels(seg, old_labels_hemi, new_labels_lh)
        seg = np.asanyarray(
            nib.load(
                op.join(
                    output_folder, "registered", "rh_segmented", subject + ".nii.gz"
                )
            ).dataobj
        ).astype("uint8")
        seg_rh = change_labels(seg, old_labels_hemi, new_labels_rh)
        seg = np.asanyarray(
            nib.load(
                op.join(
                    output_folder,
                    "registered",
                    "lob_I_IV_segmented",
                    subject + ".nii.gz",
                )
            ).dataobj
        ).astype("uint8")
        seg_ant = change_labels(seg, old_labels_ant, new_labels_ant)
        seg_complete = np.zeros(seg.shape)
        seg_complete[np.nonzero(seg_lh)] = seg_lh[np.nonzero(seg_lh)]
        seg_complete[np.nonzero(seg_rh)] = seg_rh[np.nonzero(seg_rh)]
        seg_complete[np.nonzero(seg_ant)] = seg_ant[np.nonzero(seg_ant)]
        seg_ants = ants.from_numpy(seg_complete)

        # Go back to subject space
        seg_reg = ants.apply_transforms(
            fixed=template_ants,
            moving=seg_ants,
            transformlist=reg["invtransforms"],
            interpolator="genericLabel",
        ).numpy()

        save_nifti_from_3darray(
            seg_reg,
            op.join(segm_data_dir, subject + ".nii.gz"),
            rotate=False,
            affine=subject_mri.affine,
        )

        if not debug_mode:
            for rel_path in rel_paths:
                cleanup_dir = op.join(segm_data_dir, rel_path)
                if op.exists(cleanup_dir):
                    for f in os.listdir(cleanup_dir):
                        if f.endswith((".nii.gz", ".pkl", ".json")):
                            os.remove(op.join(cleanup_dir, f))
        return nib.load(op.join(segm_data_dir, subject + ".nii.gz"))


def split_cerebellar_hemis_aseg(aseg, brain, mask, subject, output_folder, affine):
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
        rotate=False,
        affine=affine,
    )
    save_nifti_from_3darray(
        lh_split,
        op.join(output_folder, "lh", subject + "_0000.nii.gz"),
        rotate=False,
        affine=affine,
    )
    save_nifti_from_3darray(
        rh_split,
        op.join(output_folder, "rh", subject + "_0000.nii.gz"),
        rotate=False,
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
