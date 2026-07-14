#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper utilities for geometric transforms, connected-region analysis,
and NIfTI I/O.
"""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: September, 2021
# License: MIT
# ---------------------------------------------------------------------------

import logging
import os
from typing import Any

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from numpy.typing import DTypeLike, NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "save_nifti_from_3darray",
    "set_nnunet_paths",
    "change_labels",
    "rotation",
    "translation",
    "scale",
    "affine_transform",
    "find_connected_regions",
]


def load_image_volume(
    fname: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Load continuous-intensity brain data to NumPy array.

    Uses `get_fdata()` method which ensures that any
    internal NIfTI scaling factors are applied automatically. The resulting array is
    guaranteed to be high-precision float64.

    Parameters
    ----------
    fname : str
        Path to the data file to be loaded. Format should be NIfTI, MGH
        (from FreeSurfer) or other format supported by `nibabel`.

    Returns
    -------
    img_numpy : NDArray[np.float64]
        3D NumPy array containing the image data.
    affine : NDArray[np.float64] | None
        The affine transformation matrix associated with the image data.
        Will be None if the saved file does not contain an affine matrix.
    """
    img = nib.load(fname)
    # Ignore warnings due to img being generic FileBasedImage type.
    # At runtime it should be a subclass that has the correct methods.
    img_numpy = img.get_fdata()  # pyright: ignore[reportAttributeAccessIssue]
    affine = img.affine  # pyright: ignore[reportAttributeAccessIssue]

    return img_numpy, affine


def load_label_map(
    fname: str, dtype: DTypeLike | None = None
) -> tuple[NDArray[Any], NDArray[np.float64] | None]:
    """Load discrete integer labels to a NumPy array.

    Accesses the `dataobj` attribute of the loaded image and converts it to a NumPy
    array of specified type. This is useful for loading label maps without converting
    them to float (and wasting memory) as would happen with `get_fdata()`.

    Parameters
    ----------
    fname : str
        Path to the label map file to be loaded. Format should be NIfTI, MGH
        (from FreeSurfer) or other format supported by `nibabel`.
    dtype : DTypeLike | None
        The data type of the returned array. If None (default), the data type
        will be inferred from the image data.

    Returns
    -------
    label_map : NDArray[Any]
        3D NumPy array containing the label map data.
    affine : NDArray[np.float64] | None
        The affine transformation matrix associated with the label map data.
        Will be None if the saved file does not contain an affine matrix.
    """
    img = nib.load(fname)
    # Ignore warnings due to img being generic FileBasedImage type.
    # At runtime it should be a subclass that has the correct methods.
    image_data = img.dataobj  # pyright: ignore[reportAttributeAccessIssue]
    affine = img.affine  # pyright: ignore[reportAttributeAccessIssue]
    label_map = np.asanyarray(image_data)
    logger.debug(
        "Loaded label map %s with shape %s and dtype %s",
        fname,
        label_map.shape,
        label_map.dtype,
    )
    if dtype is None:
        return label_map, affine
    # Make sure the cast would not cause overflow.
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        # Only check for integer types.
        info = np.iinfo(target_dtype.type)
        min_label = label_map.min()
        max_label = label_map.max()
        logger.debug(
            "Label map %s has min value %d and max value %d",
            fname,
            min_label,
            max_label,
        )
        if min_label < info.min or max_label > info.max:
            raise ValueError(
                f"Cannot safely cast the label map from file {fname} to {dtype}. "
                f"Label map values are in the range [{min_label}, {max_label}], "
                "but the target type can only represent values in the range "
                f"[{info.min}, {info.max}]."
            )
    # Do the safe cast.
    logger.debug("Casting label map %s to dtype %s", fname, dtype)
    label_map = label_map.astype(dtype)

    return label_map, affine


def save_nifti_from_3darray(
    vol: np.ndarray, fname: str, affine: np.ndarray | None
) -> Nifti1Image:
    """
    Save a 3D numpy array as a NIfTI file.

    Parameters
    ----------
    vol : np.ndarray
        3D numpy array to be saved as a NIfTI file.
    fname : str
        Path to the output NIfTI file.
    affine : np.ndarray | None
        The affine transformation matrix for the NIfTI file.

    Returns
    -------
    nibabel.Nifti1Image
        The saved NIfTI image object.
    """
    mgz = nib.Nifti1Image(vol, affine=affine)
    nib.save(mgz, fname)
    print("saved to " + fname)
    return mgz


def set_nnunet_paths(
    raw_data_base_dir=None, preprocessed_dir=None, results_folder=None
) -> None:
    """Set nnU-Net environment variables.

    Parameters
    ----------
    raw_data_base_dir : str, optional
        Path to nnUNet raw data base directory. If None (default), skips setting this
        environment variable.
    preprocessed_dir : str, optional
        Path to nnUNet preprocessed directory. If None (default), skips setting this
        environment variable.
    results_folder : str, optional
        Path to nnUNet results folder. If None (default), skips setting this
        environment variable.
    """
    if raw_data_base_dir is not None:
        os.environ["nnUNet_raw_data_base"] = raw_data_base_dir
    if preprocessed_dir is not None:
        os.environ["nnUNet_preprocessed"] = preprocessed_dir
    if results_folder is not None:
        os.environ["RESULTS_FOLDER"] = results_folder


def change_labels(
    vol: np.ndarray, old_labels: list[int], new_labels: list[int]
) -> np.ndarray:
    """Replace specific labels in segmentation volume with new labels.

    Does not modify the original volume; returns a new volume with the specified labels
    replaced.

    Parameters
    ----------
    vol : np.ndarray
        3D numpy array mapping each voxel to a label.
    old_labels : list[int]
        List of labels to be replaced.
    new_labels : list[int]
        List of new labels to replace the old labels. The order of these new labels
        corresponds to the order of the old labels.

    Returns
    -------
    np.ndarray
        3D numpy array with the specified labels replaced.
    """
    if len(old_labels) != len(new_labels):
        raise ValueError("old_labels and new_labels must have the same length.")
    new_vol = vol.copy()
    for old_label, new_label in zip(old_labels, new_labels):
        new_vol[vol == old_label] = new_label

    return new_vol


def rotation(angles, rr):
    """
    Rotates points rr around x-axis angles[0], y-axis angles[1] and z-axis angles[2]
    around its center of gravity.
    """
    a, b, c = angles
    rot_mat = np.array(
        [
            [np.cos(b) * np.cos(c), -np.cos(b) * np.sin(c), np.sin(b)],
            [
                np.sin(a) * np.sin(b) * np.cos(c) + np.cos(a) * np.sin(c),
                -np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
                -np.sin(a) * np.cos(b),
            ],
            [
                -np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c),
                np.cos(a) * np.sin(b) * np.sin(c) + np.sin(a) * np.cos(c),
                np.cos(a) * np.cos(b),
            ],
        ]
    )
    rr_center = np.mean(rr, axis=0)
    rr_n = rr - rr_center
    rr_n = np.dot(rot_mat, rr_n.T).T
    rr_n = rr_n + rr_center
    return rr_n


def translation(r_0, rr):
    """
    Translates points rr by r_0
    """
    return rr + r_0


def scale(c, rr):
    """
    Scales points rr by c
    """
    return c * rr


def affine_transform(c, r_0, angles, rr):
    """
    Performs an affine transformation by rotation, translation and scaling.
    """
    rr = rotation(angles, rr)
    rr = translation(r_0, rr)
    rr = scale(c, rr)
    return rr


def find_connected_regions(vol, print_progress=True):
    """
    Finds connected regions in vol labeled by integers except for 0.
    """
    f_vox2int = {}
    voxels_removed = np.array([[]]).reshape((0, 3))
    delta = np.array(
        [
            [[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)]
            for z in np.arange(-1, 2)
        ]
    ).reshape(27, 3)
    labels = list(np.unique(vol))
    labels.remove(0)
    labels2regions = {}

    for val in labels:
        voxels_in_label = np.array(np.where(vol == val)).T

        f_vox2region = {}
        for c, vox in enumerate(voxels_in_label):
            f_vox2region.update({tuple(vox): 0})  # 0 means "unassigned"

        regions = []
        while 0 in list(f_vox2region.values()):
            unassigned_voxels = np.array(list(f_vox2region.keys()))[
                np.where(np.array(list(f_vox2region.values())) == 0)[0]
            ]
            f_vox2region.update({tuple(unassigned_voxels[0]): 1})
            front_line_vols = unassigned_voxels[0].reshape((1, 3))
            saved_vols = [tuple(unassigned_voxels[0])]
            while len(front_line_vols) > 0:
                for vox in front_line_vols:
                    neighbors = vox + delta
                    for neighbor in neighbors:
                        if tuple(neighbor) in list(f_vox2region.keys()):
                            if f_vox2region[tuple(neighbor)] == 0:
                                front_line_vols = np.vstack((front_line_vols, neighbor))
                                saved_vols.append(tuple(neighbor))
                                f_vox2region.update({tuple(neighbor): 1})
                    front_line_vols = front_line_vols[1 : front_line_vols.shape[0], :]
                if print_progress:
                    print(len(front_line_vols))
            regions.append(saved_vols)
        labels2regions.update({val: regions})
        if print_progress:
            print("Done with label " + str(val))

    return labels2regions
