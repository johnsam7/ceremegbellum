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

import numpy as np
import nibabel as nib
import os

__all__ = [
    'save_nifti_from_3darray', 'set_nnunet_paths', 'change_labels',
    'rotation', 'translation', 'scale',
    'affine_transform', 'find_connected_regions',
]


def save_nifti_from_3darray(vol, fname, rotate=False, affine=None):
    if rotate:
        vol = vol[:, ::-1, ::-1]
        vol = np.transpose(vol, axes=(0, 2, 1))
    mgz = nib.Nifti1Image(vol, affine=affine)
    nib.save(mgz, fname)
    print('saved to ' + fname)
    return mgz


def set_nnunet_paths(raw_data_base=None, preprocessed=None, results_folder=None):
    """Set nnU-Net environment variables.

    Parameters
    ----------
    raw_data_base : str, optional
        Path to nnUNet raw data base directory.
    preprocessed : str, optional
        Path to nnUNet preprocessed directory.
    results_folder : str, optional
        Path to nnUNet results folder.
    """
    if raw_data_base is not None:
        os.environ["nnUNet_raw_data_base"] = raw_data_base
    if preprocessed is not None:
        os.environ["nnUNet_preprocessed"] = preprocessed
    if results_folder is not None:
        os.environ["RESULTS_FOLDER"] = results_folder

def change_labels(vol, old_labels, new_labels):
    new_vol = vol.copy()
    for c, old_label in enumerate(old_labels):
        mask_inds = np.where(vol == old_label)
        new_vol[mask_inds[0], mask_inds[1], mask_inds[2]] = new_labels[c]
    return new_vol

def rotation(angles, rr):
    """
    Rotates points rr around x-axis angles[0], y-axis angles[1] and z-axis angles[2]
    around its center of gravity.
    """
    a, b, c = angles
    rot_mat = np.array([[np.cos(b)*np.cos(c), -np.cos(b)*np.sin(c), np.sin(b)],
                   [np.sin(a)*np.sin(b)*np.cos(c) + np.cos(a)*np.sin(c),
                    -np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c),
                    -np.sin(a)*np.cos(b)],
                   [ -np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c),
                    np.cos(a)*np.sin(b)*np.sin(c) + np.sin(a)*np.cos(c),
                    np.cos(a)*np.cos(b)]])
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
    return c*rr

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
    voxels_removed = np.array([[]]).reshape((0,3))
    delta = np.array([[[[x, y, z] for x in np.arange(-1, 2)] for y in np.arange(-1, 2)] for z in np.arange(-1, 2)]).reshape(27,3)
    labels = list(np.unique(vol))
    labels.remove(0)
    labels2regions = {}

    for val in labels:
        voxels_in_label = np.array(np.where(vol==val)).T

        f_vox2region = {}
        for c, vox in enumerate(voxels_in_label):
            f_vox2region.update({tuple(vox) : 0}) # 0 means "unassigned"

        regions = []
        while 0 in list(f_vox2region.values()):
            unassigned_voxels = np.array(list(f_vox2region.keys()))[np.where(np.array(list(f_vox2region.values()))==0)[0]]
            f_vox2region.update({tuple(unassigned_voxels[0]) : 1})
            front_line_vols = unassigned_voxels[0].reshape((1,3))
            saved_vols = [tuple(unassigned_voxels[0])]
            while len(front_line_vols) > 0:
                for vox in front_line_vols:
                    neighbors = vox + delta
                    for neighbor in neighbors:
                        if tuple(neighbor) in list(f_vox2region.keys()):
                            if f_vox2region[tuple(neighbor)] == 0:
                                front_line_vols = np.vstack((front_line_vols, neighbor))
                                saved_vols.append(tuple(neighbor))
                                f_vox2region.update({tuple(neighbor) : 1})
                    front_line_vols = front_line_vols[1: front_line_vols.shape[0], :]
                if print_progress:
                    print(len(front_line_vols))
            regions.append(saved_vols)
        labels2regions.update({val : regions})
        if print_progress:
            print('Done with label '+str(val))

    return labels2regions
