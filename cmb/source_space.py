#!/usr/bin/env python3
"""Source space construction for combined cerebral and cerebellar MEG/EEG analysis.

Provides functions to set up cerebellar surface source spaces, register them
to individual subject anatomy via ANTs diffeomorphic registration, and merge
them with MNE-Python cortical source spaces.
"""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
# Created: November, 2021
# License: MIT
# ---------------------------------------------------------------------------

import os
import pickle

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from .helpers import affine_transform, change_labels
from .segmentation import get_segmentation
from .visualization import plot_sagittal


def print_fs_surf(rr, tris, fname, mirror=False):
    """Convert to RAS coords and print surface to be plotted with Freeview."""
    fsVox2RAS = np.array([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128]]).T

    fs_vox = np.hstack((rr, np.ones((len(rr), 1))))
    ras = np.dot(fs_vox, fsVox2RAS)
    if mirror:
        ras[:, 0] = -ras[
            :, 0
        ]  # Note this is for MNE which mirrors the source space - remove this for alignment with RAS freeview
    nib.freesurfer.io.write_geometry(fname, ras, tris)


def setup_cerebellum_source_space(
    subjects_dir,
    subject,
    cmb_path=None,
    cerebellum_subsampling="sparse",
    calc_nn=True,
    print_fs=False,
    plot=False,
    mirror=False,
    post_process=False,
    debug_mode=False,
):
    """Set up the cerebellar surface source space.

    Requires cerebellum geometry file to be downloaded.

    Parameters
    ----------
    subjects_dir : str
        Subjects directory.
    subject : str
        Subject name.
    cmb_path : str, optional
        Path to cerebellum data folder. If None, defaults to the package
        installation directory.
    cerebellum_subsampling : 'full' | 'sparse' | 'dense'
        The spacing to use for the cerebellum.
    calc_nn: Boolean
        If True, it will calculate the normals of the cerebellum source space.
    print_fs : Boolean
        If True, it will print an fs file of the cerebellar source space that can be
        viewed with e.g. freeview.
    plot : Boolean
        If True, will plot sagittal cross-sectional plots of the cerebellar source space
        supposed on subject MR data.

    Returns
    -------
    subj_cerb: dictionary
        Dictionary containing geometry data: vertex positions (rr), faces (tris) and
        normals (nn, if calc_nn is True).

    """
    import ants
    import pandas as pd
    from scipy import signal

    if cmb_path is None:
        from . import CMB_DATA_DIR

        cmb_path = CMB_DATA_DIR

    print("starting subject " + subject + "...")
    # Load data
    subj_cerb = {}
    data_dir = os.path.join(cmb_path, "data")
    cb_data = pickle.load(open(os.path.join(data_dir, "cerebellum_geo"), "rb"))
    if cerebellum_subsampling == "full":
        rr = cb_data["verts_normal"]
        tris = cb_data["faces"]
    else:
        rr = cb_data["dw_data"][cerebellum_subsampling + "_verts"]
        tris = cb_data["dw_data"][cerebellum_subsampling + "_tris"]
        rr = affine_transform(1, np.array([0, 0, 0]), [np.pi / 2, 0, 0], rr)

    hr_vol = cb_data["hr_vol"]
    hr_segm = cb_data["parcellation"]["volume"].copy()
    old_labels = [
        12,
        33,
        36,
        43,
        46,
        53,
        56,
        60,
        63,
        66,
        70,
        73,
        74,
        75,
        76,
        77,
        78,
        80,
        83,
        84,
        86,
        87,
        90,
        93,
        96,
        100,
        103,
        106,
    ]
    hr_segm = change_labels(
        hr_segm, old_labels=old_labels, new_labels=list(range(1, 29))
    )
    # Get subject segmentation
    subj_segm = np.asanyarray(
        get_segmentation(
            subjects_dir,
            subject,
            cmb_path,
            debug_mode=debug_mode,
        ).dataobj
    )
    subj_mri = np.asanyarray(
        nib.load(os.path.join(subjects_dir, subject, "mri", "orig.mgz")).dataobj
    )

    # Crop the segmentation and the MRI to the bounding box of the cerebellum.
    pad = 3
    cerb_coords = np.nonzero(subj_segm)  # cerebellum is nonzero in segmentation
    subj_segm_cropped = _crop_image_volume(subj_segm, cerb_coords, pad=pad)

    subj_contrast = np.zeros(subj_mri.shape)
    subj_contrast[cerb_coords] = subj_mri[cerb_coords]
    subj_contrast_cropped = _crop_image_volume(subj_contrast, cerb_coords, pad=pad)

    print("Setting up adaptation to subject... ", end="", flush=True)
    hr_vol_scaled = hr_vol
    for axis in range(0, 3):
        hr_vol_scaled = signal.resample(
            hr_vol_scaled, num=subj_segm_cropped.shape[axis], axis=axis
        )
    scf = np.array(hr_vol_scaled.shape) / np.array(hr_vol.shape)
    for x in range(3):
        rr[:, x] = rr[:, x] * scf[x]
    hr_rs = np.zeros(hr_vol_scaled.shape)
    non_zero_coo_50 = np.array([np.where(hr_vol_scaled > 50)[x] for x in range(3)]).T
    non_zero_coo = np.array([np.where(hr_vol_scaled > 10)[x] for x in range(3)]).T
    hr_rs[non_zero_coo[:, 0], non_zero_coo[:, 1], non_zero_coo[:, 2]] = hr_vol_scaled[
        non_zero_coo[:, 0], non_zero_coo[:, 1], non_zero_coo[:, 2]
    ]

    # scale labels matrix (by type value vote)
    hr_label_scaled = np.zeros(
        (
            subj_segm_cropped.shape[0],
            subj_segm_cropped.shape[1],
            subj_segm_cropped.shape[2],
        )
    )
    count_matrix = np.zeros(
        (
            subj_segm_cropped.shape[0],
            subj_segm_cropped.shape[1],
            subj_segm_cropped.shape[2],
            100,
        )
    )
    count_matrix[:] = np.nan
    for x in range(hr_segm.shape[0]):
        for y in range(hr_segm.shape[1]):
            for z in range(hr_segm.shape[2]):
                target_vox = (scf * (x, y, z)).astype(int)
                ind = np.min(
                    np.where(
                        np.isnan(
                            count_matrix[target_vox[0], target_vox[1], target_vox[2], :]
                        )
                    )
                )
                count_matrix[target_vox[0], target_vox[1], target_vox[2], ind] = (
                    hr_segm[x, y, z]
                )
    for x in range(subj_segm_cropped.shape[0]):
        for y in range(subj_segm_cropped.shape[1]):
            for z in range(subj_segm_cropped.shape[2]):
                votes = count_matrix[x, y, z, :]
                votes = votes[~np.isnan(votes)]
                hr_label_scaled[x, y, z] = np.bincount(votes.astype(int)).argmax()

    # Correct verts by co-registering lower left posterior and upper right anterior corners
    correction_vector_2 = np.mean(
        [
            np.min(non_zero_coo_50, axis=0) - np.min(rr, axis=0),
            np.max(non_zero_coo_50, axis=0) - np.max(rr, axis=0),
        ],
        axis=0,
    )
    rr = rr + correction_vector_2
    print("Done.")

    # Register
    print("Fitting... ", end="", flush=True)
    subj_vec = subj_segm_cropped
    hr_vec = hr_label_scaled

    print("Fitting labels... ", end="", flush=True)
    subj_label_ants = ants.from_numpy(subj_vec.astype(float))
    hr_label_ants = ants.from_numpy(hr_label_scaled.astype(float))
    reg = ants.registration(
        fixed=subj_label_ants, moving=hr_label_ants, type_of_transform="SyNCC"
    )
    def_hr_label = ants.apply_transforms(
        fixed=subj_label_ants,
        moving=hr_label_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="genericLabel",
    )
    vox_dir = {"x": list(rr[:, 0]), "y": list(rr[:, 1]), "z": list(rr[:, 2])}
    pts = pd.DataFrame(data=vox_dir)
    rrw_0 = np.array(ants.apply_transforms_to_points(3, pts, reg["invtransforms"]))

    print("Fitting contrast... ")
    subj_contrast_cropped = subj_contrast_cropped / np.max(subj_contrast_cropped)
    hr_rs = hr_rs / np.max(hr_rs)
    subj_ants = ants.from_numpy(subj_contrast_cropped)
    hr_rs_ants = ants.from_numpy(hr_rs)
    hr_ants = ants.apply_transforms(
        fixed=subj_ants, moving=hr_rs_ants, transformlist=reg["fwdtransforms"]
    )
    reg = ants.registration(fixed=subj_ants, moving=hr_ants, type_of_transform="SyNCC")
    vox_dir = {"x": list(rrw_0[:, 0]), "y": list(rrw_0[:, 1]), "z": list(rrw_0[:, 2])}
    pts = pd.DataFrame(data=vox_dir)
    rrw_1 = np.array(ants.apply_transforms_to_points(3, pts, reg["invtransforms"]))
    hr_label_final = ants.apply_transforms(
        fixed=subj_ants,
        moving=def_hr_label,
        transformlist=reg["fwdtransforms"],
        interpolator="genericLabel",
    )

    rr_p = rrw_1 + cb_range[0]
    subj_cerb.update({"rr": rr_p})
    subj_cerb.update({"tris": tris})
    print("Done.")

    if calc_nn:
        print("Calculating normals on deformed surface...", end="", flush=True)
        (nn_def, area, area_list, nan_vertices) = calculate_normals(
            rr_p, tris, print_info=False
        )
        subj_cerb.update({"nn": nn_def})
        subj_cerb.update({"nan_nn": nan_vertices})
        print("Done.")

    # Visualize results as sagittal (x=const) cross-sections
    if plot:
        fig, ax = plot_sagittal(
            subj_mri, title="Warped points in subj vol", rr=rr_p, tris=tris
        )

    if print_fs:
        print("Saving cerebellar surface as fs files...")
        rr_def = rr_p.copy()
        for x in range(3):
            rr_def[:, x] = rr_p[:, x]
        print_fs_surf(
            rr_def, tris, os.path.join(data_dir, subject + "_cerb_cxw.fs"), mirror
        )
        print("Saved to " + os.path.join(data_dir, subject + "_cerb_cxw.fs"))

    return subj_cerb


def _crop_image_volume(
    vol: NDArray, coords: tuple[NDArray, ...], pad: int = 3
) -> NDArray:
    """Crop a 3D volume to the bounding box of non-zero coordinates.

    Parameters
    ----------
    vol : NDArray
        The 3D volume to be cropped.
    coords : tuple[NDArray, ...]
        Indices of elements to be included in the cropped volume. Typically obtained
        from `np.nonzero()`.
    pad : int, optional
        The number of voxels to pad around the bounding box. Default is 3.

    Returns
    -------
    NDArray
        The cropped volume.
    """
    # coords_range is [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    coords_range = [
        [max(0, np.min(coords[x]) - pad) for x in range(3)],
        [min(vol.shape[x], np.max(coords[x]) + pad + 1) for x in range(3)],
    ]
    volume_cropped = vol[
        coords_range[0][0] : coords_range[1][0],
        coords_range[0][1] : coords_range[1][1],
        coords_range[0][2] : coords_range[1][2],
    ]
    return volume_cropped


def calculate_normals(
    rr, tris, solid_angle_calc=False, obs_point=np.zeros(3), print_info=True
):
    """Calculate vertex normals for a triangular mesh.

    Parameters
    ----------
    rr : ndarray
        Array of vertex positions.
    tris : ndarray
        Triangle indices defining the mesh faces.
    solid_angle_calc : bool, optional
        Whether to compute the solid angle from ``obs_point``.
    obs_point : ndarray, optional
        Observation point used for solid angle calculation.
    print_info : bool, optional
        Whether to print diagnostic information.

    Returns
    -------
    ndarray
        Vertex normals computed as an unweighted average of neighboring face normals.
    """
    A = []
    area_list = []
    area = 0.0
    count = 0
    nan_vertices = []
    for x in range(len(rr)):
        A.append([])
    solid_angle = 0
    for row in tris:
        v1 = rr[row[1], :] - rr[row[0], :]
        v2 = rr[row[2], :] - rr[row[0], :]
        nml = np.cross(v1, v2)
        area = area + np.linalg.norm(nml) / 2.0
        area_list.append(area)
        nn_fc = nml / np.linalg.norm(nml)
        A[row[0]].append(nn_fc)
        A[row[1]].append(nn_fc)
        A[row[2]].append(nn_fc)
        if solid_angle_calc == True:
            R1 = rr[row[0]] - obs_point
            R2 = rr[row[1]] - obs_point
            R3 = rr[row[2]] - obs_point

            solid_angle = solid_angle + 2 * np.arctan(
                np.dot(R1, np.cross(R2, R3))
                / (
                    np.linalg.norm(R1) * np.linalg.norm(R2) * np.linalg.norm(R3)
                    + np.dot(R1, R2) * np.linalg.norm(R3)
                    + np.dot(R1, R3) * np.linalg.norm(R2)
                    + np.dot(R2, R3) * np.linalg.norm(R1)
                )
            )

    if solid_angle_calc == True:
        print("solid_angle at the point of observation estimated to:")
        print(solid_angle)

    nn = np.zeros((len(rr), 3))
    for c, ele in enumerate(A):
        vert_norm = np.zeros(3)
        for vec in ele:
            vert_norm = vert_norm + vec
        vert_norm = vert_norm / np.linalg.norm(vert_norm)
        nn[c, :] = vert_norm

    for c, ele in enumerate(A):
        if np.isnan(nn[c, :]).any():  # np.linalg.norm(nn[c,:]) == 0:
            neighbor_rows = np.where(tris == c)[0]
            neighbors = np.unique(tris[neighbor_rows])
            neighbors = neighbors[np.where(neighbors != c)]
            normal = np.mean(nn[neighbors, :], axis=0)
            nn[c, :] = normal / np.linalg.norm(normal)
            count = count + 1

            if np.isnan(nn[c, :]).any():
                nan_vertices.append(c)

    if print_info:
        print("number of nan normals that have been smoothed = " + str(count))
        print("Remaining NAN normals = " + str(len(nan_vertices)))
        print("Total surface area: " + str(area))

    return (nn, area, area_list, nan_vertices)


def setup_full_source_space(
    subject,
    subjects_dir,
    cerb_src_geometry=None,
    cerb_dir=None,
    cerb_subsampling="sparse",
    spacing="oct6",
    plot_cerebellum=False,
    debug_mode=False,
):
    """Set up a full surface source space that includes the cerebellum.

    The first element in the returned list is the combined cerebral hemispheric source
    space and the second element is the cerebellar source space.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    cerb_src_geometry : dict, optional
        Dictionary containing the cerebellar source space geometry as returned by
        ``setup_cerebellum_source_space``. If None, the function will call
        ``setup_cerebellum_source_space`` to create the cerebellar source space.
    cerb_dir : str, optional
        Path to cerebellum data folder. If None, defaults to the package
        installation directory.
    cerb_subsampling : 'full' | 'sparse' | 'dense'
        The spacing to use for the cerebellum. Can be either full, sparse or dense.
    spacing : str
        The spacing to use for cortex. Can be ``'ico#'`` for a recursively subdivided
        icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
        or ``'all'`` for all points.
    plot_cerebellum : Boolean
        If True, will plot sagittal cross-sectional plots of the cerebellar
        source space superposed on subject MR data.
    debug_mode : Boolean
        If True, intermediate results will be saved to disk.


    Returns
    -------
    src_whole: list
        List containing two source space elements: the cerebral cortex and the
        cerebellar cortex.

    """
    import mne

    if cerb_dir is None:
        from . import CMB_DATA_DIR

        cerb_dir = CMB_DATA_DIR

    assert cerb_subsampling in ["full", "sparse", "dense"], (
        "cerb_subsampling must be either 'full', 'sparse' or 'dense'"
    )
    src_cort = mne.setup_source_space(
        subject=subject, subjects_dir=subjects_dir, spacing=spacing, add_dist=False
    )
    if spacing == "all":
        src_cort[0]["use_tris"] = src_cort[0]["tris"]
        src_cort[1]["use_tris"] = src_cort[1]["tris"]
    if cerb_src_geometry is not None:
        cerb_subj_data = cerb_src_geometry
    else:
        cerb_subj_data = setup_cerebellum_source_space(
            subjects_dir,
            subject,
            cerb_dir,
            calc_nn=True,
            cerebellum_subsampling=cerb_subsampling,
            print_fs=True,
            plot=plot_cerebellum,
            mirror=False,
            post_process=True,
            debug_mode=debug_mode,
        )
    rr = (
        mne.read_surface(os.path.join(cerb_dir, "data", subject + "_cerb_cxw.fs"))[0]
        / 1000
    )
    src_whole = src_cort.copy()
    hemi_src = join_source_spaces(src_cort)
    src_whole[0] = hemi_src
    src_whole[1]["rr"] = rr
    src_whole[1]["tris"] = cerb_subj_data["tris"]
    src_whole[1]["nn"] = cerb_subj_data["nn"]
    src_whole[1]["ntri"] = src_whole[1]["tris"].shape[0]
    src_whole[1]["use_tris"] = cerb_subj_data["tris"]
    in_use = np.ones(rr.shape[0]).astype(int)
    in_use[cerb_subj_data["nan_nn"]] = 0
    src_whole[1]["inuse"] = in_use
    src_whole[1]["nuse"] = int(np.sum(src_whole[1]["inuse"]))
    src_whole[1]["vertno"] = np.nonzero(src_whole[1]["inuse"])[0]
    src_whole[1]["np"] = src_whole[1]["rr"].shape[0]

    return src_whole


def join_source_spaces(src_orig):
    if len(src_orig) != 2:
        raise ValueError("Input must be two source spaces")

    src_joined = src_orig.copy()
    src_joined = src_joined[0]
    src_joined["inuse"] = np.concatenate((src_orig[0]["inuse"], src_orig[1]["inuse"]))
    src_joined["nn"] = np.concatenate((src_orig[0]["nn"], src_orig[1]["nn"]), axis=0)
    src_joined["np"] = src_orig[0]["np"] + src_orig[1]["np"]
    src_joined["ntri"] = src_orig[0]["ntri"] + src_orig[1]["ntri"]
    src_joined["nuse"] = src_orig[0]["nuse"] + src_orig[1]["nuse"]
    src_joined["nuse_tri"] = src_orig[0]["nuse_tri"] + src_orig[1]["nuse_tri"]
    src_joined["rr"] = np.concatenate((src_orig[0]["rr"], src_orig[1]["rr"]), axis=0)
    src_joined["tris"] = np.concatenate(
        (src_orig[0]["tris"], src_orig[1]["tris"] + src_orig[0]["np"]), axis=0
    )
    try:
        src_joined["use_tris"] = np.concatenate(
            (src_orig[0]["use_tris"], src_orig[1]["use_tris"] + src_orig[0]["np"]),
            axis=0,
        )
    except Exception:
        import warnings

        warnings.warn(
            "Failed to concatenate use_tris, use_tris will be put to None. This means you will not be able to visualize"
            " the cortex in 3d but can still do all computational operations."
        )
        src_joined["use_tris"] = None
    src_joined["vertno"] = np.nonzero(src_joined["inuse"])[0]

    return src_joined
