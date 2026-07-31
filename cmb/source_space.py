#!/usr/bin/env python3
"""Source space construction for combined cerebral and cerebellar MEG/EEG analysis.

Provides functions to set up cerebellar surface source spaces, register them
to individual subject anatomy via ANTs diffeomorphic registration, and merge
them with MNE-Python cortical source spaces.
"""
# ---------------------------------------------------------------------------
# Authors: John G Samuelson <johnsam@mit.edu>
#          Christoph Dinh <christoph.dinh@brain-link.de>
#          Teemu Taivainen
# Created: November, 2021 (Modified: July, 2026)
# License: MIT
# ---------------------------------------------------------------------------

import logging
import os
import os.path as op
import pickle
import warnings
from typing import TYPE_CHECKING, Literal

import mne
import numpy as np
from nibabel.freesurfer.io import write_geometry
from numpy.typing import NDArray

from .helpers import (
    affine_transform,
    change_labels,
    convert_to_ants_image,
    load_image_volume,
)
from .segmentation import get_segmentation

if TYPE_CHECKING:
    from ants.core.ants_image import ANTsImage
    from pandas import DataFrame

logger = logging.getLogger(__name__)


def create_cerebellar_surface(
    subject: str,
    subjects_dir: str | None = None,
    cmb_path: str | None = None,
    cerebellum_subsampling: Literal["full", "sparse", "dense"] = "sparse",
    save_mesh: bool | str = True,
    registration_caching: bool = False,
) -> tuple[NDArray, NDArray]:
    """Create a cerebellar mesh in the native subject space.

    Runs the reconstruction step of ARCUS, fitting a high-resolution cerebellar atlas to
    the segmentation of the subject's cerebellum. Outputs a cerebellar mesh in
    FreeSurfer surface RAS coordinates.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    subjects_dir : str | None
        The path to the directory containing the FreeSurfer subjects reconstructions.
        If None, defaults to the SUBJECTS_DIR environment variable.
    cmb_path : str, optional
        Path to cerebellum data folder. If None, defaults to the package
        installation directory.
    cerebellum_subsampling : 'full' | 'sparse' | 'dense'
        The spacing to use for the cerebellum.
    save_mesh : bool | str
        If True (default), saves the cerebellar mesh to
        ``<subjects_dir>/<subject>/surf/cerebellum.white``. If a string is provided,
        saves the mesh to the specified path. If False, does not save the mesh to disk.
    registration_caching : Boolean
        If True, it will attemp to read cached registration transforms from disk, and if
        not found, will save the transforms to disk for future use. Defaults to False,
        which means that registration will be computed without caching.

    Returns
    -------
    rr_ras : NDArray
        n_vertices x 3 array of vertex positions in FreeSurfer surface RAS coordinates.
    tris : NDArray
        n_faces x 3 array of triangle vertex indices defining the mesh faces.
    """
    import ants
    from ants.registration import (
        apply_transforms,
        apply_transforms_to_points,
    )
    from scipy import signal

    # Use MNE-Python to fall back to SUBJECTS_DIR environment variable if needed.
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)  # pyright: ignore[reportAssignmentType]
    # Cast Path object to string for compatibility.
    subjects_dir = str(subjects_dir)

    if cmb_path is None:
        from . import CMB_DATA_DIR

        cmb_path = CMB_DATA_DIR

    if isinstance(save_mesh, str):
        mesh_fname = save_mesh
        os.makedirs(op.dirname(mesh_fname), exist_ok=True)
    elif save_mesh is True:
        mesh_fname = op.join(subjects_dir, subject, "surf", "cerebellum.white")
        os.makedirs(op.dirname(mesh_fname), exist_ok=True)
    else:
        mesh_fname = None

    data_dir = op.join(cmb_path, "data")

    if registration_caching:
        # Save registration transforms to a cache directory.
        registration_cache_dir = op.join(data_dir, "atlas_fitting_cache")
        os.makedirs(registration_cache_dir, exist_ok=True)
    else:
        # No caching.
        registration_cache_dir = None

    logger.info("Starting to create cerebellar surface for subject %s...", subject)

    with open(op.join(data_dir, "cerebellum_geo"), "rb") as cb_geo_file:
        cb_data = pickle.load(cb_geo_file)

    # Load the template cerebellar mesh. The mesh is in the voxel space of the
    # volumetric atlas.
    if cerebellum_subsampling == "full":
        rr = cb_data["verts_normal"]
        tris = cb_data["faces"]
    else:
        rr = cb_data["dw_data"][cerebellum_subsampling + "_verts"]
        tris = cb_data["dw_data"][cerebellum_subsampling + "_tris"]
        rr = affine_transform(1, np.array([0, 0, 0]), [np.pi / 2, 0, 0], rr)

    # Get the volumetric atlas (high resolution segmentation).
    hr_segm = cb_data["parcellation"]["volume"].copy()
    # Get virtual MRI volume of the atlas.
    hr_vol = cb_data["hr_vol"]

    # Adjust the labels of the volumetric atlas to match the subject's segmentation.
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
    # Get subject segmentation (registered to brain.mgz).
    subject_labels = np.asanyarray(
        get_segmentation(
            subjects_dir,
            subject,
            cmb_path,
            debug_mode=registration_caching,
        ).dataobj
    )
    # Get subject MRI.
    # orig.mgz is in same space as brain.mgz, so segmentation and orig.mgz are aligned.
    subj_mri, _ = load_image_volume(op.join(subjects_dir, subject, "mri", "orig.mgz"))

    # Crop the segmentation and the MRI to the bounding box of the cerebellum.
    pad = 3
    cerb_coords = np.nonzero(subject_labels)  # cerebellum is nonzero in segmentation
    subject_labels, cb_range = _crop_image_volume(subject_labels, cerb_coords, pad=pad)

    subj_contrast = np.zeros(subj_mri.shape)
    # Fill the cerebellum region with MRI values.
    # Can use cerb_coords to index because subject MRI and segmentation are aligned.
    subj_contrast[cerb_coords] = subj_mri[cerb_coords]
    subj_contrast, _ = _crop_image_volume(subj_contrast, cerb_coords, pad=pad)

    logger.info("Setting up adaptation to subject... ")

    # Resample the cerebellar volume to the subject's segmentation size.
    logger.debug("Resampling template MRI to the subject's segmentation size...")
    hr_vol_scaled = hr_vol
    for axis in range(3):
        hr_vol_scaled = signal.resample(
            hr_vol_scaled, num=subject_labels.shape[axis], axis=axis
        )
    # Help type checkers understand the type of hr_vol_scaled.
    assert isinstance(hr_vol_scaled, np.ndarray), (
        "Resampled signal should be a NumPy array."
    )
    scaling_factor = np.array(hr_vol_scaled.shape) / np.array(hr_vol.shape)
    logger.debug("Shape of volmetric atlas: %s", hr_vol.shape)
    logger.debug("Shape of resampled atlas: %s", hr_vol_scaled.shape)
    logger.debug("Scaling factor: %s", scaling_factor)

    # Clean up the resampled volume by removing low value voxels.
    # Voxels with value below 10 are set to zero.
    hr_volume_resampled = np.where(hr_vol_scaled > 10, hr_vol_scaled, 0)

    # Resample the segmentation to the subject's segmentation size.
    logger.debug(
        "Resampling template segmentation to the subject's segmentation size..."
    )
    hr_labels_scaled = _scale_labels_majority_vote(
        hr_segm, subject_labels, scaling_factor
    )

    # Correct vertices by co-registering lower left posterior and upper right
    # anterior corners between scaled volume and mesh.
    non_zero_coords_50 = np.argwhere(hr_vol_scaled > 50)
    rr = _align_mesh_to_volume(rr, scaling_factor, target_coords=non_zero_coords_50)

    logger.info("Done setting up adaptation to subject.")

    # Register labels of high resolution atlas to labels of subject's segmentation.
    subj_label_ants = ants.from_numpy(subject_labels.astype(float))
    hr_label_ants = ants.from_numpy(hr_labels_scaled.astype(float))
    logger.info("Fitting labels... ")
    # Compute or get cached registration.
    reg = _get_registration(
        fixed=subj_label_ants,
        moving=hr_label_ants,
        type_of_transform="SyNCC",
        reg_cache_dir=registration_cache_dir,
        reg_fname_prefix=f"{subject}_labels_",
    )

    # Apply the registration to the mesh vertices.
    # NOTE: Intentionally using invtransforms to warp the mesh to the subject space.
    warped_rr = np.array(
        apply_transforms_to_points(3, _coords_to_dataframe(rr), reg["invtransforms"])
    )

    logger.info("Fitting contrast... ")
    subj_ants = convert_to_ants_image(subj_contrast, normalize=True)
    hr_rs_ants = convert_to_ants_image(hr_volume_resampled, normalize=True)
    # Also warp the volume template to the subject space.
    hr_rs_ants = apply_transforms(
        fixed=subj_ants, moving=hr_rs_ants, transformlist=reg["fwdtransforms"]
    )
    # Register the warped atlas volume to the subject volume to refine the registration.
    reg = _get_registration(
        fixed=subj_ants,
        moving=hr_rs_ants,
        type_of_transform="SyNCC",
        reg_cache_dir=registration_cache_dir,
        reg_fname_prefix=f"{subject}_contrast_",
    )

    # Apply the refined registration to both volume and mesh.
    rr_double_warped = np.array(
        apply_transforms_to_points(
            3, _coords_to_dataframe(warped_rr), reg["invtransforms"]
        )
    )
    # Go from bounding box coordinates back to subject voxel coordinates.
    rr_final = rr_double_warped + cb_range[0]
    # Convert to FreeSurfer surface RAS coordinates.
    rr_ras = _convert_to_surface_ras(rr_final)

    if save_mesh:
        write_geometry(mesh_fname, rr_ras, tris)

    return rr_ras, tris


def _coords_to_dataframe(rr) -> "DataFrame":
    """Convert (N, 3) array of coords to a pandas DataFrame 'x', 'y', 'z'."""
    import pandas as pd

    rr_dictionary = {"x": list(rr[:, 0]), "y": list(rr[:, 1]), "z": list(rr[:, 2])}
    return pd.DataFrame(data=rr_dictionary)


def _align_mesh_to_volume(
    rr: NDArray, scaling_factor: NDArray, target_coords: NDArray
) -> NDArray:
    """Scale and spatially translates surface mesh to align with a target bounding box.

    Parameters
    ----------
    rr : NDArray
        The N x 3 array of mesh vertices in the source voxel space.
    scaling_factor : NDArray
        The scaling factors for the X, Y, and Z axes (3 elements).
    target_coords : NDArray
        The coordinates of the target volume (e.g., high-intensity voxels) used
        to calculate the target bounding box.

    Returns
    -------
    NDArray
        The scaled and spatially shifted N x 3 mesh vertices.
    """
    rr_scaled = rr * scaling_factor

    # Correct vertices by co-registering lower left posterior and upper right
    # anterior corners.
    target_min_coords = np.min(target_coords, axis=0)
    target_max_coords = np.max(target_coords, axis=0)
    mesh_min = np.min(rr_scaled, axis=0)
    mesh_max = np.max(rr_scaled, axis=0)

    translation_vector = np.mean(
        [
            target_min_coords - mesh_min,
            target_max_coords - mesh_max,
        ],
        axis=0,
    )

    return rr_scaled + translation_vector


def _scale_labels_majority_vote(
    hr_segm: NDArray, subj_segm: NDArray, scaling_factor: NDArray
) -> NDArray:
    """Scale the labels from the high-resolution segmentation to subject's segmentation.

    NOTE: This function could use some optimization. The custom logic could possibly be
    replaced with, for example, `scipy.ndimage.zoom`.

    Parameters
    ----------
    hr_segm : NDArray
        High-resolution segmentation volume as a 3D NumPy array.
    subj_segm : NDArray
        Subject's segmentation volume as a 3D NumPy array.
    scaling_factor : NDArray
        Scaling factor for each axis as a 1D NumPy array of length 3.

    Returns
    -------
    hr_label_scaled : NDArray
        Scaled labels volume as a 3D NumPy array.
    """
    # scale labels matrix (by type value vote)
    hr_label_scaled = np.zeros(
        (
            subj_segm.shape[0],
            subj_segm.shape[1],
            subj_segm.shape[2],
        )
    )
    count_matrix = np.zeros(
        (
            subj_segm.shape[0],
            subj_segm.shape[1],
            subj_segm.shape[2],
            100,
        )
    )
    count_matrix[:] = np.nan
    for x in range(hr_segm.shape[0]):
        for y in range(hr_segm.shape[1]):
            for z in range(hr_segm.shape[2]):
                target_vox = (scaling_factor * (x, y, z)).astype(int)
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
    for x in range(subj_segm.shape[0]):
        for y in range(subj_segm.shape[1]):
            for z in range(subj_segm.shape[2]):
                votes = count_matrix[x, y, z, :]
                votes = votes[~np.isnan(votes)]
                hr_label_scaled[x, y, z] = np.bincount(votes.astype(int)).argmax()

    return hr_label_scaled


def _crop_image_volume(
    vol: NDArray, coords: tuple[NDArray, ...], pad: int = 3
) -> tuple[NDArray, list]:
    """Crop a 3D volume to the bounding box of given coordinates, with optional padding.

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
    vol_cropped : NDArray
        The cropped volume.
    coords_range : list
        List of two lists containing the minimum and maximum coordinates of the cropped
        volume [[min_x, min_y, min_z], [max_x, max_y, max_z]].
    """
    # coords_range is [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    coords_range = [
        [max(0, np.min(coords[x]) - pad) for x in range(3)],
        [min(vol.shape[x], np.max(coords[x]) + pad + 1) for x in range(3)],
    ]
    vol_cropped = vol[
        coords_range[0][0] : coords_range[1][0],
        coords_range[0][1] : coords_range[1][1],
        coords_range[0][2] : coords_range[1][2],
    ]
    return vol_cropped, coords_range


def _convert_to_surface_ras(rr: NDArray) -> NDArray:
    """Convert surface geometry to FreeSurfer surface RAS coordinates.

    Surface RAS is the FreeSurfer coordinate frame, making the output compatible with
    FreeSurfer tools like freeview.

    Parameters
    ----------
    rr : NDArray
        N x 3 Array of vertex positions in the FreeSurfer voxel space.

    Returns
    -------
    NDArray
        N x 3 Array of vertex positions in FreeSurfer surface RAS coordinates.
    """
    rotation = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
    translation = np.array([128, -128, 128])
    ras = rr @ rotation + translation

    return ras


def _get_registration(
    fixed: "ANTsImage",
    moving: "ANTsImage",
    type_of_transform: str = "SyNCC",
    reg_cache_dir: str | None = None,
    reg_fname_prefix: str = "",
) -> dict:
    """Get or compute the registration between two images.

    Parameters
    ----------
    fixed : ANTsImage
        The fixed image for registration.
    moving : ANTsImage
        The moving image for registration.
    type_of_transform : str
        The type of transform to use for registration. Default is "SyNCC".
    reg_cache_dir : str | None
        Directory to cache registration results. If None (default), registration will
        be computed without caching.
    reg_fname_prefix : str
        Prefix for the registration output filenames. For example, if reg_fname_prefix
        is 'subject1_', the forward transform file will be saved as
        'subject1_Composite.h5' and the inverse as
        'subject1_InverseComposite.h5'.

    Returns
    -------
    dict
        Dictionary containing file paths for forward and inverse transforms,
        as returned by ANTs registration. Keys are 'fwdtransforms' and 'invtransforms'.
    """
    from ants.registration import registration

    if reg_cache_dir is None:
        # Perform registration without caching.
        logger.info("Performing registration...")
        reg = registration(
            fixed=fixed, moving=moving, type_of_transform=type_of_transform
        )
        logger.info("Registration complete.")
        return reg
    os.makedirs(reg_cache_dir, exist_ok=True)

    # Cache consists of forward and inverse tranforms.
    reg_forward_fname = f"{reg_fname_prefix}Composite.h5"
    reg_inverse_fname = f"{reg_fname_prefix}InverseComposite.h5"
    reg_forward_path = op.join(reg_cache_dir, reg_forward_fname)
    reg_inverse_path = op.join(reg_cache_dir, reg_inverse_fname)

    if op.exists(reg_forward_path) and op.exists(reg_inverse_path):
        logger.info(
            "Using existing registration transforms:\n  %s\n  %s",
            reg_forward_path,
            reg_inverse_path,
        )
        reg = {
            "fwdtransforms": [reg_forward_path],
            "invtransforms": [reg_inverse_path],
        }
        return reg

    # Perform registration and save the transforms to the cache directory.

    # ANTs will automatically append 'Composite.h5' and 'InverseComposite.h5'
    transform_fname_prefix = op.join(reg_cache_dir, reg_fname_prefix)
    logger.info(
        "Performing registration and saving transforms to cache:\n  %s\n  %s",
        reg_forward_path,
        reg_inverse_path,
    )
    reg = registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
        outprefix=transform_fname_prefix,  # save transforms
        write_composite_transform=True,  # combine warp and affine to a single file
    )
    logger.info("Registration complete. Transforms saved to cache.")
    return reg


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
        logger.debug(
            "solid_angle at the point of observation estimated to be %f", solid_angle
        )

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
        logger.debug("number of nan normals that have been smoothed = %d", count)
        logger.debug("Remaining NAN normals = %d", len(nan_vertices))
        logger.debug("Total surface area: %f", area)

    return (nn, area, area_list, nan_vertices)


def setup_full_source_space(
    subject: str,
    subjects_dir: str | None = None,
    cerebellum_surf_fname: str | None = None,
    spacing: str | int = "oct6",
) -> mne.SourceSpaces:
    """Set up a full surface source space that includes the cerebellum.

    The first element in the returned `SourceSpaces` list is the combined cerebral
    hemispheric source space and the second element is the cerebellar source space.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    subjects_dir : str | None
        The path to the directory containing the FreeSurfer subjects reconstructions.
        If None, defaults to the SUBJECTS_DIR environment variable.
    cerebellum_surf_fname : str | None
        Path to the cerebellum surface mesh file. If None, defaults to
        ``<subjects_dir>/<subject>/surf/cerebellum.white``
    spacing : str | int
        The spacing parameter for ``mne.setup_source_space``.

    Returns
    -------
    src_whole: mne.SourceSpaces
        List containing two source space elements: the cerebral cortex and the
        cerebellar cortex.
    """
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)  # pyright: ignore[reportAssignmentType]
    # Cast Path object to string for compatibility.
    subjects_dir = str(subjects_dir)

    logger.info("Setting up cerebral source space for subject %s...", subject)
    src_cort = mne.setup_source_space(
        subject=subject,
        subjects_dir=subjects_dir,
        spacing=spacing,  # pyright: ignore[reportArgumentType],
        add_dist=False,
    )
    if spacing == "all":
        # use_tris is None when spacing is 'all', idk if it is intentional.
        src_cort[0]["use_tris"] = src_cort[0]["tris"]  # pyright: ignore
        src_cort[1]["use_tris"] = src_cort[1]["tris"]  # pyright: ignore

    # Read the geometry of the cerebellar surface mesh.
    if cerebellum_surf_fname is None:
        cerebellum_surf_fname = op.join(
            subjects_dir, subject, "surf", "cerebellum.white"
        )
    surface_data = mne.read_surface(cerebellum_surf_fname)

    cerb_rr = surface_data[0]
    assert isinstance(cerb_rr, np.ndarray), "Surface vertices should be a NumPy array."
    cerb_rr = cerb_rr / 1000.0  # Convert from mm to m!

    cerb_tris = surface_data[1]
    assert isinstance(cerb_tris, np.ndarray), (
        "Surface triangles should be a NumPy array."
    )

    # Calculate normals.
    logger.info("Calculating normals on deformed surface...")
    (nn, _, _, nan_vertices) = calculate_normals(cerb_rr, cerb_tris, print_info=False)
    logger.info("Done.")

    # Make a dictionary to hold the cerebellar source space data.
    cerb_subj_data = {
        "rr": cerb_rr,
        "tris": cerb_tris,
        "nn": nn,
        "nan_nn": nan_vertices,
    }

    logger.info("Concatenating cerebellar source space with cerebral source space...")

    src_whole = src_cort.copy()
    hemi_src = join_source_spaces(src_cort)
    src_whole[0] = hemi_src

    src_whole[1]["rr"] = cerb_rr
    src_whole[1]["tris"] = cerb_subj_data["tris"]
    src_whole[1]["nn"] = cerb_subj_data["nn"]
    src_whole[1]["ntri"] = src_whole[1]["tris"].shape[0]
    src_whole[1]["use_tris"] = cerb_subj_data["tris"]

    in_use = np.ones(cerb_rr.shape[0]).astype(int)
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
        warnings.warn(
            "Failed to concatenate use_tris, use_tris will be put to None. This means you will not be able to visualize"
            " the cortex in 3d but can still do all computational operations."
        )
        src_joined["use_tris"] = None
    src_joined["vertno"] = np.nonzero(src_joined["inuse"])[0]

    return src_joined
