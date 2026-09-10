import copy
import os
import shutil
from typing import Literal

import mne
import numpy as np
from numpy.typing import NDArray

from . import visualization as viz
from .source_space import _join_source_spaces


def get_cerebellum_data(cmb_dir: os.PathLike[str] | str | None = None) -> None:
    """Check if the required cerebellum data are available and download if not.

    Parameters
    ----------
    cmb_path : path-like | None, optional
        Path to the ceremegbellum data folder. If None, defaults to the
        package installation directory.
    """
    if cmb_dir is None:
        from . import CMB_DATA_DIR

        cmb_dir = CMB_DATA_DIR
    if (
        os.path.exists(os.path.join(cmb_dir, "data", "cerebellum_geo"))
        and os.path.isdir(
            os.path.join(
                cmb_dir,
                "nnUNet",
                "RESULTS_FOLDER",
                "nnUNet",
                "3d_fullres",
                "Task001_mask",
            )
        )
        and os.path.isdir(
            os.path.join(
                cmb_dir,
                "nnUNet",
                "RESULTS_FOLDER",
                "nnUNet",
                "3d_fullres",
                "Task002_lh",
            )
        )
        and os.path.isdir(
            os.path.join(
                cmb_dir,
                "nnUNet",
                "RESULTS_FOLDER",
                "nnUNet",
                "3d_fullres",
                "Task003_rh",
            )
        )
        and os.path.isdir(
            os.path.join(
                cmb_dir,
                "nnUNet",
                "RESULTS_FOLDER",
                "nnUNet",
                "3d_fullres",
                "Task004_refine_lobsI_IV",
            )
        )
        and os.path.exists(os.path.join(cmb_dir, "data", "brain.nii"))
    ):
        print("The required atlas data and segmentation models seem to be downloaded.")
    else:
        import zipfile

        _download_url = "https://osf.io/sdn9h/download"
        zip_path = os.path.join(cmb_dir, "tmp", "ceremegbellum.zip")
        print("Seems like some data are missing. No problem, fetching...")
        os.makedirs(os.path.join(cmb_dir, "tmp"), exist_ok=True)
        os.makedirs(os.path.join(cmb_dir, "data"), exist_ok=True)
        os.makedirs(
            os.path.join(cmb_dir, "nnUNet", "RESULTS_FOLDER", "nnUNet", "3d_fullres"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(cmb_dir, "nnUNet", "nnUNet_preprocessed"), exist_ok=True
        )
        os.makedirs(
            os.path.join(cmb_dir, "nnUNet", "nnUNet_raw_data_base"), exist_ok=True
        )

        if not os.path.exists(zip_path):
            try:
                from pooch import retrieve

                retrieve(
                    url=_download_url,
                    known_hash="sha256:1d07115e5d9d04c5b6b4e681f881a7c87a4b06fca07837226dcaf6746e286d54",
                    fname="ceremegbellum.zip",
                    path=os.path.join(cmb_dir, "tmp"),
                )
            except Exception as e:
                print(
                    f"\nDownload failed: {e}\n"
                    f"You can download the file manually from:\n"
                    f"  {_download_url}\n"
                    f"and save it to:\n"
                    f"  {zip_path}\n"
                    f"Then re-run get_cerebellum_data()."
                )
                return
        else:
            print("Found previously downloaded zip file. Skipping download.")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(cmb_dir, "tmp"))
        shutil.move(
            os.path.join(cmb_dir, "tmp", "osf_data", "cerebellum_geo"),
            os.path.join(cmb_dir, "data", "cerebellum_geo"),
        )
        shutil.move(
            os.path.join(cmb_dir, "tmp", "osf_data", "brain.nii"),
            os.path.join(cmb_dir, "data", "brain.nii"),
        )
        # Move all Task* directories
        tmp_osf = os.path.join(cmb_dir, "tmp", "osf_data")
        dest = os.path.join(cmb_dir, "nnUNet", "RESULTS_FOLDER", "nnUNet", "3d_fullres")
        for item in os.listdir(tmp_osf):
            if item.startswith("Task"):
                shutil.move(os.path.join(tmp_osf, item), os.path.join(dest, item))
        shutil.rmtree(os.path.join(cmb_dir, "tmp"), ignore_errors=True)  # clean up
        print("Done.")
    return


def get_plot_data_from_stc(
    stc: mne.SourceEstimate | mne.MixedSourceEstimate,
    fwd_src: mne.SourceSpaces,
    time_point: float,
    cerebellum_geo: dict,
    cerebellum_subsampling: Literal["sparse", "dense"],
    cerebellum_idx: int = 1,
    cortex_smooth: int | None | Literal["nearest"] = None,
    cerebellum_smooth: int = 0,
) -> tuple[NDArray, NDArray]:
    """Get the data of one time point for plotting from a SourceEstimate object.

    This works as a bridge between the `SourceEstimate`object and the visualization
    functions in this package. Supports both legacy CMB source spaces where cerebellum
    source space is the second element in the `SourceSpaces` list, and mixed source
    spaces where cerebellum source space is the third element in the `SourceSpaces`
    list. This function will be hopefully removed in the future when MNE-Python
    handles the plotting natively.

    Parameters
    ----------
    stc : mne.SourceEstimate | mne.MixedSourceEstimate
        The source estimate object containing the data to plot.
    fwd_src : mne.SourceSpaces
        The source spaces used in the forward model.
    time_point : float
        The time point to extract data for plotting.
    cerebellum_geo : dict
        The cerebellum geometry data loaded from the cerebellum_geo file.
    cerebellum_subsampling : Literal["sparse", "dense"]
        The subsampling used for the cerebellum source space.
    cerebellum_idx : int, optional
        The index of the cerebellum source space in the `SourceSpaces` list.
        Allowed values are 1 (legacy CMB source space) or 2 (mixed source space).
        Defaults to 1.
    cortex_smooth : int | None | Literal["nearest"]
        Passed for `mne.morph._hemi_morph`.
        Controls spatial interpolation smoothing. If an integer, applies exactly
        that many iterative averaging steps (0 leaves data at sparse vertices only).
        If ``None``, automatically iterates until all unmapped vertices are filled
        (capped at 100 steps). If ``"nearest"``, maps every vertex to the single
        closest source vertex without blending.
    cerebellum_smooth: int, optional
        Number of smoothing iterations to apply to the interpolated data. Each
        iteration averages the value at each vertex with its neighbors. By default 0,
        which means no smoothing is applied.

    Returns
    -------
    cortex_data : NDArray
        The data for each vertex in the cortical source space at the specified time
        point.
    cerebellum_data : NDArray
        The data for each vertex in the cerebellar source space at the specified time
        point.
    """
    if cerebellum_idx not in [1, 2]:
        raise ValueError(f"Invalid cerebellum index: {cerebellum_idx}. Must be 1 or 2.")
    if cerebellum_idx == 1:
        n_cortex_verts = fwd_src[0]["nuse"]
        fwd_cortex_src = fwd_src[0]
        fwd_cerebellum_src = fwd_src[1]
    else:
        n_cortex_verts = fwd_src[0]["nuse"] + fwd_src[1]["nuse"]
        # Concatenate the two cortical source spaces to be compatible
        # with the visualization functions.
        fwd_cortex_src = _join_source_spaces(fwd_src[:2])
        fwd_cerebellum_src = fwd_src[2]
    assert isinstance(fwd_cortex_src, dict)
    assert isinstance(fwd_cerebellum_src, dict)

    stc_data = stc.data
    assert isinstance(stc_data, np.ndarray)
    time_idx = stc.time_as_index(time_point)[0]

    cortex_data = stc_data[:n_cortex_verts, time_idx]
    cerebellum_data = stc_data[n_cortex_verts:, time_idx]

    cortex_data = viz.morph_cortex_data(
        cort_data=cortex_data, fwd_cortex_src=fwd_cortex_src, smooth=cortex_smooth
    )
    cerebellum_data = viz.morph_cerebellum_data(
        data=cerebellum_data,
        fwd_cerebellum_src=fwd_cerebellum_src,
        cerebellum_geo=cerebellum_geo,
        subsampling=cerebellum_subsampling,
        smoothing_steps=cerebellum_smooth,
    )

    return cortex_data, cerebellum_data


def get_subsampled_cerebellum_labels(
    cb_data: dict,
    subsampling: Literal["sparse", "dense"] = "sparse",
    set_hemi_cerebellum: bool = True,
) -> list[mne.Label]:
    """Get list of patch labels for subsampled cerebellum.

    The vertices of each label in the list correspond to the vertex indices
    in the subsampled cerebellum source space.

    Parameters
    ----------
    cb_data : dict
        The cerebellum geometry data loaded from the cerebellum_geo file.
    subsampling : "sparse" | "dense", optional
        The subsampling used for the cerebellum source space. Defaults to "sparse".
    set_hemi_cerebellum : bool, optional
        If True (default), sets the `hemi` attribute of each label to "cerebellum".
    """
    labels = cb_data["dw_data"][f"labels {subsampling}"]
    if set_hemi_cerebellum:
        labels = copy.deepcopy(labels)  # Avoid modifying original labels.
        for label in labels:
            label.hemi = "cerebellum"

    return labels
