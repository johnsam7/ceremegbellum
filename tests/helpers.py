"""Helper functions for tests."""

import zipfile
from pathlib import Path

from nibabel import Nifti1Image
from numpy.testing import assert_allclose


def set_up_cmb_data(zipped_data_path: Path, cmb_path: Path) -> None:
    """Set up the CMB data directory by extracting the contents of a zipped file.

    Parameters
    ----------
    zipped_data_path : Path
        Path to the zipped CMB data. Should contain all the files relevant for the
        current test and follow the directory structure of CMB data directory.
    cmb_path : Path
        Directory to which the zipped data should be extracted. Same path as the
        `cmb_path` argument used in the function under test.
    """
    # Make sure the output directory exists.
    cmb_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zipped_data_path, "r") as zf:
        zf.extractall(cmb_path)


def assert_niftis_equal(
    img1: Nifti1Image,
    img2: Nifti1Image,
    check_header: bool = True,
    tolerance: float = 1e-5,
) -> None:
    """Assert that two NIfTI images are equal.

    Compares the affine matrices, data arrays, and optionally the headers of two NIfTI
    images.

    Parameters
    ----------
    img1 : Nifti1Image
        The first NIfTI image to compare.
    img2 : Nifti1Image
        The second NIfTI image to compare.
    check_header : bool, optional
        If True, also compare the headers of the two images. Default is True.
    tolerance : float, optional
        The absolute tolerance for comparing the affine matrices and data arrays.
        Default is 1e-5.
    """
    affine_1 = img1.affine
    affine_2 = img2.affine
    assert affine_1 is not None and affine_2 is not None
    assert_allclose(
        affine_1,
        affine_2,
        atol=tolerance,
        err_msg="NIfTI affine matrices do not match.",
    )
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()
    assert_allclose(
        data1,
        data2,
        atol=tolerance,
        equal_nan=True,
        err_msg="NIfTI image data arrays do not match.",
    )
    if check_header:
        assert img1.header == img2.header, "NIfTI headers do not match."
