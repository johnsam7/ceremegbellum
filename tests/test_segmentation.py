import os.path as op
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from mne.datasets import sample
from numpy.testing import assert_array_equal

from cmb.segmentation import get_segmentation


def test_segmentation(tmp_path: Path) -> None:
    """Test the segmentation on MNE sample subject against a reference segmentation.

    Registration or nnU-Net predictions are not actually performed. Instead, precomputed
    registration and segmentations from a zip file are used. This happens automatically,
    because segmentation pipeline checks for the existence of cache files and uses them
    if they are present.
    """
    # Use MNE sample subject.
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, "subjects")
    subject = "sample"

    test_data_dir = Path(__file__).parent / "data"
    model_segmentation_path = test_data_dir / "sample_segmentation.nii.gz"

    # Extract template brain.nii, registration cache, and segmentation cache to
    # a temporary directory.
    test_cmb_data = test_data_dir / "sample_segmentation_cache.zip"
    cmb_path = tmp_path / "cmb"
    _set_up_cmb_data(test_cmb_data, output_dir=cmb_path)

    segmentation_nifti = get_segmentation(
        subjects_dir, subject, str(cmb_path), debug_mode=False
    )
    segmentation = np.asanyarray(segmentation_nifti.dataobj)
    true_segmentation = np.asanyarray(
        nib.Nifti1Image.from_filename(model_segmentation_path).dataobj
    )
    assert_array_equal(
        segmentation,
        true_segmentation,
        err_msg="Segmentation does not match reference segmentation.",
    )


def _set_up_cmb_data(zipped_data_path: Path, output_dir: Path) -> None:
    """Extract CMB data to given output directory.

    Parameters
    ----------
    zipped_data_path : str
        Path to the zipped CMB data.
    output_dir : str
        Directory where the data should be extracted.
    """
    with zipfile.ZipFile(zipped_data_path, "r") as zf:
        zf.extractall(output_dir)
