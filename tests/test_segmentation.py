import os
import os.path as op
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
from mne.datasets import sample
from numpy.testing import assert_allclose, assert_array_equal

from cmb.segmentation import get_segmentation


def test_segmentation_with_cache(tmp_path: Path) -> None:
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
    true_segmentation_nifti = nib.Nifti1Image.from_filename(model_segmentation_path)

    # Assert that segmentation arrays match exactly.
    assert_array_equal(
        np.asanyarray(segmentation_nifti.dataobj),
        np.asanyarray(true_segmentation_nifti.dataobj),
        err_msg="Segmentation does not match reference segmentation.",
    )
    # Assert that affines are close.
    segmentation_affine = segmentation_nifti.affine
    true_affine = true_segmentation_nifti.affine
    assert segmentation_affine is not None, "Segmentation affine is None."
    assert true_affine is not None, "Reference affine is None."
    assert segmentation_affine.shape == (4, 4), "Segmentation affine is not 4x4."
    assert_allclose(
        segmentation_affine,
        true_affine,
        err_msg="Segmentation affine does not match reference affine.",
    )


def test_segmentation_with_mock_data_and_mock_predictions(
    tmp_path, monkeypatch
) -> None:
    """Test segmentation with mock data and a mocked nnU-Net prediction function."""
    rng = np.random.default_rng(seed=42)  # For reproducibility
    # Mock the nnU-Net prediction function.
    mock_nnunet = MagicMock(side_effect=_fake_nnunet_prediction)
    monkeypatch.setattr("cmb.segmentation._run_nnunet_prediction", mock_nnunet)

    # Setup temporary directory structure.
    subjects_dir = tmp_path / "subjects"
    subject = "dummy_sub"
    mri_dir = subjects_dir / subject / "mri"
    mri_dir.mkdir(parents=True)

    cmb_path = tmp_path / "cmb"
    template_dir = cmb_path / "data"
    template_dir.mkdir(parents=True)

    # Create dummy MRI.
    affine = np.eye(4)
    tiny_brain = rng.random((20, 20, 20), dtype=np.float32)
    nib.save(nib.Nifti1Image(tiny_brain, affine), mri_dir / "brain.mgz")

    # Create dummy FreeSurfer segmentation (aseg.mgz) with two labels.
    tiny_aseg = np.zeros((20, 20, 20), dtype=np.uint8)
    tiny_aseg[5:15, 5:10, 5:15] = 7  # Fake Left Hemisphere seed
    tiny_aseg[5:15, 10:15, 5:15] = 46  # Fake Right Hemisphere seed
    nib.save(nib.Nifti1Image(tiny_aseg, affine), mri_dir / "aseg.mgz")

    # Use the same tiny_brain as the template brain.
    nib.save(nib.Nifti1Image(tiny_brain, affine), template_dir / "brain.nii")

    segmentation_nifti = get_segmentation(
        subjects_dir=str(subjects_dir),
        subject=subject,
        cmb_path=str(cmb_path),
        debug_mode=False,
    )
    assert segmentation_nifti is not None
    assert segmentation_nifti.shape == (20, 20, 20)
    assert segmentation_nifti.get_data_dtype() == np.uint8

    affine = segmentation_nifti.affine
    assert affine is not None, "Segmentation affine is None."
    assert affine.shape == (4, 4), "Segmentation affine is not 4x4."
    assert np.allclose(affine, np.eye(4)), "Segmentation affine is not identity."

    # Verify the mock intercepted exactly 4 nnUNet tasks (Mask, LH, RH, Lobe Refine)
    assert mock_nnunet.call_count == 4


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


def _fake_nnunet_prediction(
    model_folder: str, input_folder: str, output_folder: str
) -> None:
    """Mock nnU-Net prediction function for testing purposes.

    Generates dummy NIfTI files based on the input shapes.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith("_0000.nii.gz"):
            # Extract base subject name (e.g., 'sample_0000.nii.gz' -> 'sample').
            subject = file.replace("_0000.nii.gz", "")

            # Load the input to get the exact spatial dimensions and affine.
            in_path = op.join(input_folder, file)
            img = nib.Nifti1Image.from_filename(in_path)

            # Create a dummy mask prediction
            dummy_data = np.ones(img.shape, dtype=np.uint8)

            # Inject a block of label '2' for `_extract_lob_I_IV` function.`
            dummy_data[5:15, 5:15, 5:15] = 2

            # Save the fake prediction to the output folder
            out_path = op.join(output_folder, f"{subject}.nii.gz")
            nib.save(nib.Nifti1Image(dummy_data, img.affine), out_path)
