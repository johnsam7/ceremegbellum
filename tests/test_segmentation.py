import os
import os.path as op
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest
from mne.datasets import sample
from pytest import MonkeyPatch

from cmb.segmentation import get_segmentation

from .helpers import assert_niftis_equal, set_up_cmb_data


@pytest.mark.requires_data
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
    set_up_cmb_data(test_cmb_data, cmb_path)

    segmentation_nifti = get_segmentation(
        subject,
        subjects_dir,
        cmb_path,
        save_segmentation=False,
        intermediate_caching=True,  # use cached predictions and registration
        recompute=True,
    )
    true_segmentation_nifti = nib.Nifti1Image.from_filename(model_segmentation_path)
    assert_niftis_equal(
        segmentation_nifti, true_segmentation_nifti, check_header=False, tolerance=1e-5
    )


def test_segmentation_with_mock_data_and_mock_predictions(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Test segmentation with mock data and a mocked nnU-Net prediction function."""
    rng = np.random.default_rng(seed=42)  # For reproducibility
    # Mock the nnU-Net prediction function.
    mock_nnunet = MagicMock(side_effect=_fake_nnunet_prediction)
    monkeypatch.setattr("cmb.segmentation._run_nnunet_prediction", mock_nnunet)

    subject = "dummy_sub"
    subjects_dir, cmb_dir = _create_mock_data(tmp_path, rng, subject)

    segmentation_nifti = get_segmentation(
        subjects_dir=subjects_dir,
        subject=subject,
        cmb_dir=cmb_dir,
        save_segmentation=True,
        segmentation_fname=None,  # Use default location
        recompute=True,
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

    # Verify that the segmentation was also saved to the expected output path.
    saved_segmentation_fname = (
        subjects_dir / subject / "mri" / "cerebellum_segmentation.nii.gz"
    )
    assert saved_segmentation_fname.exists(), "Segmentation file was not saved."

    saved_segmentation_nifti = nib.Nifti1Image.from_filename(saved_segmentation_fname)
    assert_niftis_equal(
        segmentation_nifti, saved_segmentation_nifti, check_header=False, tolerance=1e-5
    )


def test_segmentation_save_and_load(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that segmentation can be saved and loaded correctly.

    Tests that the segmentation is saved to disk, can be loaded from disk, and that
    recomputing the segmentation overwrites the saved file.
    """
    rng = np.random.default_rng(seed=42)
    # Mock the nnU-Net prediction function.
    mock_nnunet = MagicMock(side_effect=_fake_nnunet_prediction)
    monkeypatch.setattr("cmb.segmentation._run_nnunet_prediction", mock_nnunet)

    subject = "dummy_sub"
    subjects_dir, cmb_dir = _create_mock_data(tmp_path, rng, subject)

    # Run segmentation and save to the default location.
    segmentation_nifti = get_segmentation(subject, subjects_dir, cmb_dir)
    # Check that the segmentation was saved correctly.
    save_file = subjects_dir / subject / "mri" / "cerebellum_segmentation.nii.gz"
    saved_segmentation_nifti = nib.Nifti1Image.from_filename(save_file)
    assert_niftis_equal(
        segmentation_nifti, saved_segmentation_nifti, check_header=False, tolerance=1e-5
    )

    # Test that the segmentation can be loaded from the saved file.
    # Poison the mock to ensure that the segmentation is not recomputed.
    mock_nnunet.reset_mock()
    mock_nnunet.side_effect = AssertionError(
        "nnU-Net prediction should not be called when loading from cache."
    )
    loaded_segmentation_nifti = get_segmentation(subject, subjects_dir, cmb_dir)
    assert_niftis_equal(
        saved_segmentation_nifti,
        loaded_segmentation_nifti,
        check_header=False,
        tolerance=1e-5,
    )

    # Now test that if we set recompute=True, the segmentation is recomputed and
    # overwritten on disk.
    mock_nnunet.reset_mock()  # Reset the mock to allow it to be called again
    mock_nnunet.side_effect = _fake_nnunet_prediction
    # Poison the saved segmentation to ensure that the file is actually overwritten.
    poisoned_data = rng.integers(0, 255, size=(20, 20, 20), dtype=np.uint8)
    nib.save(nib.Nifti1Image(poisoned_data, np.eye(4)), save_file)
    recomputed_segmentation_nifti = get_segmentation(
        subject, subjects_dir, cmb_dir, recompute=True
    )
    # Check that the mock was called, indicating that the segmentation was recomputed.
    assert mock_nnunet.call_count > 0
    recomputed_saved_segmentation_nifti = nib.Nifti1Image.from_filename(save_file)
    assert_niftis_equal(
        recomputed_segmentation_nifti,
        recomputed_saved_segmentation_nifti,
        check_header=False,
        tolerance=1e-5,
    )

    # Test saving to a custom filename.
    custom_fname = tmp_path / "custom_segmentation.nii.gz"
    segmentation_nifti_custom = get_segmentation(
        subject, subjects_dir, cmb_dir, segmentation_fname=custom_fname
    )
    # Load the saved file and check that it matches the returned segmentation.
    mock_nnunet.reset_mock()  # Reset the mock to ensure no recomputation
    mock_nnunet.side_effect = AssertionError(
        "nnU-Net prediction should not be called when loading from cache."
    )
    loaded_custom_segmentation_nifti = get_segmentation(
        subject, subjects_dir, cmb_dir, segmentation_fname=custom_fname
    )
    assert_niftis_equal(
        segmentation_nifti_custom,
        loaded_custom_segmentation_nifti,
        check_header=True,
        tolerance=1e-5,
    )

    # Test no saving.
    mock_nnunet.reset_mock()  # Reset the mock to allow it to be called again
    mock_nnunet.side_effect = _fake_nnunet_prediction
    custom_fname_no_save = tmp_path / "no_save_segmentation.nii.gz"
    _ = get_segmentation(
        subject,
        subjects_dir,
        cmb_dir,
        segmentation_fname=custom_fname_no_save,
        save_segmentation=False,
    )
    assert not custom_fname_no_save.exists()

    # Test recompute=True but save_segmentation=False on an EXISTING file
    mock_nnunet.reset_mock()
    mock_nnunet.side_effect = _fake_nnunet_prediction
    # Poison the custom_fname file from the previous test block
    poisoned_data_2 = rng.integers(0, 255, size=(20, 20, 20), dtype=np.uint8)
    nib.save(nib.Nifti1Image(poisoned_data_2, np.eye(4)), custom_fname)
    recomputed_no_save_segmentation = get_segmentation(
        subject,
        subjects_dir,
        cmb_dir,
        segmentation_fname=custom_fname,
        recompute=True,
        save_segmentation=False,
    )
    # The mock should be called because we forced a recompute
    assert mock_nnunet.call_count > 0
    # But the file on disk should remain the poisoned version, not the new one
    file_on_disk = nib.Nifti1Image.from_filename(custom_fname)
    assert not np.array_equal(
        recomputed_no_save_segmentation.get_fdata(), file_on_disk.get_fdata()
    )


def _create_mock_data(
    tmp_path: Path, rng: np.random.Generator, subject: str
) -> tuple[Path, Path]:
    """Create directory structure and mock data expected by segmentation function."""
    # Setup temporary directory structure.
    subjects_dir = tmp_path / "subjects"
    mri_dir = subjects_dir / subject / "mri"
    mri_dir.mkdir(parents=True)

    cmb_dir = tmp_path / "cmb"
    template_dir = cmb_dir / "data"
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

    return subjects_dir, cmb_dir


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
