import os.path as op
import pickle
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest
from mne.datasets import sample
from numpy.testing import assert_allclose, assert_array_equal

from cmb.source_space import create_cerebellar_surface

from .helpers import set_up_cmb_data


@pytest.mark.requires_data
def test_create_cerebellar_surface_with_cache(tmp_path: Path) -> None:
    """Test cerebellar mesh creation for MNE sample subject against a reference mesh.

    Uses precomputed segmentation and cached registration transforms (to avoid running
    long and possibly nondeterministic registration).
    """
    # Use MNE sample subject.
    data_path = sample.data_path()
    subjects_dir = op.join(data_path, "subjects")
    subject = "sample"

    test_data_dir = Path(__file__).parent / "data"
    model_surface_path = test_data_dir / "sample_cerebellum.white"

    # Store cerebellum data in a temporary directory.
    cmb_path = tmp_path / "cerebellum_data"

    # Extract subject segmentation, registration cache and cerebellum atlas geometry
    # to the temporary CMB directory.
    test_cmb_data = test_data_dir / "sample_mesh_creation_cache.zip"
    set_up_cmb_data(test_cmb_data, cmb_path)

    mesh_fname = str(tmp_path / "cerebellum_mesh" / "output_mesh.surf")

    surface = create_cerebellar_surface(
        subject,
        subjects_dir,
        str(cmb_path),
        cerebellum_subsampling="sparse",
        save_mesh=True,
        mesh_fname=mesh_fname,
        registration_caching=True,
    )
    true_rr, true_tris = mne.read_surface(model_surface_path)  # pyright: ignore[reportAssignmentType]
    assert isinstance(true_rr, np.ndarray), "True vertices are not a numpy array."
    assert isinstance(true_tris, np.ndarray), "True triangles are not a numpy array."

    # Assert that the surface vertices and triangles match exactly.
    assert_allclose(
        surface["rr"],
        true_rr,
        rtol=1e-5,
        atol=1e-4,
        err_msg="Surface vertices do not match reference surface.",
    )
    assert_array_equal(
        surface["tris"],
        true_tris,
        err_msg="Surface triangles do not match reference surface.",
    )
    # Check that mesh file was saved and is correct.
    assert op.exists(mesh_fname), "Mesh file was not saved."
    saved_rr, saved_tris = mne.read_surface(mesh_fname)  # pyright: ignore[reportAssignmentType]
    assert isinstance(saved_rr, np.ndarray), "Saved vertices are not a numpy array."
    assert isinstance(saved_tris, np.ndarray), "Saved triangles are not a numpy array."

    assert_allclose(
        saved_rr,
        surface["rr"],
        rtol=1e-5,
        atol=1e-4,
        err_msg="Saved mesh vertices do not match returned surface vertices.",
    )
    assert_array_equal(
        saved_tris,
        surface["tris"],
        err_msg="Saved mesh triangles do not match returned surface triangles.",
    )


def test_create_cerebellar_surface_with_mock_data(tmp_path: Path) -> None:
    """Test cerebellar mesh creation with mock data.

    Does not test the accuracy of the mesh, but checks that the function runs and
    produces a mesh file.
    """
    rng = np.random.default_rng(seed=42)  # For reproducibility
    subject = "mock_subject"

    subjects_dir, cmb_dir, verts, faces = _create_mock_data(tmp_path, rng, subject)

    mesh_fname = tmp_path / "cerebellum_mesh" / "output_mesh.surf"

    surface = create_cerebellar_surface(
        subject=subject,
        subjects_dir=str(subjects_dir),
        cmb_path=str(cmb_dir),
        cerebellum_subsampling="full",
        save_mesh=True,
        mesh_fname=str(mesh_fname),
        registration_caching=False,
    )

    assert surface["rr"].shape == verts.shape
    assert surface["tris"].shape == faces.shape

    # Verify the saved mesh.
    assert mesh_fname.exists(), "Mesh file was not saved to the expected location."
    saved_rr, saved_tris = mne.read_surface(mesh_fname)  # pyright: ignore[reportAssignmentType]
    assert isinstance(saved_rr, np.ndarray)
    assert isinstance(saved_tris, np.ndarray)

    assert_allclose(
        saved_rr,
        surface["rr"],
        rtol=1e-5,
        atol=1e-4,
    )
    assert_array_equal(
        saved_tris,
        surface["tris"],
    )


def _create_mock_data(
    tmp_path: Path, rng: np.random.Generator, subject: str
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    # Setup mock MRI directory and save a fake MRI there.
    subjects_dir = tmp_path / "subjects"
    subj_mri_dir = subjects_dir / subject / "mri"
    subj_mri_dir.mkdir(parents=True)

    fake_mri_data = rng.random((20, 20, 20)) * 100
    affine = np.eye(4)  # identity affine
    nib.save(nib.Nifti1Image(fake_mri_data, affine), subj_mri_dir / "orig.mgz")

    # Setup mock CMB directory and save a fake segmentation and
    # cerebellum geometry file there.
    cmb_dir = tmp_path / "cmb"
    cmb_data_dir = cmb_dir / "data"
    segmentation_dir = cmb_data_dir / "segm_folder"
    segmentation_dir.mkdir(parents=True)
    # Segmentation
    fake_seg_data = np.zeros((20, 20, 20))
    fake_seg_data[5:15, 5:15, 5:15] = 1  # Fake cerebellum ROI
    nib.save(
        nib.Nifti1Image(fake_seg_data, affine),
        segmentation_dir / f"{subject}.nii.gz",
    )
    # Cerebellum geometry
    hr_vol = rng.random((30, 30, 30)) * 100  # virtual MRI volume
    hr_segm = np.zeros((30, 30, 30))  # template segmentation
    hr_segm[5:25, 5:25, 5:25] = 1  # Fake cerebellum ROI
    # A simple tetrahedron (4 vertices)
    verts = np.array([[5, 5, 5], [10, 5, 5], [7, 10, 5], [7, 7, 10]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    # NOTE: Tailored for 'full' subsampling.
    fake_geo = {
        "hr_vol": hr_vol,
        "parcellation": {"volume": hr_segm},
        "verts_normal": verts,
        "faces": faces,
    }
    with open(cmb_data_dir / "cerebellum_geo", "wb") as f:
        pickle.dump(fake_geo, f)

    return subjects_dir, cmb_dir, verts, faces
