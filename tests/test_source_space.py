import os.path as op
import pickle
import sys
from pathlib import Path

# Need this to make sure that ants.registration is available for mocking in tests
import ants.registration  # ruff: ignore[F401]
import mne
import nibabel as nib
import numpy as np
import pytest
from mne.datasets import sample
from numpy.testing import assert_allclose, assert_array_equal
from pytest_mock import MockerFixture

from cmb.source_space import _join_source_spaces, create_cerebellar_surface

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

    rr, tris = create_cerebellar_surface(
        subject,
        subjects_dir,
        str(cmb_path),
        cerebellum_subsampling="sparse",
        save_mesh=mesh_fname,
        registration_caching=True,
    )
    true_rr, true_tris = mne.read_surface(model_surface_path)  # pyright: ignore[reportAssignmentType]
    assert isinstance(true_rr, np.ndarray)
    assert isinstance(true_tris, np.ndarray)

    assert_allclose(
        rr,
        true_rr,
        rtol=1e-5,
        atol=1e-4,
    )
    assert_array_equal(
        tris,
        true_tris,
    )
    # Check that mesh file was saved and is correct.
    assert op.exists(mesh_fname), "Mesh file was not saved."
    saved_rr, saved_tris = mne.read_surface(mesh_fname)  # pyright: ignore[reportAssignmentType]
    assert isinstance(saved_rr, np.ndarray)
    assert isinstance(saved_tris, np.ndarray)

    assert_allclose(
        saved_rr,
        true_rr,
        rtol=1e-5,
        atol=1e-4,
    )
    assert_array_equal(
        saved_tris,
        true_tris,
    )


def test_create_cerebellar_surface_with_mock_data(tmp_path: Path) -> None:
    """Test cerebellar mesh creation with mock data.

    Does not test the accuracy of the mesh, but checks that the function runs and
    produces a mesh file.
    """
    rng = np.random.default_rng(seed=42)  # For reproducibility
    subject = "mock_subject"

    subjects_dir, cmb_dir, verts, faces = _create_mock_data(tmp_path, rng, subject)

    rr, tris = create_cerebellar_surface(
        subject=subject,
        subjects_dir=str(subjects_dir),
        cmb_path=str(cmb_dir),
        cerebellum_subsampling="full",
        save_mesh=True,  # save to FreeSurfer default location
        registration_caching=False,
    )

    assert rr.shape == verts.shape
    assert tris.shape == faces.shape

    # Verify the saved mesh.
    mesh_fname = subjects_dir / subject / "surf" / "cerebellum.white"
    assert mesh_fname.exists(), "Mesh file was not saved to the expected location."
    saved_rr, saved_tris = mne.read_surface(mesh_fname)  # pyright: ignore[reportAssignmentType]
    assert isinstance(saved_rr, np.ndarray)
    assert isinstance(saved_tris, np.ndarray)

    assert_allclose(
        saved_rr,
        rr,
        rtol=1e-5,
        atol=1e-4,
    )
    assert_array_equal(
        saved_tris,
        tris,
    )


class TestRegistrationCache:
    """Test the registration caching mechanism in create_cerebellar_surface."""

    def test_without_caching(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that no cache is created when registration_caching=False."""
        rng = np.random.default_rng(seed=42)
        subject = "mock_subject"

        subjects_dir, cmb_dir, _, _ = _create_mock_data(tmp_path, rng, subject)
        cache_dir = cmb_dir / "data" / "atlas_fitting_cache"

        # Get ANTS registration function this way because of namespace
        # collision of ants.registration module and the ants.registration function.
        ants_reg_module = sys.modules["ants.registration"]
        ants_registration = mocker.spy(ants_reg_module, "registration")

        create_cerebellar_surface(
            subject=subject,
            subjects_dir=str(subjects_dir),
            cmb_path=str(cmb_dir),
            cerebellum_subsampling="full",
            save_mesh=False,
            registration_caching=False,
        )
        assert not cache_dir.exists() or not any(cache_dir.iterdir()), (
            "Cache directory should be empty or not exist when no caching is used."
        )
        assert ants_registration.call_count == 2, (
            "Expected 2 calls to ANTS registration (one for labels, one for contrast)"
        )

    def test_with_caching(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Test that cache is created and used when registration_caching=True."""
        rng = np.random.default_rng(seed=42)
        subject = "mock_subject"

        subjects_dir, cmb_dir, _, _ = _create_mock_data(tmp_path, rng, subject)
        cache_dir = cmb_dir / "data" / "atlas_fitting_cache"

        # Get ANTS registration function this way because of namespace
        # collision of ants.registration module and the ants.registration function.
        ants_reg_module = sys.modules["ants.registration"]
        ants_registration = mocker.spy(ants_reg_module, "registration")

        # First run: should create cache.
        create_cerebellar_surface(
            subject=subject,
            subjects_dir=str(subjects_dir),
            cmb_path=str(cmb_dir),
            cerebellum_subsampling="full",
            save_mesh=False,
            registration_caching=True,
        )
        assert cache_dir.exists() and any(cache_dir.iterdir()), (
            "Cache directory should exist and contain files after first run."
        )
        assert ants_registration.call_count == 2, (
            "Expected 2 calls to ANTS registration (one for labels, one for contrast)"
        )
        first_call_count = ants_registration.call_count

        # Second run: should use cache, so no additional calls to ANTS registration.
        create_cerebellar_surface(
            subject=subject,
            subjects_dir=str(subjects_dir),
            cmb_path=str(cmb_dir),
            cerebellum_subsampling="full",
            save_mesh=False,
            registration_caching=True,
        )
        second_call_count = ants_registration.call_count

        assert second_call_count == first_call_count, (
            "ANTS registration should not be called again when using cached transforms."
        )


@pytest.mark.requires_data
class TestSourceSpaceJoining:
    """Test the _join_source_spaces function for different 'use_tris' scenarios."""

    def test_source_space_joining_with_use_tris(self) -> None:
        """Test that _join_source_spaces correctly joins two source spaces.

        Covers the case where the 'use_tris' field is not None, which occurs
        when using spacing defined with a string.
        """
        # Use MNE sample subject.
        data_path = sample.data_path()
        subjects_dir = data_path / "subjects"
        subject = "sample"

        src_file = subjects_dir / subject / "bem" / "sample-oct-6-src.fif"
        src = mne.read_source_spaces(src_file, verbose=False)
        src_orig = src.copy()
        src_joined = _join_source_spaces(src)

        self._assert_correct_joining(src_orig, src_joined)
        # Check "use_tris" separately.
        offset = src_orig[0]["np"]
        expected_use_tris = np.concatenate(
            (src_orig[0]["use_tris"], src_orig[1]["use_tris"] + offset), axis=0
        )
        assert_array_equal(src_joined["use_tris"], expected_use_tris)

    def test_source_space_joining_without_use_tris(self) -> None:
        """Test that _join_source_spaces correctly joins two source spaces.

        Covers the case where the 'use_tris' field is None, which occurs
        when using spacing defined with an integer.
        """
        # Use MNE sample subject.
        data_path = sample.data_path()
        subjects_dir = data_path / "subjects"
        subject = "sample"

        src = mne.setup_source_space(
            subject,
            spacing=2,  # pyright: ignore[reportArgumentType]
            subjects_dir=subjects_dir,
            add_dist=False,
        )
        assert src[0]["use_tris"] is None
        assert src[1]["use_tris"] is None

        src_orig = src.copy()
        src_joined = _join_source_spaces(src)

        self._assert_correct_joining(src_orig, src_joined)
        # Check "use_tris" separately.
        assert src_joined["use_tris"] is None

    def _assert_correct_joining(
        self, src_orig: mne.SourceSpaces, src_joined: dict
    ) -> None:
        """Assert that joined source space has correct properties.

        Skips "use_tris" field.
        """
        # Assert scalar values are properly summed.
        assert src_joined["np"] == src_orig[0]["np"] + src_orig[1]["np"]
        assert src_joined["ntri"] == src_orig[0]["ntri"] + src_orig[1]["ntri"]
        assert src_joined["nuse"] == src_orig[0]["nuse"] + src_orig[1]["nuse"]
        assert (
            src_joined["nuse_tri"] == src_orig[0]["nuse_tri"] + src_orig[1]["nuse_tri"]
        )

        # Assert raw concatenations (no offsets).
        expected_inuse = np.concatenate((src_orig[0]["inuse"], src_orig[1]["inuse"]))
        assert_array_equal(src_joined["inuse"], expected_inuse)

        expected_nn = np.concatenate((src_orig[0]["nn"], src_orig[1]["nn"]), axis=0)
        assert_array_equal(src_joined["nn"], expected_nn)

        expected_rr = np.concatenate((src_orig[0]["rr"], src_orig[1]["rr"]), axis=0)
        assert_array_equal(src_joined["rr"], expected_rr)

        # Assert offset concatenations
        # Triangles from the right hemisphere must be offset by the total point count of
        # the left.
        offset = src_orig[0]["np"]

        expected_tris = np.concatenate(
            (src_orig[0]["tris"], src_orig[1]["tris"] + offset), axis=0
        )
        assert_array_equal(src_joined["tris"], expected_tris)

        # Assert dynamically generated vertno
        # The 'vertno' array should strictly contain the indices where 'inuse' == 1
        expected_vertno = np.nonzero(expected_inuse)[0]
        assert_array_equal(src_joined["vertno"], expected_vertno)


def _create_mock_data(
    tmp_path: Path, rng: np.random.Generator, subject: str
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """Create mock data suitable for testing cerebellar mesh creation.

    Subsampling must be set to 'full'.
    """
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
