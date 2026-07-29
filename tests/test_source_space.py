import os.path as op
from pathlib import Path

import mne
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
    long and possibly nondeterministic registration).ö
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

    surface = create_cerebellar_surface(
        subject,
        subjects_dir,
        str(cmb_path),
        cerebellum_subsampling="sparse",
        print_fs=True,
        debug_mode=True,  # to use the cached registration
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
