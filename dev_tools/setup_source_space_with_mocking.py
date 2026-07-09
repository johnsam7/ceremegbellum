"""Set up full source space including cerebellum and save it to a pickle file."""

import os.path as op
import pickle
from pathlib import Path
from unittest.mock import patch

from mne.datasets import sample

from cmb import setup_full_source_space

cerebellum_data_dir = Path("dir/here")
output_file = Path(__file__).parent / "reference_src.pkl"

# Use MNE sample subject.
data_path = sample.data_path()
subjects_dir = op.join(data_path, "subjects")
subject = "sample"


def dummy_ants_registration(*args, **kwargs):
    """Mock function to replace ANTs registration in tests.

    Replaces the ANTs SyNCC registration.
    Returns a fake dictionary that mimics what ANTs would output,
    but with empty transform lists so the coordinates don't drift.
    """
    return {
        "fwdtransforms": [],
        "invtransforms": [],
        "warpedmovout": kwargs.get("moving"),  # Just return the moving image unchanged
    }


@patch("source_space.ants.registration", new=dummy_ants_registration)
def setup_and_save_source_space() -> None:
    """Set up the source space and save it to a pickle file."""
    # Make sure that these match to the parameters used when doing the reference
    # source space creation.
    spacing = 2
    cerebellum_subsampling = "sparse"

    src = setup_full_source_space(
        subject,
        subjects_dir,
        cerebellum_data_dir,
        cerb_subsampling=cerebellum_subsampling,
        plot_cerebellum=False,
        spacing=spacing,
    )

    with open(output_file, "wb") as f:
        pickle.dump(src, f)


if __name__ == "__main__":
    setup_and_save_source_space()
