"""Helper functions for tests."""

import zipfile
from pathlib import Path


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
