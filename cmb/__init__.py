"""Cere-MEG-Bellum (CMB)."""

import os as _os

from ._version import __version__
from .functions import get_cerebellum_data
from .source_space import setup_full_source_space
from .utils import is_float

# Default data directory: <package_dir>/ (atlas data goes into data/, nnUNet/, etc.)
CMB_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "")


__all__ = [
    "CMB_DATA_DIR",
    "get_cerebellum_data",
    "setup_full_source_space",
    "is_float",
    "__version__",
]
