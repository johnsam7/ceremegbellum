"""Cere-MEG-Bellum (CMB)."""

import os as _os

from ._version import __version__
from .functions import get_cerebellum_data, get_plot_data_from_stc
from .segmentation import segment_cerebellum
from .source_space import create_cerebellar_surface, setup_full_source_space

# Deprecated, will just instruct users to use the new plotting functions instead.
from .visualization import plot_cerebellum_data

# Default data directory: <package_dir>/ (atlas data goes into data/, nnUNet/, etc.)
CMB_DATA_DIR = _os.path.join(_os.path.dirname(__file__), "")


__all__ = [
    "CMB_DATA_DIR",
    "get_cerebellum_data",
    "get_plot_data_from_stc",
    "segment_cerebellum",
    "create_cerebellar_surface",
    "setup_full_source_space",
    "plot_cerebellum_data",
    "__version__",
]
