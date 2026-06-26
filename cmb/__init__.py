"""Cere-MEG-Bellum (CMB)"""

import os as _os

from ._version import __version__

# Default data directory: <package_dir>/ (atlas data goes into data/, nnUNet/, etc.)
CMB_DATA_DIR = _os.path.join(_os.path.dirname(__file__), '')

from .functions import get_cerebellum_data
from .source_space import setup_full_source_space
from .visualization import plot_cerebellum_data
from .utils import is_float
