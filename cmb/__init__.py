"""Cere-MEG-Bellum (CMB)"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

from ._version import __version__

from .functions import (get_cerebellum_data)
from .source_space import (setup_full_source_space)
from .visualization import (plot_cerebellum_data)

from .test import is_float