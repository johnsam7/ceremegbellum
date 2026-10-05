"""Sphinx configuration for the Cere-MEG-Bellum documentation."""

import datetime
import importlib.metadata
import os

# Optional Qt backend of the 3D plots must not need a display when building docs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# -- Project information -----------------------------------------------------

project = "Cere-MEG-Bellum"
author = "John G. W. Samuelsson, Christoph Dinh"
copyright = f"2021-{datetime.date.today().year}, authors of CMB"  # noqa: A001
release = importlib.metadata.version("cmb")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_heading_anchors = 3

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_typehints = "description"

# -- Napoleon (NumPy docstrings only) ----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "NDArray": "numpy.typing.NDArray",
    "npt.NDArray": "numpy.typing.NDArray",
    "np.ndarray": "numpy.ndarray",
    "np.float64": "numpy.float64",
    "np.floating": "numpy.floating",
    "np.intp": "numpy.intp",
    "np.uint8": "numpy.uint8",
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "SourceSpaces": "mne.SourceSpaces",
    "mne.SourceSpaces": "mne.SourceSpaces",
    "Figure": "matplotlib.figure.Figure",
    "Axes": "matplotlib.axes.Axes",
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "pyvista": ("https://docs.pyvista.org", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/johnsam7/ceremegbellum",
    "logo": {"text": "Cere-MEG-Bellum"},
    "navigation_depth": 4,
    "show_nav_level": 0,
    "show_toc_level": 2,
}
