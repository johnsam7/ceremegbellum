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
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
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
# Types are documented (in numpydoc form) in the docstrings themselves, so do not also
# render the annotations into the signatures. This matches MNE-Python.
autodoc_typehints = "none"

# -- numpydoc (NumPy-style docstrings) ----------------------------------------

numpydoc_show_class_members = False
numpydoc_xref_param_type = True  # link the types in "name : type" lines
numpydoc_xref_aliases = {
    "ndarray": "numpy.ndarray",
    "NDArray": "numpy.typing.NDArray",
    "DTypeLike": "numpy.typing.DTypeLike",
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "SourceSpaces": "mne.SourceSpaces",
    "SourceEstimate": "mne.SourceEstimate",
    "MixedSourceEstimate": "mne.MixedSourceEstimate",
    "Label": "mne.Label",
    "Figure": "matplotlib.figure.Figure",
    "Triangulation": "matplotlib.tri.Triangulation",
}
# Words in the type lines that are prose, not types.
numpydoc_xref_ignore = {
    "optional",
    "or",
    "of",
    "shape",
    "path-like",
    "n_x",
    "n_y",
    "n_z",
    "m_x",
    "m_y",
    "m_z",
    "n_vertices",
    "n_faces",
    "n_points",
    "n_known",
    "n_used",
    "...",
}

# -- sphinx-gallery ----------------------------------------------------------

# The examples need a GPU, ANTs and the MNE sample dataset, so they are rendered but
# never executed when building the docs.
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "plot_gallery": False,
    "filename_pattern": r"^$",
    "download_all_examples": False,
    "write_computation_times": False,
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
