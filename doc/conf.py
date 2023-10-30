# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_bootstrap_theme

project = 'Cere-MEG-Bellum'
copyright = '2023, John Samuelsson'
author = 'John Samuelsson'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True

html_theme = 'bootstrap'

html_static_path = ['_static']

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("GitHub", "https://github.com/jasmainak/cbm", True)
    ],
    'bootswatch_theme': "yeti"
}
