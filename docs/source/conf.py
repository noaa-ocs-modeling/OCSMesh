# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
from os import PathLike
from pathlib import Path
import sys
import importlib.metadata
from datetime import datetime

from setuptools import config


def repository_root(path: PathLike = None) -> Path:
    if path is None:
        path = __file__
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_file():
        path = path.parent
    if '.git' in (child.name for child in path.iterdir()) or path == path.parent:
        return path
    else:
        return repository_root(path.parent)


sys.path.insert(0, str(repository_root()))

# -- Project information -----------------------------------------------------
metadata = importlib.metadata.metadata('ocsmesh')

project = metadata['Name']
author = metadata['Author-email']
release = metadata['Version']
print(list(metadata))
for i in metadata:
    print(i, metadata[i])
copyright = f'{datetime.now().year}, Office of Coast Survey (OCS), National Oceanic and Atmospheric Administration (NOAA)'


# -- General configuration ---------------------------------------------------

autoclass_content = 'both'  # include both class docstring and __init__
autodoc_default_options = {
    # Make sure that any autodoc declarations show the right members
    'members': True,
    'inherited-members': True,
    'private-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autosummary_generate = True  # Make _autosummary files and include them
napoleon_numpy_docstring = False  # Force consistency, leave only Google
napoleon_use_rtype = False  # More legible

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Need the autodoc and autosummary packages to generate our docs.
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # The Napoleon extension allows for nicer argument formatting.
#    'sphinx.ext.napoleon',
    'm2r2',
    'numpydoc',
    'sphinxarg.ext',
    'sphinx.ext.githubpages'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
source_suffix = ['.rst', '.md']
