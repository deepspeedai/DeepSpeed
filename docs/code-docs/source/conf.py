# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
import sys

# -- Project information -----------------------------------------------------

project = 'DeepSpeed'
copyright = '2020, Microsoft'
author = 'Microsoft'

# The full version, including alpha/beta/rc tags
with open("../../../version.txt", "r") as f:
    release = f.readline().rstrip()

master_doc = 'index'

autodoc_member_order = 'bysource'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_rtd_theme',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx.ext.autosectionlabel',
]

pygments_style = 'sphinx'

# autodoc_pyandtic config
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_field_signature_prefix = ' '
autodoc_pydantic_model_signature_prefix = 'class'
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_summary_list_order = 'bysource'
autodoc_pydantic_model_member_order = 'bysource'
autodoc_pydantic_field_list_validators = False

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

# GitHub integration
html_context = {
    "display_github": True,
    "github_user": "microsoft",
    "github_repo": "DeepSpeed",
    "github_version": "master",
    "conf_py_path": "/docs/code-docs/source/",
}

sys.path.insert(0, os.path.abspath('../../../'))

# Prepend module names to class descriptions?
add_module_names = True

autoclass_content = 'auto'

autodoc_mock_imports = ["apex", "mpi4py", "tensorboardX", "numpy", "cupy"]
