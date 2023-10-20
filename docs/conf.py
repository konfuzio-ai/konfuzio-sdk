"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to Document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation document_folder, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme
import os
import sys

print(os.path.abspath('../konfuzio_sdk'))
sys.path.insert(0, os.path.abspath('../konfuzio_sdk'))

# Path for custom extension (jupyter notebook validation)
sys.path.append(os.path.abspath('sphinx_custom_extensions'))

# -- Project information -----------------------------------------------------


project = 'Konfuzio'
copyright = '2023, Helm und Nagel GmbH'
author = 'Helm und Nagel GmbH'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'm2r2',
    'sphinx.ext.githubpages',
    'sphinx_toolbox.collapse',
    'sphinx_sitemap',
    'sphinxcontrib.mermaid',
    'notfound.extension',
    "sphinx_copybutton",
    'myst_nb',
    'validate_nb'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'layout.html']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
##
html_show_sourcelink = True
source_suffix = ['.rst', '.md']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sdk/tutorials/*.ipynb']

# make sure that make html starts with the index.rst
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- HTML -------------------------------------------------------------------
html_logo = '_static/docs__static_square_transparent_super_small.png'
html_favicon = '_static/full_green_square.png'
html_css_files = ['custom.css']
html_show_sphinx = False
html_baseurl = 'https://dev.konfuzio.com'
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


sitemap_url_scheme = "{link}"

# sphinx-notfound-page
notfound_urls_prefix = '/'

# MyST-NB

# No execution timeout
nb_execution_timeout = -1

# Raise an exception on failed execution, rather than emitting a warning
nb_execution_raise_on_error = True

# Print traceback to stderr on execution error
nb_execution_show_tb = True