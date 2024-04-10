import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tuduam'
author = 'Damien Keijzer, Saullo Castro'
copyright = '2024-, Damien Keijzer,, Saullo G. P. Castro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.doctest',
    'sphinx-pydantic',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx.ext.inheritance_diagram',
]

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")))


templates_path = ['_templates']
exclude_patterns = []
source_suffix = ".rst"
html_theme = 'pydata_sphinx_theme'
autosummary_generate = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
