import sys
import pathlib as pl
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
    'sphinx.ext.todo',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx.ext.inheritance_diagram',
]

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ".rst"
html_theme = 'pydata_sphinx_theme'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
