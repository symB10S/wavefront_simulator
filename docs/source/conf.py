import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'wavefronts'
copyright = '2023, Joanthan Meerholz'
author = 'Joanthan Meerholz'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',]

templates_path = ['_templates']
exclude_patterns = []
autodoc_member_order = 'bysource'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
