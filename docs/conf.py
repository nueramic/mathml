import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import nueramic_mathml

# -- Project information -----------------------------------------------------

project = 'Mathematics in ML'
copyright = '2022, Victor Barbarich, Adelina Tsoi'
author = 'Victor Barbarich, Adelina Tsoi'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx.ext.graphviz'
]
copybutton_prompt_text = r'>>> |\.\.\. |\$ |'
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    "light_logo": "nueramic-logo-cropped-black.svg",
    "dark_logo": "nueramic-logo-cropped-white.svg",
}
html_favicon = '_static/nueramic-logo-cropped-black.svg'

html_title = 'Nueramic. MathML'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


add_module_names = False
autodoc_typehints = "description"
autodoc_class_signature = "separated"
latex_elements = {'extraclassoptions': 'openany,oneside',
                  'extrapackages': r'\usepackage{tikz}'
                                   r'\usetikzlibrary{shapes,positioning}'
                                   r'\usepackage{amsmath}'}

math_number_all = True
math_numfig = False
latex_use_xindy = False

# -- GraphViz configuration ----------------------------------
graphviz_output_format = 'svg'