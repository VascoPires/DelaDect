# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DelaDect'
copyright = '2025, Vasco D. C. Pires'
author = 'Vasco D. C. Pires'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'deladect_logo.svg'
html_theme_options = {
    'logo_only': True
}


def setup(app):
    """Register custom CSS for all supported Sphinx versions."""
    add_css = getattr(app, "add_css_file", None)
    if add_css is not None:
        add_css("custom.css")
    else:  # pragma: no cover - compatibility with very old Sphinx
        app.add_stylesheet("custom.css")
