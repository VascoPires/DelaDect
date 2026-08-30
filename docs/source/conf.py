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
copyright = '2026, Vasco D. C. Pires'
author = 'Vasco D. C. Pires'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_design',
]

autosummary_generate = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = project
html_static_path = ['_static']
html_logo = 'deladect_logo.svg'
html_show_sourcelink = False
html_copy_source = False

# Repository metadata powers the theme's "Edit this page" links.
html_context = {
    'github_user': 'VascoPires',
    'github_repo': 'DelaDect',
    'github_version': 'main',
    'doc_path': 'docs/source',
}

html_theme_options = {
    'github_url': 'https://github.com/VascoPires/DelaDect',
    'icon_links': [
        {
            'name': 'Launch on Binder',
            'url': 'https://mybinder.org/v2/gh/VascoPires/DelaDect/HEAD',
            'icon': 'fa-solid fa-rocket',
            'type': 'fontawesome',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/deladect/',
            'icon': 'fa-brands fa-python',
            'type': 'fontawesome',
        },
        {
            # TODO: replace with the DOI landing page once the paper is published,
            # e.g. 'https://doi.org/10.xxxx/xxxxxxx'.
            'name': 'Paper (DOI pending)',
            'url': 'https://github.com/VascoPires/DelaDect#citing-deladect',
            'icon': 'fa-solid fa-file-lines',
            'type': 'fontawesome',
        },
    ],
    'collapse_navigation': True,
    'use_edit_page_button': True,
    'navigation_with_keys': False,
    'show_navbar_depth': 1,
    'max_navbar_depth': 4,
    'show_prev_next': False,
}


def setup(app):
    """Register custom CSS for all supported Sphinx versions."""
    add_css = getattr(app, "add_css_file", None)
    if add_css is not None:
        add_css("custom.css")
    else:  # pragma: no cover - compatibility with very old Sphinx
        app.add_stylesheet("custom.css")
