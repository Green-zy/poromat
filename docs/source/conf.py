# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'poromat'
copyright = '2025, Yun Zhou'
author = 'Yun Zhou'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",            
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode"
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autoapi_dirs = ["../src"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'
html_theme_options = {
    "nav_title": " ",  
    "base_url": "",  
    "color_primary": "blue-grey",
    "color_accent": "cyan",
    "repo_url": "https://github.com/Green-zy/poromat",  
    "repo_name": "poromat",
    "repo_type": "github",
    "globaltoc_depth": 2,  
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True
}
html_title = "Poromat Documentation"
html_sidebars = {
    "**": ["globaltoc.html", "localtoc.html", "searchbox.html", "sourcelink.html"],
}
