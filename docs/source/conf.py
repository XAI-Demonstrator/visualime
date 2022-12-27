# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = "visuaLIME"
copyright = "2022, The XAI Demonstrator team"
author = "The XAI Demonstrator team"

import visualime  # noqa: E402

release = visualime.__version__
version = visualime.__version__

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.imgmath",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for autocod
autodoc_member_order = "bysource"

# -- Options for numpydoc
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "optional",
    "default",
    "of",
    "RGB",
    "RGBA",
    "type_without_description",
    "BadException",
}
numpydoc_xref_aliases = {
    "ints": "int",
    "strs": "str",
    "floats": "float",
}
numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}

# -- Options for Sphinx-Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../user_guide",  # path to your example scripts
    "gallery_dirs": "./user_guide",  # path to where to save gallery generated output
    "image_scrapers": ("matplotlib",),
    "filename_pattern": "/",
    "compress_images": ("images", "thumbnails"),
}
