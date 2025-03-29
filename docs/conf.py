import os
import subprocess
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, Path("../..").resolve())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pattern Discovery Kit"
copyright = "2025, Mansour Benbakoura"
author = "Mansour Benbakoura"
release = "0.2.3"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Pulls docstrings
    "sphinx.ext.autosummary",  # Generates summaries
    "sphinx.ext.napoleon",  # Supports Google/Numpy docstrings
    "sphinx.ext.viewcode",  # Links to source code
]

autosummary_generate = True  # Automatically generate summaries

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # Show type hints in descriptions

# -- Napoleon settings (for Google/Numpy docstrings) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"


# -- Auto-generate API docs --------------------------------------------------
def run_apidoc(_):
    """Run sphinx-apidoc to auto-generate .rst files"""
    module_path = os.path.abspath("../../my_package")  # Adjust path
    output_path = os.path.abspath("./source")
    subprocess.run(
        ["sphinx-apidoc", "-o", output_path, module_path, "--force"], check=True
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)
