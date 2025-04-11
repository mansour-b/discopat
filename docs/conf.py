import os
import subprocess
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
on_rtd = os.environ.get("READTHEDOCS") == "True"

if on_rtd:
    sys.path.insert(0, os.path.abspath(".."))  # or '../src' if using src layout


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pattern Discovery Kit"
copyright = "2025, Mansour Benbakoura"
author = "Mansour Benbakoura"
release = "0.2.3"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Pull docstrings
    "sphinx.ext.autosummary",  # Generate summaries
    "sphinx.ext.napoleon",  # Support Google/Numpy docstrings
    "sphinx.ext.viewcode",  # Links to source code,
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path where to save generated output
    "filename_pattern": r"plot_",  # include only files starting with plot_
}
if on_rtd:
    sphinx_gallery_conf["reset_modules"] = (("sphinxext.add_package_to_path",),)


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
    """Run sphinx-apidoc to auto-generate .rst files."""
    module_path = Path("../patrick").resolve()
    output_path = Path("./source").resolve()
    subprocess.run(
        ["sphinx-apidoc", "-o", output_path, module_path, "--force"], check=True
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)
