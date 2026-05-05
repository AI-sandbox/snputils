from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import warnings

try:
    from nbformat import MissingIDFieldWarning
except ImportError:  # pragma: no cover - available whenever myst-nb is installed
    MissingIDFieldWarning = None
else:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
warnings.filterwarnings("ignore", message="Cell is missing an id field.*")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "snputils"
author = "The snputils developers"
copyright = f"{datetime.now().year}, {author}"

try:
    release = version("snputils")
except PackageNotFoundError:
    release = "dev"

version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "show"

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_typehints = "none"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
suppress_warnings = ["myst.header"]

autodoc_mock_imports = [
    "adjustText",
    "allel",
    "joblib",
    "matplotlib",
    "nbformat",
    "numpy",
    "pandas",
    "pgenlib",
    "plotly",
    "plotly_express",
    "polars",
    "pygrgl",
    "scipy",
    "sklearn",
    "torch",
    "tqdm",
    "zstandard",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
html_theme = "furo"
html_title = "snputils"
html_logo = "../assets/logo.png"
html_favicon = "https://snputils.org/su_favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["_extra"]
html_show_sourcelink = True
html_theme_options = {
    "source_repository": "https://github.com/AI-sandbox/snputils/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#1f766f",
        "color-brand-content": "#195f5a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ec9bf",
        "color-brand-content": "#8adbd4",
    },
}
