"""Sphinx build."""

import sys
import tomllib
from datetime import date
from os import path

# make sure one level up is on python path
sys.path.append(path.abspath(".."))

with open("../pyproject.toml", "rb") as fh:
    toml = tomllib.load(fh)

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

pyproject = toml["project"]

project = pyproject["name"]
version = pyproject["version"]
release = pyproject["version"]

copyright = f"{date.today().year:d} Erik Brinkman"  # noqa: A001
author = "Erik Brinkman"
