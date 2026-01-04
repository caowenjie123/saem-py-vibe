"""
Backward-compatible setup.py for saemix package.

This file is maintained for compatibility with older pip versions and
editable installs. The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os


# Read version from _version.py (single source of truth)
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "saemix", "_version.py")
    version_dict = {}
    with open(version_file, "r", encoding="utf-8") as f:
        exec(f.read(), version_dict)
    return version_dict["__version__"]


setup(
    name="saemix",
    version=get_version(),
    packages=find_packages(include=["saemix", "saemix.*"]),
)
