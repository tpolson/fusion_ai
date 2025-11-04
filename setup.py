"""Setup script for fusion-ai package."""
from setuptools import setup, find_packages

setup(
    name="fusion-ai",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "fusion_ai": ["fuses/*.fuse", "fuses/*.lua"],
    },
)
