"""Setup script for vbti robotics package."""
from setuptools import setup, find_packages

setup(
    name="vbti",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "fire",
        "tqdm",
    ],
    python_requires=">=3.10",
)
