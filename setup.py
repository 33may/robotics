"""Setup script for smolvla_in_isaac package."""
from setuptools import setup, find_packages

setup(
    name="smolvla_in_isaac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Note: numpy<2.0 required for Isaac Sim compatibility
        # torch and lerobot installed separately to avoid dependency conflicts
    ],
    python_requires=">=3.10",
)
