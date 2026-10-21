"""Minimal package setup so that `pip install -e .` enables `python -m src.*`."""
from setuptools import find_packages, setup

setup(
    name="rl-spatial-decision-making",
    version="4.0.0",
    description=("Reproducible simulations, ablations, and figures for the review "
                 "'Reinforcement Learning for Spatial Decision Making: "
                 "Methodological Gaps in GIScience and a Research Agenda'."),
    author="Manuscript authors",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scipy>=1.10",
    ],
    extras_require={"dev": ["pytest>=7.4"]},
)
