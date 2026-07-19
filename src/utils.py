"""
Shared utilities for the RL-for-Spatial-Decision-Making simulations.

Contains:
- Reproducibility helpers (global seeding).
- Plot styling defaults.
- Output-directory resolution.

Every simulation module imports from here rather than duplicating boilerplate.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Repository root resolution — works whether scripts are run from the repo
# root, from src/, or from a notebook.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed the built-in random and NumPy RNGs.

    Parameters
    ----------
    seed : int
        Integer seed applied to Python's `random` module and to
        `numpy.random.seed`. Use this at the top of every simulation to
        make results reproducible across machines.
    """
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------
# Plot styling
# ----------------------------------------------------------------------
def configure_plot_style() -> None:
    """Apply a consistent matplotlib style across all figures."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save_figure(fig: plt.Figure, name: str, dpi: int = 400) -> Path:
    """Save a figure to `figures/<name>.png` at the requested DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Filename stem (no extension).
    dpi : int, default 400
        Resolution. 400 is print-quality; 200 is screen-quality.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved PNG.
    """
    out_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return out_path


def moving_average(x, window: int = 20):
    """Simple trailing moving average (length N - window + 1)."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


# Apply style on import so scripts don't need to call it explicitly.
configure_plot_style()
