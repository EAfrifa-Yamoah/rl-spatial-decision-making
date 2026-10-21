"""Shared figure style and the document-wide colour semantics.

Every colour has exactly one meaning across all figures of the manuscript
and supplement (Okabe and Ito palette, checked for colour-vision deficiency):

  BLUE      value-based RL learner (Q-learning, SARSA) with a full state
  SKY       value-based RL learner with a restricted or compressed state
  ORANGE    policy-gradient RL learner (REINFORCE) and policy-based methods
  GREEN     model-based and planning methods (conceptual figures only)
  PURPLE    RL augmentations: multi-agent, hierarchical, transfer, POMDP
  VERMILION hazards and safety: moving obstacles, collisions, safety layers
  greys     non-learning baselines and heuristics; environment structure
            (walls and static obstacles are near-black)

Within a colour, line style, marker shape and marker fill distinguish
conditions, and every series is labelled directly or in a single legend.
Sequential maps: viridis for learned state values, cividis for the
concentration field. No figure carries an embedded title; captions live in
the manuscript.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
VERMILION = "#D55E00"
INK = "#1A1A1A"
GREY_DARK = "#4D4D4D"
GREY = "#808080"
GREY_LIGHT = "#BFBFBF"
GRID = "#E6E6E6"
OBSTACLE = "#262626"

FULL_WIDTH = 6.7   # inches; fits a 170 mm text block
DPI = 600


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": GREY_DARK,
        "axes.labelcolor": INK,
        "xtick.color": GREY_DARK,
        "ytick.color": GREY_DARK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="right", color=INK)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, path_stem):
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.pdf")
    plt.close(fig)
