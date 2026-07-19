"""
Figure 1 — Conceptual illustration of the Spatial Markov Decision Process.

A schematic of the agent–environment interaction loop with the Bellman
equation and three canonical spatial state representations: discrete
grid, continuous space with trajectory, and graph / network.

Run:
    python -m src.illustrations.fig01_mdp_framework
"""
from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import save_figure


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Agent box
    agent_box = FancyBboxPatch((0.6, 3.5), 3.2, 2.2, boxstyle="round,pad=0.1",
                                linewidth=2, edgecolor="#1f4e79", facecolor="#d9e7f5")
    ax.add_patch(agent_box)
    ax.text(2.2, 5.25, "AGENT", fontsize=13, fontweight="bold",
            ha="center", color="#1f4e79")
    ax.text(2.2, 4.75, "Policy π(a|s)", fontsize=10, ha="center")
    ax.text(2.2, 4.35, "Value function V(s)", fontsize=10, ha="center")
    ax.text(2.2, 3.95, "Q-function Q(s,a)", fontsize=10, ha="center")

    # Environment box
    env_box = FancyBboxPatch((8.2, 3.5), 3.2, 2.2, boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor="#c55a11", facecolor="#fce4d6")
    ax.add_patch(env_box)
    ax.text(9.8, 5.25, "SPATIAL ENVIRONMENT", fontsize=12, fontweight="bold",
            ha="center", color="#c55a11")
    ax.text(9.8, 4.75, "Terrain / Grid / Map", fontsize=10, ha="center")
    ax.text(9.8, 4.35, "Obstacles, resources", fontsize=10, ha="center")
    ax.text(9.8, 3.95, "Transition T(s'|s,a)", fontsize=10, ha="center")

    # Action arrow
    ax.add_patch(FancyArrowPatch((3.9, 5.3), (8.1, 5.3), arrowstyle="->",
                                  mutation_scale=25, linewidth=2.5, color="#2e7d32"))
    ax.text(6, 5.6, "Action  aₜ", fontsize=12, fontweight="bold",
            ha="center", color="#2e7d32")
    ax.text(6, 5.35, "(move, sense, place)", fontsize=9, ha="center",
            color="#2e7d32", style="italic")

    # State arrow
    ax.add_patch(FancyArrowPatch((8.1, 4.5), (3.9, 4.5), arrowstyle="->",
                                  mutation_scale=25, linewidth=2.5, color="#6a1b9a"))
    ax.text(6, 4.7, "State  sₜ₊₁", fontsize=11, fontweight="bold",
            ha="center", color="#6a1b9a")

    # Reward arrow
    ax.add_patch(FancyArrowPatch((8.1, 3.75), (3.9, 3.75), arrowstyle="->",
                                  mutation_scale=25, linewidth=2.5, color="#b71c1c"))
    ax.text(6, 3.85, "Reward  rₜ₊₁", fontsize=11, fontweight="bold",
            ha="center", color="#b71c1c")

    # Bellman equation box
    eq_box = FancyBboxPatch((3.5, 6.0), 5, 0.6, boxstyle="round,pad=0.05",
                             linewidth=1, edgecolor="#333333", facecolor="#f5f5f5")
    ax.add_patch(eq_box)
    ax.text(6, 6.3,
            r"Bellman: $V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0 = s\right]$",
            fontsize=11, ha="center")

    # Bottom: three representations
    ax.text(6, 2.9, "Spatial state representations",
            fontsize=11, fontweight="bold", ha="center")

    # (a) grid
    ax.text(1.7, 2.4, "(a) Discrete grid", fontsize=9, ha="center", fontweight="bold")
    grid_x, grid_y, cell = 0.7, 0.5, 0.25
    for i in range(6):
        for j in range(6):
            color = "#ffffff"
            if (i, j) == (0, 0):
                color = "#81c784"
            elif (i, j) == (5, 5):
                color = "#e57373"
            elif (i, j) in [(2, 1), (2, 2), (2, 3), (3, 3), (4, 1)]:
                color = "#424242"
            ax.add_patch(mpatches.Rectangle(
                (grid_x + j * cell, grid_y + i * cell), cell, cell,
                facecolor=color, edgecolor="#999999", linewidth=0.5))

    # (b) continuous
    ax.text(6, 2.4, "(b) Continuous space", fontsize=9, ha="center", fontweight="bold")
    cx, cy = 4.8, 0.5
    ax.add_patch(mpatches.Rectangle((cx, cy), 1.8, 1.6,
                                     facecolor="#f0f4c3",
                                     edgecolor="#666666", linewidth=1))
    t = np.linspace(0, 1, 50)
    traj_x = cx + 0.15 + 1.5 * t
    traj_y = cy + 0.3 + 1.0 * np.sin(3 * t) * (1 - t) + 0.5 * t
    ax.plot(traj_x, traj_y, "-", color="#1565c0", linewidth=2)
    ax.plot(cx + 0.15, cy + 0.3, "o", color="#2e7d32", markersize=8)
    ax.plot(cx + 1.65, cy + 1.3, "*", color="#c62828", markersize=14)
    ax.add_patch(mpatches.Circle((cx + 0.9, cy + 0.9), 0.15, facecolor="#424242"))
    ax.add_patch(mpatches.Circle((cx + 1.3, cy + 0.5), 0.12, facecolor="#424242"))

    # (c) graph
    ax.text(10, 2.4, "(c) Graph / network", fontsize=9,
            ha="center", fontweight="bold")
    nodes = [(8.9, 0.8), (9.4, 1.6), (10.1, 0.9), (10.7, 1.8),
             (11.1, 0.8), (9.7, 2.1), (10.4, 1.3)]
    edges = [(0, 1), (0, 2), (1, 2), (1, 5), (2, 6), (6, 3),
             (6, 4), (3, 4), (5, 3), (2, 4)]
    for e in edges:
        ax.plot([nodes[e[0]][0], nodes[e[1]][0]],
                [nodes[e[0]][1], nodes[e[1]][1]], "-",
                color="#888888", linewidth=1, zorder=1)
    for i, (nx, ny) in enumerate(nodes):
        color = "#1565c0"
        if i == 0:
            color = "#2e7d32"
        if i == 4:
            color = "#c62828"
        ax.plot(nx, ny, "o", color=color, markersize=10, zorder=2,
                markeredgecolor="white", markeredgewidth=1.5)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig1_mdp_framework")
    args = parser.parse_args()

    fig = build_figure()
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig01] Saved figure to {out}")


if __name__ == "__main__":
    main()
