"""
Figure 3 — Taxonomy of RL methods for spatial problems.

Three top-level algorithm families (value-based, policy-based,
model-based) over five augmentations relevant to spatial realism
(hierarchical, multi-agent, transfer/meta, POMDP, safe/constrained).

Run:
    python -m src.illustrations.fig03_taxonomy
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils import save_figure


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # Root
    root = FancyBboxPatch((5.8, 8.3), 2.4, 0.7, boxstyle="round,pad=0.05",
                          linewidth=2, edgecolor="#263238", facecolor="#cfd8dc")
    ax.add_patch(root)
    ax.text(7, 8.65, "Reinforcement Learning",
            fontsize=11, fontweight="bold", ha="center")

    branches = [
        {"x": 2.2, "title": "Value-based", "color": "#e3f2fd", "edge": "#0d47a1",
         "algos": ["Q-learning", "SARSA", "DQN", "Double DQN", "Rainbow"],
         "best_for": "Discrete action\nspaces\n(grids, graphs)"},
        {"x": 7.0, "title": "Policy-based / Actor-Critic",
         "color": "#fff3e0", "edge": "#e65100",
         "algos": ["REINFORCE", "A2C / A3C", "PPO", "DDPG", "SAC", "TD3"],
         "best_for": "Continuous\ncontrol\n(robotics, UAVs)"},
        {"x": 11.5, "title": "Model-based / Planning-augmented",
         "color": "#e8f5e9", "edge": "#1b5e20",
         "algos": ["Dyna-Q", "MuZero", "MBPO", "PlaNet", "Dreamer"],
         "best_for": "Sample-efficient\nplanning\n(limited data)"},
    ]

    for b in branches:
        ax.plot([7, b["x"]], [8.3, 7.3], color="#666666",
                linewidth=1.2, zorder=1)
        box = FancyBboxPatch((b["x"] - 1.4, 6.5), 2.8, 0.8,
                              boxstyle="round,pad=0.05", linewidth=2,
                              edgecolor=b["edge"], facecolor=b["color"])
        ax.add_patch(box)
        ax.text(b["x"], 6.9, b["title"], fontsize=11, fontweight="bold",
                ha="center", color=b["edge"])

        for i, algo in enumerate(b["algos"]):
            ax.text(b["x"], 6.15 - i * 0.32, f"• {algo}",
                    fontsize=9.5, ha="center")

        best_box = FancyBboxPatch((b["x"] - 1.35, 3.3), 2.7, 1.25,
                                   boxstyle="round,pad=0.05", linewidth=1,
                                   edgecolor="#555555", facecolor="white")
        ax.add_patch(best_box)
        ax.text(b["x"], 4.32, "Best suited for:",
                fontsize=9.5, ha="center", fontweight="bold",
                style="italic", color="#555")
        for i, line in enumerate(b["best_for"].split("\n")):
            ax.text(b["x"], 3.95 - i * 0.25, line, fontsize=9, ha="center")

    ax.text(7, 2.8, "Augmentations for spatial realism",
            fontsize=12, fontweight="bold", ha="center")

    augs = [
        {"x": 1.8,  "name": "Hierarchical RL",
         "desc": "Option/subgoal\nabstraction for\nlong horizons"},
        {"x": 4.4,  "name": "Multi-agent RL",
         "desc": "Cooperation, competition,\nswarm coordination"},
        {"x": 7.0,  "name": "Transfer / Meta",
         "desc": "Generalise across\nregions, domains,\ntasks"},
        {"x": 9.6,  "name": "POMDP / Memory",
         "desc": "Partial observability,\nSLAM, active sensing"},
        {"x": 12.2, "name": "Safe / Constrained RL",
         "desc": "CMDPs, shielding\nfor deployment"},
    ]
    for a in augs:
        box = FancyBboxPatch((a["x"] - 1.15, 1.0), 2.3, 1.5,
                              boxstyle="round,pad=0.05", linewidth=1.5,
                              edgecolor="#4a148c", facecolor="#f3e5f5")
        ax.add_patch(box)
        ax.text(a["x"], 2.15, a["name"], fontsize=10, fontweight="bold",
                ha="center", color="#4a148c")
        ax.text(a["x"], 1.55, a["desc"], fontsize=8.5, ha="center")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig2_taxonomy")
    args = parser.parse_args()

    fig = build_figure()
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig03] Saved figure to {out}")


if __name__ == "__main__":
    main()
