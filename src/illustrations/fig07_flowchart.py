"""
Figure 7 — Practitioner's decision flowchart for spatial RL.

Each decision diamond routes the practitioner toward a recommended class
of methods based on problem characteristics (action-space cardinality,
state-space size, data availability, agent count, horizon, observability,
generalisation needs, and safety criticality).

Run:
    python -m src.illustrations.fig07_flowchart
"""
from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import save_figure


def _diamond(ax, x, y, text, w=3.0, h=0.9,
             color="#fff9c4", edge="#f57f17") -> None:
    d = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.05",
                                 linewidth=1.8, edgecolor=edge, facecolor=color)
    ax.add_patch(d)
    ax.text(x, y, text, fontsize=9.5, ha="center", va="center", fontweight="bold")


def _rect(ax, x, y, text, w=3.0, h=0.8,
          color="#e8f5e9", edge="#2e7d32") -> None:
    r = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.05",
                        linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(r)
    ax.text(x, y, text, fontsize=9, ha="center", va="center")


def _arrow(ax, x1, y1, x2, y2, label="", label_x=None, label_y=None,
           label_color="#333") -> None:
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                         mutation_scale=18, linewidth=1.3, color="#444444")
    ax.add_patch(a)
    if label:
        lx = label_x if label_x is not None else (x1 + x2) / 2
        ly = label_y if label_y is not None else (y1 + y2) / 2
        ax.text(lx, ly, label, fontsize=9, ha="center", fontweight="bold",
                color=label_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12.8)
    ax.axis("off")

    _diamond(ax, 6, 12.2, "Spatial problem\nwith sequential\ndecisions?", w=3.4, h=1.1)
    _rect(ax, 10.3, 12.2, "Use classical\nspatial optimization\n(e.g., MILP, heuristics)",
          w=3.0, h=1.1, color="#ffebee", edge="#b71c1c")
    _arrow(ax, 7.7, 12.2, 8.8, 12.2, "No")
    _arrow(ax, 6, 11.65, 6, 11.05, "Yes")

    _diamond(ax, 6, 10.6, "Action space\ndiscrete or\ncontinuous?", w=3.2, h=1.0)
    _arrow(ax, 4.4, 10.6, 2.6, 10.6, "Discrete", label_y=10.85)
    _arrow(ax, 7.6, 10.6, 9.4, 10.6, "Continuous", label_y=10.85)

    _diamond(ax, 2.2, 9.5, "State space\nsmall &\nenumerable?",
             w=2.6, h=1.0, color="#fff9c4", edge="#f57f17")
    _arrow(ax, 2.2, 10.1, 2.2, 10.0)
    _rect(ax, 0.95, 8.15, "Tabular Q-learning\nor SARSA",
          w=2.0, h=0.9, color="#e3f2fd", edge="#0d47a1")
    _arrow(ax, 1.6, 9.0, 1.3, 8.55, "Yes", label_x=1.15, label_y=8.85)
    _rect(ax, 3.5, 8.15, "DQN family\n(Double DQN,\nRainbow)",
          w=2.0, h=0.9, color="#e3f2fd", edge="#0d47a1")
    _arrow(ax, 2.8, 9.0, 3.2, 8.55, "No", label_x=3.3, label_y=8.85)

    _diamond(ax, 9.8, 9.5, "Training data\nabundant?",
             w=2.6, h=1.0, color="#fff9c4", edge="#f57f17")
    _arrow(ax, 9.8, 10.1, 9.8, 10.0)
    _rect(ax, 8.5, 8.15, "DDPG, TD3, SAC\n(model-free\ncontinuous control)",
          w=2.2, h=0.9, color="#fff3e0", edge="#e65100")
    _arrow(ax, 9.2, 9.0, 8.8, 8.55, "Yes", label_x=8.6, label_y=8.85)
    _rect(ax, 11.1, 8.15, "Model-based RL\n(Dreamer, MBPO)\nor sim-to-real",
          w=2.2, h=0.9, color="#fff3e0", edge="#e65100")
    _arrow(ax, 10.4, 9.0, 10.8, 8.55, "No", label_x=10.95, label_y=8.85)

    _diamond(ax, 6, 6.5, "Multiple cooperating\nor competing agents?",
             w=4.0, h=0.9)
    _arrow(ax, 0.95, 7.7, 4.4, 6.6)
    _arrow(ax, 3.5, 7.7, 4.4, 6.55)
    _arrow(ax, 8.5, 7.7, 7.6, 6.55)
    _arrow(ax, 11.1, 7.7, 7.6, 6.55)

    _rect(ax, 10.2, 6.5, "Add MARL\n(MADDPG, QMIX,\nCTDE)",
          w=2.8, h=0.9, color="#f3e5f5", edge="#4a148c")
    _arrow(ax, 8.0, 6.5, 8.8, 6.5, "Yes", label_y=6.75)
    _arrow(ax, 6, 6.05, 6, 5.6, "No", label_x=5.7, label_y=5.8)

    _diamond(ax, 6, 5.1, "Long horizon with\nreusable sub-tasks?",
             w=4.0, h=0.9)
    _rect(ax, 10.2, 5.1, "Hierarchical RL\n(options, FuN,\nHIRO)",
          w=2.8, h=0.9, color="#f3e5f5", edge="#4a148c")
    _arrow(ax, 8.0, 5.1, 8.8, 5.1, "Yes", label_y=5.35)
    _arrow(ax, 6, 4.65, 6, 4.2, "No", label_x=5.7, label_y=4.4)

    _diamond(ax, 6, 3.7, "Partial observability\nor active sensing?",
             w=4.0, h=0.9)
    _rect(ax, 10.2, 3.7,
          "POMDPs, recurrent\npolicies, DRQN, belief\nstate models",
          w=2.8, h=1.0, color="#f3e5f5", edge="#4a148c")
    _arrow(ax, 8.0, 3.7, 8.8, 3.7, "Yes", label_y=3.95)
    _arrow(ax, 6, 3.25, 6, 2.8, "No", label_x=5.7, label_y=3.0)

    _diamond(ax, 6, 2.3, "Deploy across\nregions / domains?",
             w=4.0, h=0.9)
    _rect(ax, 10.2, 2.3,
          "Transfer learning,\ndomain randomization,\nmeta-RL (MAML)",
          w=2.8, h=1.0, color="#f3e5f5", edge="#4a148c")
    _arrow(ax, 8.0, 2.3, 8.8, 2.3, "Yes", label_y=2.55)
    _arrow(ax, 6, 1.85, 6, 1.4, "No", label_x=5.7, label_y=1.6)

    _diamond(ax, 6, 0.9, "Safety-critical\ndeployment?", w=3.6, h=0.8)
    _rect(ax, 10.0, 0.9,
          "Constrained RL (CMDP),\nshielding, risk-sensitive\npolicies",
          w=3.2, h=0.9, color="#ffebee", edge="#b71c1c")
    _arrow(ax, 7.8, 0.9, 8.4, 0.9, "Yes", label_y=1.15)
    _rect(ax, 2.3, 0.9,
          "Proceed with\nstandard model\n+ validate via\nsimulation",
          w=2.4, h=1.0, color="#e8f5e9", edge="#2e7d32")
    _arrow(ax, 4.2, 0.9, 3.5, 0.9, "No", label_y=1.15)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig7_flowchart")
    args = parser.parse_args()

    fig = build_figure()
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig07] Saved figure to {out}")


if __name__ == "__main__":
    main()
