"""Conceptual Figures 1 (spatial MDP), 2 (algorithm taxonomy) and 8
(practitioner flowchart), drawn with the same colour semantics as the
data figures (see style.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]


def tint(hex_color, a):
    """Blend a colour with white (a = share of the colour)."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    return tuple(1 - a * (1 - rgb))


def box(ax, x, y, w, h, text="", fc="white", ec=S.GREY_DARK, lw=0.8, fs=7, weight="normal",
        ha="center", va="center", r=0.02, tc=S.INK, linespacing=1.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.012
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, fontweight=weight, color=tc,
                zorder=3, linespacing=linespacing)


def arrow(ax, x0, y0, x1, y1, color=S.GREY_DARK, lw=0.9, ms=7, style="-|>", conn="arc3"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0, connectionstyle=conn), zorder=1)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------- Figure 1
def figure1():
    fig, ax = canvas(S.FULL_WIDTH, 4.3)
    # agent and environment
    box(ax, 0.04, 0.60, 0.30, 0.33, "", ec=S.INK, lw=1.0)
    ax.text(0.19, 0.885, "Agent", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(0.19, 0.76, "Policy  $\\pi(a \\mid s)$\n\nValue functions\n$V^{\\pi}(s)$,  $Q^{\\pi}(s, a)$",
            ha="center", va="center", fontsize=7.2, linespacing=1.3)
    box(ax, 0.66, 0.60, 0.30, 0.33, "", ec=S.INK, lw=1.0)
    ax.text(0.81, 0.885, "Spatial environment", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(0.81, 0.76, "Terrain, obstacles, resources\n\nTransition  $T(s' \\mid s, a)$\n"
            "Reward  $R(s, a, s')$", ha="center", va="center", fontsize=7.2, linespacing=1.3)
    arrow(ax, 0.345, 0.86, 0.655, 0.86, lw=1.1)
    ax.text(0.5, 0.875, "action $a_t$  (move, sense, place, allocate)", ha="center", va="bottom", fontsize=7)
    arrow(ax, 0.655, 0.74, 0.345, 0.74, lw=1.1)
    ax.text(0.5, 0.755, "next state $s_{t+1}$", ha="center", va="bottom", fontsize=7)
    arrow(ax, 0.655, 0.64, 0.345, 0.64, lw=1.1)
    ax.text(0.5, 0.655, "reward $r_{t+1}$", ha="center", va="bottom", fontsize=7)
    # value definition and Bellman expectation equation
    box(ax, 0.10, 0.455, 0.80, 0.10, "", fc="#F4F4F4", ec=S.GREY_LIGHT, lw=0.6)
    ax.text(0.5, 0.505,
            "$V^{\\pi}(s)=\\mathbb{E}_{\\pi}\\left[\\sum_{t\\geq 0}\\gamma^{t} r_{t+1}\\mid s_0=s\\right]"
            "=\\sum_{a}\\pi(a\\mid s)\\sum_{s'}T(s'\\mid s,a)\\left[R(s,a,s')+\\gamma V^{\\pi}(s')\\right]$",
            ha="center", va="center", fontsize=7.6)
    # three representations
    ax_g = fig.add_axes([0.07, 0.04, 0.24, 0.36])
    n = 6
    obst = {(1, 1), (3, 1), (3, 2), (3, 3), (1, 4)}
    for r in range(n):
        for c in range(n):
            ax_g.add_patch(Rectangle((c, r), 1, 1, fc=S.OBSTACLE if (r, c) in obst else "white",
                                     ec=S.GREY_LIGHT, lw=0.5))
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 5)]
    ax_g.plot([c + 0.5 for r, c in path], [r + 0.5 for r, c in path], color=S.BLUE, lw=1.4)
    ax_g.plot(0.5, 0.5, "o", ms=6, mfc="white", mec=S.INK, mew=1)
    ax_g.plot(5.5, 5.5, "*", ms=9, mfc=S.INK, mec="white", mew=0.4)
    ax_g.set_xlim(0, n)
    ax_g.set_ylim(0, n)
    ax_g.set_aspect("equal")
    ax_g.axis("off")
    ax_g.text(-0.05, 1.03, "(a)", transform=ax_g.transAxes, fontsize=8.5, fontweight="bold", ha="right")

    ax_c = fig.add_axes([0.385, 0.04, 0.24, 0.36])
    ax_c.add_patch(Rectangle((0, 0), 1, 1, fc="#F7F7F7", ec=S.GREY_DARK, lw=0.6))
    for (cx, cy, rr) in ((0.42, 0.55, 0.09), (0.70, 0.30, 0.07), (0.22, 0.25, 0.06)):
        ax_c.add_patch(Circle((cx, cy), rr, fc=S.OBSTACLE, ec="none"))
    t = np.linspace(0, 1, 200)
    xs = 0.08 + 0.84 * t
    ys = 0.08 + 0.84 * t ** 1.6 + 0.16 * np.sin(np.pi * t) * np.cos(2.4 * np.pi * t)
    ax_c.plot(xs, ys, color=S.BLUE, lw=1.4)
    ax_c.plot(xs[0], ys[0], "o", ms=6, mfc="white", mec=S.INK, mew=1)
    ax_c.plot(0.92, 0.92, "*", ms=9, mfc=S.INK, mec="white", mew=0.4)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_aspect("equal")
    ax_c.axis("off")
    ax_c.text(-0.05, 1.03, "(b)", transform=ax_c.transAxes, fontsize=8.5, fontweight="bold", ha="right")

    ax_n = fig.add_axes([0.70, 0.04, 0.24, 0.36])
    nodes = {0: (0.05, 0.12), 1: (0.30, 0.55), 2: (0.40, 0.92), 3: (0.55, 0.20), 4: (0.70, 0.52),
             5: (0.88, 0.80), 6: (0.95, 0.12)}
    edges = [(0, 1), (1, 2), (1, 3), (2, 5), (3, 4), (4, 5), (3, 6), (4, 6), (5, 6), (0, 3)]
    route = [(0, 3), (3, 4), (4, 6)]
    for a, b in edges:
        on = (a, b) in route or (b, a) in route
        ax_n.plot(*zip(nodes[a], nodes[b]), color=S.BLUE if on else S.GREY_LIGHT, lw=1.6 if on else 0.9, zorder=1)
    for k, (x, y) in nodes.items():
        if k == 0:
            ax_n.plot(x, y, "o", ms=6.5, mfc="white", mec=S.INK, mew=1, zorder=3)
        elif k == 6:
            ax_n.plot(x, y, "*", ms=10, mfc=S.INK, mec="white", mew=0.4, zorder=3)
        else:
            ax_n.plot(x, y, "o", ms=5.5, mfc=S.GREY, mec="white", mew=0.6, zorder=2)
    ax_n.set_xlim(-0.03, 1.03)
    ax_n.set_ylim(-0.03, 1.03)
    ax_n.set_aspect("equal")
    ax_n.axis("off")
    ax_n.text(-0.05, 1.03, "(c)", transform=ax_n.transAxes, fontsize=8.5, fontweight="bold", ha="right")
    S.save(fig, ROOT / "results/figures/Figure_1_SpatialMDP")


# ------------------------------------------------------------- Figure 2
def figure2():
    fig, ax = canvas(S.FULL_WIDTH, 4.4)
    box(ax, 0.38, 0.88, 0.24, 0.085, "Reinforcement learning", fc="white", ec=S.INK, lw=1.0,
        fs=8, weight="bold")
    fams = [
        ("Value-based", S.BLUE, "Q-learning, SARSA\nDQN, Double DQN\nRainbow",
         "Discrete actions:\ngrids, graphs, routing"),
        ("Policy-based and\nactor-critic", S.ORANGE, "REINFORCE, A2C/A3C\nPPO, DDPG\nTD3, SAC",
         "Continuous control:\nUAVs, vehicles, robots"),
        ("Model-based and\nplanning", S.GREEN, "Dyna-Q, MBPO\nPlaNet, Dreamer\nMuZero",
         "Costly interaction:\nfield trials, ecology"),
    ]
    xs = (0.05, 0.37, 0.69)
    for (name, col, algs, best), x in zip(fams, xs):
        box(ax, x, 0.66, 0.26, 0.11, name, fc=tint(col, 0.22), ec=col, lw=1.2, fs=7.6, weight="bold")
        ax.text(x + 0.13, 0.575, algs, ha="center", va="center", fontsize=7, linespacing=1.35)
        box(ax, x + 0.02, 0.385, 0.22, 0.10, best, fc="white", ec=tint(col, 0.8), lw=0.7, fs=6.6)
        arrow(ax, 0.5, 0.88, x + 0.13, 0.775, lw=0.8, conn="arc3,rad=0")
    ax.plot([0.04, 0.96], [0.325, 0.325], color=S.GREY_LIGHT, lw=0.6)
    ax.text(0.5, 0.29, "Augmentations that target spatial realities (combine with any family)",
            ha="center", va="center", fontsize=7, style="italic", color=S.GREY_DARK)
    augs = [
        ("Hierarchical", "options, sub-goals for\nlong horizons", S.PURPLE),
        ("Multi-agent", "fleets, swarms,\npatrols", S.PURPLE),
        ("Transfer and\nmeta-learning", "regions, seasons,\nembodiments", S.PURPLE),
        ("Partial\nobservability", "POMDPs, memory,\nactive sensing", S.PURPLE),
        ("Safe and\nconstrained", "CMDPs, shields,\nfallback control", S.VERMILION),
    ]
    w = 0.172
    for i, (name, sub, col) in enumerate(augs):
        x = 0.04 + i * (w + 0.013)
        box(ax, x, 0.04, w, 0.20, "", fc=tint(col, 0.18), ec=col, lw=1.1)
        ax.text(x + w / 2, 0.185, name, ha="center", va="center", fontsize=7, fontweight="bold",
                linespacing=1.15)
        ax.text(x + w / 2, 0.085, sub, ha="center", va="center", fontsize=6.4, linespacing=1.25)
    S.save(fig, ROOT / "results/figures/Figure_2_Taxonomy")


# ------------------------------------------------------------- Figure 8
def figure8():
    fig, ax = canvas(S.FULL_WIDTH, 6.2)
    q = dict(fc="white", ec=S.INK, lw=0.9, fs=6.9, weight="bold")
    # trunk
    box(ax, 0.17, 0.875, 0.34, 0.095,
        "Can a classical method solve it?\n(DP, MPC, rolling-horizon MILP, heuristic)", **q)
    box(ax, 0.62, 0.885, 0.20, 0.075, "Use the classical\nmethod and stop", fc=tint(S.GREY, 0.25),
        ec=S.GREY_DARK, lw=0.8, fs=6.8)
    arrow(ax, 0.51, 0.922, 0.62, 0.922)
    ax.text(0.565, 0.93, "yes", ha="center", va="bottom", fontsize=6.5)
    arrow(ax, 0.34, 0.875, 0.34, 0.80)
    ax.text(0.348, 0.84, "no", ha="left", va="center", fontsize=6.5)
    box(ax, 0.20, 0.725, 0.28, 0.075, "Discrete or continuous\naction space?", **q)
    # discrete branch
    ax.plot([0.20, 0.125], [0.762, 0.762], color=S.GREY_DARK, lw=0.9, solid_capstyle="butt", zorder=1)
    arrow(ax, 0.125, 0.762, 0.125, 0.695)
    ax.text(0.16, 0.77, "discrete", ha="center", va="bottom", fontsize=6.5)
    box(ax, 0.02, 0.62, 0.21, 0.075, "State space small\nand enumerable?", **q)
    box(ax, 0.005, 0.50, 0.115, 0.08, "Tabular\nQ-learning,\nSARSA", fc=tint(S.BLUE, 0.22), ec=S.BLUE, fs=6.4)
    box(ax, 0.13, 0.50, 0.115, 0.08, "DQN family\n(Double DQN,\nRainbow)", fc=tint(S.BLUE, 0.22), ec=S.BLUE, fs=6.4)
    arrow(ax, 0.07, 0.62, 0.062, 0.58)
    arrow(ax, 0.18, 0.62, 0.188, 0.58)
    ax.text(0.055, 0.60, "yes", ha="right", va="center", fontsize=6.3)
    ax.text(0.195, 0.60, "no", ha="left", va="center", fontsize=6.3)
    # continuous branch
    ax.plot([0.48, 0.555], [0.762, 0.762], color=S.GREY_DARK, lw=0.9, solid_capstyle="butt", zorder=1)
    arrow(ax, 0.555, 0.762, 0.555, 0.695)
    ax.text(0.52, 0.77, "continuous", ha="center", va="bottom", fontsize=6.5)
    box(ax, 0.45, 0.62, 0.21, 0.075, "Can interaction be\nscaled cheaply?", **q)
    box(ax, 0.40, 0.50, 0.135, 0.08, "Actor-critic\n(DDPG, TD3, SAC)", fc=tint(S.ORANGE, 0.25), ec=S.ORANGE, fs=6.4)
    box(ax, 0.545, 0.50, 0.15, 0.08, "Model-based or\noffline RL (MBPO,\nDreamer, CQL)", fc=tint(S.GREEN, 0.22), ec=S.GREEN, fs=6.4)
    arrow(ax, 0.50, 0.62, 0.47, 0.58)
    arrow(ax, 0.61, 0.62, 0.62, 0.58)
    ax.text(0.468, 0.603, "yes", ha="right", va="center", fontsize=6.3)
    ax.text(0.623, 0.60, "no", ha="left", va="center", fontsize=6.3)
    # merge into the overlay checklist
    for x in (0.062, 0.188, 0.468, 0.62):
        arrow(ax, x, 0.50, x, 0.455, lw=0.7)
    ax.plot([0.062, 0.62], [0.455, 0.455], color=S.GREY_DARK, lw=0.7)
    arrow(ax, 0.34, 0.455, 0.34, 0.415)
    box(ax, 0.02, 0.025, 0.69, 0.39, "", fc="#FAFAFA", ec=S.GREY_LIGHT, lw=0.7)
    ax.text(0.035, 0.395, "Design axes to check on every branch (several usually apply)",
            ha="left", va="center", fontsize=6.9, fontweight="bold", color=S.INK)
    rows = [
        ("Several decision makers?", "Add multi-agent RL: CTDE, MADDPG, QMIX", S.PURPLE),
        ("Long horizon with reusable sub-tasks?", "Add hierarchy: options, FeUdal, HIRO", S.PURPLE),
        ("Partial observability or active sensing?", "Belief state or memory: POMDP, DRQN", S.PURPLE),
        ("Deployment beyond the training region?", "Transfer, domain randomisation, meta-RL;\nheld-out geography evaluation", S.PURPLE),
        ("Safety-critical consequences?", "Constrained RL, shielding, fallback\ncontroller, monitoring plan", S.VERMILION),
    ]
    for i, (qq, rec, col) in enumerate(rows):
        y = 0.315 - i * 0.068
        box(ax, 0.035, y, 0.315, 0.055, qq, fc="white", ec=S.INK, lw=0.7, fs=6.0, weight="bold")
        arrow(ax, 0.35, y + 0.0275, 0.385, y + 0.0275, lw=0.7)
        ax.text(0.3675, y + 0.034, "yes", ha="center", va="bottom", fontsize=5.6)
        box(ax, 0.385, y, 0.315, 0.055, rec, fc=tint(col, 0.2), ec=col, lw=0.8, fs=6.0)
    # side panel: always do
    box(ax, 0.74, 0.025, 0.24, 0.79, "", fc="white", ec=S.GREY_DARK, lw=0.8)
    ax.text(0.86, 0.79, "Always, whatever\nthe branch", ha="center", va="center", fontsize=7, fontweight="bold")
    steps = ["Write the MDP on one page;\nhave a domain expert\nchallenge it",
             "Benchmark classical and\nrandom baselines in the\nsame simulator",
             "Ablate the reward (sparse\nversus full) and report both",
             "Report at least 30 seeds\nwith intervals",
             "Evaluate on held-out\ngeography; report the gap",
             "Instrument for drift and\nplan a rollback"]
    for i, stp in enumerate(steps):
        y = 0.655 - i * 0.124
        box(ax, 0.755, y, 0.21, 0.10, stp, fc="white", ec=S.INK, lw=0.6, fs=6.2)
    S.save(fig, ROOT / "results/figures/Figure_8_Flowchart")


def main():
    S.apply()
    figure1()
    figure2()
    figure8()


if __name__ == "__main__":
    main()
