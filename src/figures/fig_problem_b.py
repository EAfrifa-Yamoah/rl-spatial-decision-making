"""Figure 4 (Problem B) from results/raw/problem_b.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
LEARNER = {
    "iql_own": dict(label="Independent, own position", color=S.SKY, ls="-"),
    "iql_joint": dict(label="Independent, joint position", color=S.BLUE, ls="-"),
    "jal": dict(label="Centralised joint action", color=S.BLUE, ls=(0, (4, 1.5))),
}


def rolling(x, w=100):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode="valid")


def draw_map(ax, cfg):
    rows, cols = cfg["rows"], cfg["cols"]
    for (r, c) in cfg["walls"]:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=S.OBSTACLE, lw=0))
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color=S.GRID, lw=0.4, zorder=0)
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color=S.GRID, lw=0.4, zorder=0)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.GREY_DARK)
    cr = cfg["corridor_row"]
    # agent 1 (moves right) and agent 2 (moves left), canonical starts
    ax.plot(1, cr, "o", ms=6, mfc="white", mec=S.INK, mew=1.0, zorder=5)
    ax.plot(9, cr, "s", ms=5.5, mfc="white", mec=S.INK, mew=1.0, zorder=5)
    (g0r, g0c), (g1r, g1c) = cfg["goals"]
    ax.plot(g0c, g0r, "*", ms=9, mfc=S.INK, mec="white", mew=0.4, zorder=5)
    ax.plot(g1c, g1r, "*", ms=9, mfc="white", mec=S.INK, mew=0.8, zorder=5)
    ax.annotate("", xy=(3.4, cr + 0.33), xytext=(1.4, cr + 0.33),
                arrowprops=dict(arrowstyle="-|>", color=S.GREY_DARK, lw=0.7, mutation_scale=6))
    ax.annotate("", xy=(6.6, cr - 0.33), xytext=(8.6, cr - 0.33),
                arrowprops=dict(arrowstyle="-|>", color=S.GREY_DARK, lw=0.7, mutation_scale=6))


def spacetime(ax, run, spec, cfg):
    traj = run["canonical"]["trajectory"]
    goals = [tuple(g) for g in cfg["goals"]]
    ax.axhspan(3.5, 6.5, color=S.GREY_LIGHT, alpha=0.45, lw=0)
    for i, (ls, star_fill) in enumerate(((("-"), S.INK), ((0, (2, 1.2)), "white"))):
        pos = [tuple(p[i]) for p in traj]
        end = next((t for t, p in enumerate(pos) if p == goals[i]), len(pos) - 1)
        cols = [p[1] for p in pos[:end + 1]]
        ax.plot(np.arange(end + 1), cols, color=spec["color"], lw=1.3, ls=ls, drawstyle="steps-post")
        if pos[end] == goals[i]:
            ax.plot(end, cols[-1], "*", ms=7, mfc=star_fill, mec=S.INK, mew=0.6, zorder=6)
    c0 = [p[0][1] for p in traj]
    c1 = [p[1][1] for p in traj]
    ax.plot(0, c0[0], "o", ms=4.5, mfc="white", mec=S.INK, mew=0.8, zorder=5)
    ax.plot(0, c1[0], "s", ms=4, mfc="white", mec=S.INK, mew=0.8, zorder=5)
    ax.set_ylim(-0.5, cfg["cols"] - 0.5)
    ax.set_yticks([0, 5, 10])
    ax.set_xlim(0, cfg["max_steps"])
    ax.set_xlabel("Time step (greedy rollout)")
    ax.text(0.98, 5, "corridor", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=6, color=S.GREY_DARK)


def main():
    S.apply()
    d = json.loads((ROOT / "results/raw/problem_b.json").read_text())
    cfg = d["config"]
    runs = d["runs"]
    fig = plt.figure(figsize=(S.FULL_WIDTH, 4.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.55)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3)

    ax = fig.add_subplot(top[0])
    draw_map(ax, cfg)
    handles = [Line2D([], [], color=S.GREY_DARK, ls="-", marker="o", mfc="white", mec=S.INK, ms=4.5, lw=1.2,
                      label="Agent 1: start, path, goal"),
               Line2D([], [], color=S.GREY_DARK, ls=(0, (2, 1.2)), marker="s", mfc="white", mec=S.INK, ms=4,
                      lw=1.2, label="Agent 2: start, path, goal")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1, handlelength=2.6)
    S.panel_label(ax, "a")

    for k, (name, letter) in enumerate((("iql_own", "b"), ("iql_joint", "c"))):
        ax = fig.add_subplot(top[1 + k])
        run = next(r for r in runs if r["learner"] == name and r["seed"] == 0)
        spacetime(ax, run, LEARNER[name], cfg)
        ax.set_ylabel("Column" if k == 0 else "")
        S.panel_label(ax, letter)

    ax = fig.add_subplot(bot[0])
    for name, spec in LEARNER.items():
        M = np.array([rolling(r["train_success"]) for r in runs if r["learner"] == name])
        x = np.arange(M.shape[1]) + 100
        q1, med, q3 = np.quantile(M, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(x, q1, q3, color=spec["color"], alpha=0.18, lw=0)
        ax.plot(x, med, color=spec["color"], ls=spec["ls"], lw=1.3, label=spec["label"].replace(", ", ",\n").replace("Centralised joint", "Centralised\njoint"))
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episodes in which both\nagents reach their goals\n(rolling 100, proportion)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, cfg["budget"])
    S.ygrid(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.06), handlelength=1.7, handletextpad=0.5, labelspacing=0.7, borderaxespad=0.2)
    S.panel_label(ax, "d")

    ax = fig.add_subplot(bot[1])
    g = np.random.default_rng(3)
    for i, (name, spec) in enumerate(LEARNER.items()):
        vals = np.array([r["eval_success"] for r in runs if r["learner"] == name])
        x = i + g.uniform(-0.18, 0.18, len(vals))
        mk = "o" if name != "jal" else "D"
        ax.scatter(x, vals, s=8, color=spec["color"], alpha=0.6, lw=0, marker=mk, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color=S.INK, lw=1.4, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Independent,\nown position", "Independent,\njoint position", "Centralised\njoint action"])
    ax.set_ylabel("Greedy evaluation success\n(50 start pairs per seed)")
    ax.set_ylim(-0.04, 1.04)
    S.ygrid(ax)
    S.panel_label(ax, "e")
    S.save(fig, ROOT / "results/figures/Figure_4_ProblemB")


if __name__ == "__main__":
    main()
