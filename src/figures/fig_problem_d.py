"""Figure 6 (Problem D) from results/raw/problem_d.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
COND = {
    ("blind", "fixed"): dict(color=S.SKY, ls="-", marker="o", label="Row-blind, trained on row 4"),
    ("blind", "varied"): dict(color=S.SKY, ls=(0, (4, 1.5)), marker="s", label="Row-blind, trained on rows 2-4"),
    ("aware", "fixed"): dict(color=S.BLUE, ls="-", marker="o", label="Row-aware, trained on row 4"),
    ("aware", "varied"): dict(color=S.BLUE, ls=(0, (4, 1.5)), marker="s", label="Row-aware, trained on rows 2-4"),
}
SNAP_T = (0, 8, 12, 16)


def snapshot(ax, traj, t, cfg):
    g = cfg["grid"]
    for k in range(g + 1):
        ax.axvline(k - 0.5, color=S.GRID, lw=0.4, zorder=0)
        ax.axhline(k - 0.5, color=S.GRID, lw=0.4, zorder=0)
    orow = traj[0][1][0][0]
    ax.add_patch(Rectangle((-0.5, orow - 0.5), g, 1, color=S.VERMILION, alpha=0.12, lw=0))
    path = np.array([traj[k][0] for k in range(t + 1)])
    ax.plot(path[:, 1], path[:, 0], color=S.SKY, lw=1.4, zorder=3)
    ax.plot(0, 0, "o", ms=4.5, mfc="white", mec=S.INK, mew=0.8, zorder=4)
    gr, gc = cfg["goal"]
    ax.plot(gc, gr, "*", ms=8, mfc=S.INK, mec="white", mew=0.4, zorder=4)
    ar, ac = traj[t][0]
    ax.plot(ac, ar, "o", ms=6, mfc=S.SKY, mec=S.INK, mew=0.7, zorder=6)
    _, ocol, odir = traj[t][1][0]
    ax.add_patch(Rectangle((ocol - 0.38, orow - 0.38), 0.76, 0.76, color=S.VERMILION, lw=0, zorder=5))
    ax.annotate("", xy=(ocol + (0.95 if odir == 0 else -0.95), orow),
                xytext=(ocol + (0.42 if odir == 0 else -0.42), orow),
                arrowprops=dict(arrowstyle="-|>", color=S.VERMILION, lw=0.8, mutation_scale=6), zorder=5)
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.GREY_DARK)
    ax.text(0.03, 0.97, f"t = {t}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=S.INK, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def main():
    S.apply()
    d = json.loads((ROOT / "results/raw/problem_d.json").read_text())
    cfg = d["config"]
    runs = d["runs"]
    fig = plt.figure(figsize=(S.FULL_WIDTH, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.38)
    top = outer[0].subgridspec(1, 4, wspace=0.08)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.95, 1.25, 0.9], wspace=0.42)

    demo = next(r for r in runs if r["state"] == "blind" and r["regime"] == "fixed" and r["seed"] == 0)
    traj = demo["rollout_row4"]["traj"]
    for i, t in enumerate(SNAP_T):
        ax = fig.add_subplot(top[i])
        snapshot(ax, traj, t, cfg)
        if i == 0:
            S.panel_label(ax, "a")

    # (b) training curves
    ax = fig.add_subplot(bot[0])
    for (st, rg), spec in COND.items():
        M = np.array([np.convolve(np.array(r["train_success"], float), np.ones(100) / 100, mode="valid")
                      for r in runs if r["state"] == st and r["regime"] == rg])
        x = np.arange(M.shape[1]) + 100
        q1, med, q3 = np.quantile(M, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(x, q1, q3, color=spec["color"], alpha=0.15, lw=0)
        ax.plot(x, med, color=spec["color"], ls=spec["ls"], lw=1.2)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Training success\n(rolling 100 episodes)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, cfg["budget"])
    S.ygrid(ax)
    S.panel_label(ax, "b")

    # (c) evaluation success by obstacle row
    ax = fig.add_subplot(bot[1])
    rows = cfg["test_rows"]
    offs = {("blind", "fixed"): -0.21, ("blind", "varied"): -0.07, ("aware", "fixed"): 0.07, ("aware", "varied"): 0.21}
    for (st, rg), spec in COND.items():
        sel = [r for r in runs if r["state"] == st and r["regime"] == rg]
        train_rows = cfg["train_rows"][rg]
        means, los, his = [], [], []
        for row in rows:
            v = np.array([r["eval_rows"][str(row)] for r in sel])
            g = np.random.default_rng(row)
            bs = v[g.integers(0, len(v), size=(10000, len(v)))].mean(1)
            means.append(v.mean())
            los.append(v.mean() - np.quantile(bs, 0.025))
            his.append(np.quantile(bs, 0.975) - v.mean())
        x = np.array(rows) + offs[(st, rg)]
        ax.plot(x, means, color=spec["color"], ls=spec["ls"], lw=0.7, alpha=0.6, zorder=2)
        ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor=spec["color"], elinewidth=0.8, capsize=0, zorder=2)
        for xi, row, m in zip(x, rows, means):
            filled = row in train_rows
            ax.plot(xi, m, spec["marker"], ms=4.2, mfc=spec["color"] if filled else "white",
                    mec=spec["color"], mew=1.0, zorder=3)
    ax.set_xticks(rows)
    ax.set_xlabel("Obstacle row at evaluation")
    ax.set_ylabel("Greedy success\n(mean over 30 seeds)")
    ax.set_ylim(-0.04, 1.06)
    S.ygrid(ax)
    S.panel_label(ax, "c")

    # (d) two simultaneous obstacles
    ax = fig.add_subplot(bot[2])
    g = np.random.default_rng(5)
    for i, ((st, rg), spec) in enumerate(COND.items()):
        v = np.array([r["eval_two_obstacles"] for r in runs if r["state"] == st and r["regime"] == rg])
        x = i + g.uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=7, color=spec["color"], marker=spec["marker"], alpha=0.6, lw=0, zorder=2)
        ax.plot([i - 0.27, i + 0.27], [v.mean()] * 2, color=S.INK, lw=1.3, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Blind\n4", "Blind\n2-4", "Aware\n4", "Aware\n2-4"], fontsize=6.2)
    ax.set_xlabel("State, training rows")
    ax.set_ylabel("Greedy success, two\nobstacles (rows 2 and 4)")
    ax.set_ylim(-0.04, 1.04)
    S.ygrid(ax)
    S.panel_label(ax, "d")

    handles = [Line2D([], [], color=spec["color"], ls=spec["ls"], marker=spec["marker"], ms=4,
                      mfc=spec["color"], mec=spec["color"], lw=1.2, label=spec["label"]) for spec in COND.values()]
    handles += [Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=S.GREY_DARK,
                       label="Hollow marker: row held out from training")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, handlelength=2.6,
               columnspacing=1.4)
    S.save(fig, ROOT / "results/figures/Figure_6_ProblemD")


if __name__ == "__main__":
    main()
