"""Figure 5 (Problem C) and Supplementary Figure S2 (shifted fields and
budget curves) from results/raw/problem_c.json and results/manifest.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
N = 30
ALG = {"reinforce": dict(color=S.ORANGE, label="REINFORCE (policy gradient)", marker="o"),
       "qlearn": dict(color=S.SKY, label="Linear Q-learning (value based)", marker="D")}
REWARDS = ("rmse", "conc")
REWARD_TICK = {"rmse": "RMSE reduction", "conc": "Measured\nconcentration"}
HEUR_REF = {"random": dict(ls=(0, (1, 1.5)), label="Random"),
            "ucb": dict(ls="-", label="UCB, $\\kappa$ = 1.5"),
            "ivr": dict(ls=(0, (4, 2)), label="IVR"),
            "spacefill": dict(ls=(0, (5, 1.5, 1, 1.5)), label="Space filling"),
            "maxvar": dict(ls=(0, (6, 1.5, 1, 1.5, 1, 1.5)), label="Max variance"),
            "greedy": dict(ls=(0, (2, 2)), label="Greedy, $\\kappa$ = 0")}
HEUR_MARK = {"random": "x", "greedy": "v", "ucb": "^", "maxvar": "s", "ivr": "D", "spacefill": "P"}
ALL_HEUR = ("random", "greedy", "ucb", "maxvar", "ivr", "spacefill")


def placements_panel(ax, field, order, style, letter):
    F = np.asarray(field).reshape(N, N)
    ax.imshow(F, origin="lower", cmap="cividis", interpolation="nearest")
    xy = np.array([(c % N, c // N) for c in order])
    ax.scatter(xy[:, 0], xy[:, 1], **style, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.GREY_DARK)
    S.panel_label(ax, letter)


def interaction(ax, block, metric, ylabel, heur_rules, label_side=True, prior=None):
    for alg, spec in ALG.items():
        xs, ys, lo, hi = [], [], [], []
        for i, rw in enumerate(REWARDS):
            s = block["learned"][f"{alg}_{rw}"][metric]
            off = -0.05 if alg == "reinforce" else 0.05
            xs.append(i + off)
            ys.append(s["estimate"])
            lo.append(s["estimate"] - s["ci95"][0])
            hi.append(s["ci95"][1] - s["estimate"])
        ax.plot(xs, ys, color=spec["color"], lw=1.2, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=spec["color"], elinewidth=1.0, zorder=3)
        for x, y, rw in zip(xs, ys, REWARDS):
            ax.plot(x, y, spec["marker"], ms=5, mfc=spec["color"] if rw == "rmse" else "white",
                    mec=spec["color"], mew=1.2, zorder=4)
    ys = []
    for rule in heur_rules:
        y = block["heur"][rule][metric]["estimate"]
        ax.axhline(y, color=S.GREY, lw=0.9, ls=HEUR_REF[rule]["ls"], zorder=1)
        ys.append((y, HEUR_REF[rule]["label"]))
    if prior is not None:
        ax.axhline(prior, color=S.INK, lw=0.9, ls=(0, (1, 1)), zorder=1)
        ys.append((prior, "No sensors (prior mean)"))
    if label_side:
        ax.figure.canvas.draw()
        lo_, hi_ = ax.get_ylim()
        gap = 0.075 * (hi_ - lo_)
        ys.sort()
        placed = []
        for y, lab in ys:
            yy = y if not placed else max(y, placed[-1] + gap)
            placed.append(yy)
        # shift down if the top label overflows the axis
        over = placed[-1] - (hi_ - 0.01 * (hi_ - lo_))
        if over > 0:
            placed = [p - over for p in placed]
            for k in range(len(placed) - 2, -1, -1):
                placed[k] = min(placed[k], placed[k + 1] - gap)
        for (y, lab), yy in zip(ys, placed):
            ink = S.INK if lab.startswith("No sensors") else S.GREY_DARK
            ax.text(1.445, yy, lab, va="center", ha="left", fontsize=6, color=ink)
            # short leader from the line end to its label, so displaced labels stay attached
            ax.plot([1.40, 1.435], [y, yy], color=ink, lw=0.5, clip_on=False, zorder=1)
    ax.set_xlim(-0.35, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([REWARD_TICK[r] for r in REWARDS])
    ax.set_xlabel("Reward used for training")
    ax.set_ylabel(ylabel)
    S.ygrid(ax)


def main():
    S.apply()
    d = json.loads((ROOT / "results/raw/problem_c.json").read_text())
    m = json.loads((ROOT / "results/manifest.json").read_text())["C"]
    field = d["field0"]
    orders = d["orders_field0"]

    fig = plt.figure(figsize=(S.FULL_WIDTH, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.2], hspace=0.42)
    top = outer[0].subgridspec(1, 4, wspace=0.08)
    bot = outer[1].subgridspec(1, 2, wspace=0.62)
    heur_style = dict(s=16, marker="o", facecolor="white", edgecolor=S.INK, linewidths=0.8)
    panels = [
        ("ucb", heur_style, "a"),
        ("spacefill", dict(heur_style, marker="s", s=14), "b"),
        ("reinforce_rmse", dict(s=18, marker="o", facecolor=S.ORANGE, edgecolor=S.INK, linewidths=0.6), "c"),
        ("reinforce_conc", dict(s=18, marker="o", facecolor="white", edgecolor=S.ORANGE, linewidths=1.3), "d"),
    ]
    for i, (name, sty, letter) in enumerate(panels):
        ax = fig.add_subplot(top[i])
        placements_panel(ax, field, orders[name], sty, letter)
    im = ax.images[0]
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("True concentration", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_linewidth(0.5)

    blk = m["standard"]
    ax = fig.add_subplot(bot[0])
    interaction(ax, blk, "rmse", "Terminal RMSE after 20 sensors\n(median over 30 test fields)", ALL_HEUR)
    S.panel_label(ax, "e")
    ax = fig.add_subplot(bot[1])
    interaction(ax, blk, "recall", "Top decile hotspot recall\n(median over 30 test fields)", ALL_HEUR)
    S.panel_label(ax, "f")

    handles = [Line2D([], [], color=ALG["reinforce"]["color"], marker="o", ms=4.5, mfc=ALG["reinforce"]["color"],
                      lw=1.2, label=ALG["reinforce"]["label"]),
               Line2D([], [], color=ALG["qlearn"]["color"], marker="D", ms=4, mfc=ALG["qlearn"]["color"],
                      lw=1.2, label=ALG["qlearn"]["label"]),
               Line2D([], [], color=S.GREY, lw=0.9, label="Heuristic rules (no learning)"),
               Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=S.GREY_DARK,
                      label="Hollow: trained on concentration")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2,
               handlelength=2.4, columnspacing=1.6)
    S.save(fig, ROOT / "results/figures/Figure_5_ProblemC")

    # ------------- Supplementary figure: budget curves and shifted fields
    fig = plt.figure(figsize=(S.FULL_WIDTH, 5.2))
    gsx = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.55, wspace=0.75)
    axes = [fig.add_subplot(gsx[0, :]), fig.add_subplot(gsx[1, 0]), fig.add_subplot(gsx[1, 1])]
    ax = axes[0]
    for rule in ALL_HEUR:
        T = np.array([o["rmse_traj"] for o in d["heuristics_standard"][rule]])
        ax.plot(np.arange(T.shape[1]), np.median(T, 0), color=S.GREY, ls="-", lw=0.9,
                marker=HEUR_MARK[rule], ms=3.2, mfc="white", mec=S.GREY_DARK, mew=0.7, markevery=(2, 3))
    for name in ("reinforce_rmse", "reinforce_conc", "qlearn_rmse", "qlearn_conc"):
        alg, rw = name.split("_")
        T = np.array([[o["rmse_traj"] for o in r["eval"]["standard"]] for r in d["learned"] if r["learner"] == name])
        med = np.median(T.mean(0), 0)
        ax.plot(np.arange(len(med)), med, color=ALG[alg]["color"], lw=1.3,
                ls="-" if rw == "rmse" else (0, (3, 1.5)))
    ax.set_xlabel("Number of sensors placed")
    ax.set_ylabel("RMSE (median over\n30 test fields)")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    S.ygrid(ax)
    S.panel_label(ax, "a")
    hl = [Line2D([], [], color=S.GREY, lw=0.9, marker=HEUR_MARK[r], ms=3.2, mfc="white", mec=S.GREY_DARK,
                 mew=0.7, label=HEUR_REF[r]["label"]) for r in ALL_HEUR]
    hl += [Line2D([], [], color=S.ORANGE, lw=1.2, label="REINFORCE, RMSE reward"),
           Line2D([], [], color=S.ORANGE, lw=1.2, ls=(0, (3, 1.5)), label="REINFORCE, concentration reward"),
           Line2D([], [], color=S.SKY, lw=1.2, label="Q-learning, RMSE reward"),
           Line2D([], [], color=S.SKY, lw=1.2, ls=(0, (3, 1.5)), label="Q-learning, concentration reward")]
    ax.legend(handles=hl, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.6, fontsize=6)
    blk = m["shift"]
    interaction(axes[1], blk, "rmse", "Terminal RMSE,\nshifted fields", ALL_HEUR, label_side=True,
                prior=blk["prior_rmse"]["estimate"])
    S.panel_label(axes[1], "b")
    interaction(axes[2], blk, "recall", "Hotspot recall,\nshifted fields", ALL_HEUR, label_side=True)
    S.panel_label(axes[2], "c")
    for ax in axes[1:]:
        ax.set_xticklabels(["RMSE\nreduction", "Concen-\ntration"])
    S.save(fig, ROOT / "results/figures/Figure_S2_ProblemC_budget_shift")


if __name__ == "__main__":
    main()


def kappa_figure():
    S.apply()
    m = json.loads((ROOT / "results/manifest.json").read_text())["C"]["standard"]
    ks = sorted(m["kappa"], key=float)
    x = [float(k) for k in ks]
    est = [m["kappa"][k]["estimate"] for k in ks]
    lo = [m["kappa"][k]["ci95"][0] for k in ks]
    hi = [m["kappa"][k]["ci95"][1] for k in ks]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.fill_between(x, lo, hi, color=S.GREY_LIGHT, alpha=0.6, lw=0)
    ax.plot(x, est, color=S.GREY_DARK, lw=1.2, marker="o", ms=3.5, mfc="white", mec=S.GREY_DARK)
    mv = m["heur"]["maxvar"]["rmse"]
    ax.errorbar([6.2], [mv["estimate"]], yerr=[[mv["estimate"] - mv["ci95"][0]], [mv["ci95"][1] - mv["estimate"]]],
                fmt="s", ms=3.5, mfc="white", mec=S.GREY_DARK, ecolor=S.GREY_DARK, elinewidth=0.8)
    for name, col, ls in (("reinforce_rmse", S.ORANGE, "-"), ("qlearn_rmse", S.SKY, (0, (4, 1.5)))):
        ax.axhline(m["learned"][name]["rmse"]["estimate"], color=col, lw=1.0, ls=ls)
    ax.set_xticks(x + [6.2])
    ax.set_xticklabels([("%g" % v) for v in x] + ["max\nvar."])
    ax.set_xlabel("UCB exploration coefficient $\\kappa$")
    ax.set_ylabel("Terminal RMSE (median over\n30 test fields)")
    S.ygrid(ax)
    handles = [Line2D([], [], color=S.ORANGE, lw=1.0, label="REINFORCE, RMSE reward"),
               Line2D([], [], color=S.SKY, lw=1.0, ls=(0, (4, 1.5)), label="Q-learning, RMSE reward")]
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    S.save(fig, ROOT / "results/figures/Figure_S1_ProblemC_kappa")


if __name__ == "__main__":
    kappa_figure()
