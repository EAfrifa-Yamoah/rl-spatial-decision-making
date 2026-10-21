"""Figure 3 (Problem A) from results/raw/problem_a.json and results/manifest.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
REWARD_ORDER = ("shaped", "step_cost", "pbrs", "sparse", "random_dense")
REWARD_LABEL = {"shaped": "Shaped", "step_cost": "Step cost only", "pbrs": "Potential based",
                "sparse": "Sparse", "random_dense": "Random dense"}
REWARD_LS = {"shaped": "-", "step_cost": (0, (5, 1.5)), "pbrs": (0, (1, 1)),
             "sparse": (0, (6, 2, 1, 2)), "random_dense": (0, (2, 2))}
REWARD_MK = {"shaped": "o", "step_cost": "s", "pbrs": "o", "sparse": "^", "random_dense": "s"}
REP_ORDER = ("tabular", "rowcol", "window", "hand")
REP_LABEL = {"tabular": "Tabular", "rowcol": "Row and\ncolumn",
             "window": "3 x 3\nwindow", "hand": "Hand-\nengineered"}
REWARD_TICK = {"shaped": "Shaped", "step_cost": "Step\ncost", "pbrs": "Potential\nbased",
               "sparse": "Sparse", "random_dense": "Random\ndense"}
ARROW = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}  # action -> (dx, dy) in plot coords


def draw_grid(ax, cfg):
    g = cfg["grid"]
    for (r, c) in cfg["obstacles"]:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=S.OBSTACLE, lw=0))
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.GREY_DARK)
    sr, sc = cfg["start"]
    gr, gc = cfg["goal"]
    ax.plot(sc, sr, "o", ms=6, mfc="white", mec=S.INK, mew=1.0, zorder=5)
    ax.plot(gc, gr, "*", ms=9, mfc=S.INK, mec="white", mew=0.5, zorder=5)


def dotplot(ax, groups, colors, labels, ylabel, rng_seed=0):
    g = np.random.default_rng(rng_seed)
    for i, (vals, col) in enumerate(zip(groups, colors)):
        vals = np.asarray(vals)
        x = i + g.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=7, color=col, alpha=0.55, lw=0, zorder=2)
        med = np.median(vals)
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color=S.INK, lw=1.4, zorder=3)
        ax.plot([i, i], [q1, q3], color=S.INK, lw=0.8, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.04, 1.04)
    S.ygrid(ax)


def main():
    S.apply()
    d = json.loads((ROOT / "results/raw/problem_a.json").read_text())
    cfg = d["config"]
    shaped0 = next(r for r in d["A1"] if r["reward"] == "shaped" and r["seed"] == 0)

    fig = plt.figure(figsize=(S.FULL_WIDTH, 6.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.85, 1.0], hspace=0.62)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1.12, 1.1], wspace=0.55)
    bottom = outer[2].subgridspec(1, 2, wspace=0.12)

    # (a) map and greedy policy (seed 0, shaped reward)
    ax = fig.add_subplot(top[0])
    draw_grid(ax, cfg)
    X, Y, U, Vv = [], [], [], []
    for key, a in shaped0["policy"].items():
        r, c = map(int, key.split(","))
        if (r, c) == tuple(cfg["start"]):
            continue
        dx, dy = ARROW[a]
        X.append(c - 0.3 * dx); Y.append(r - 0.3 * dy); U.append(dx); Vv.append(dy)
    ax.quiver(X, Y, U, Vv, color=S.BLUE, angles="xy", scale_units="xy", scale=1 / 0.6,
              width=0.012, headwidth=3.5, headlength=3.5, headaxislength=3.2, zorder=3)
    S.panel_label(ax, "a")

    # (b) learned state values
    ax = fig.add_subplot(top[1])
    V = np.array(shaped0["value_map"], dtype=float)
    Vm = np.ma.masked_invalid(V)
    im = ax.imshow(Vm, origin="lower", cmap="viridis", interpolation="nearest")
    draw_grid(ax, cfg)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, orientation="horizontal")
    cb.set_label("Learned state value, max$_a$ Q(s, a)")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    S.panel_label(ax, "b")

    # (c) shaped reward training return, 30 seeds
    ax = fig.add_subplot(top[2])
    R = np.array([r["returns"] for r in d["A1"] if r["reward"] == "shaped"])
    ep = np.arange(1, R.shape[1] + 1)
    med = np.median(R, 0)
    q1, q3 = np.quantile(R, [0.25, 0.75], axis=0)
    ax.fill_between(ep, q1, q3, color=S.BLUE, alpha=0.22, lw=0)
    ax.plot(ep, med, color=S.BLUE, lw=1.2)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return, shaped reward")
    ax.set_xlim(1, R.shape[1])
    S.ygrid(ax)
    S.panel_label(ax, "c")

    # (d) proportion of seeds whose greedy policy reaches the goal from the start
    ax = fig.add_subplot(outer[1])
    for rw in REWARD_ORDER:
        G = np.array([r["greedy_curve"] for r in d["A1"] if r["reward"] == rw], float)
        x = np.arange(1, G.shape[1] + 1) * 10
        y = G.mean(0)
        ax.plot(x, y, color=S.BLUE, ls=REWARD_LS[rw], lw=1.2, label=REWARD_LABEL[rw],
                marker=REWARD_MK[rw], markevery=(REWARD_ORDER.index(rw) * 2, 10), ms=3.5,
                mfc="white" if rw in ("pbrs", "random_dense") else S.BLUE, mec=S.BLUE, mew=0.8)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Proportion of seeds with\ngreedy success from start")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, cfg["budget"])
    S.ygrid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=3.2, labelspacing=0.5)
    S.panel_label(ax, "d")

    # (e) multi-start greedy success by reward
    ax = fig.add_subplot(bottom[0])
    groups = [[r["eval_multistart"] for r in d["A1"] if r["reward"] == rw] for rw in REWARD_ORDER]
    dotplot(ax, groups, [S.BLUE] * 5, [REWARD_TICK[rw] for rw in REWARD_ORDER],
            "Free cells from which the greedy\npolicy reaches the goal (proportion)")
    ax.set_xlabel("Reward specification (tabular state)")
    S.panel_label(ax, "e")

    # (f) multi-start greedy success by state representation
    ax_f = fig.add_subplot(bottom[1], sharey=ax)
    ax = ax_f
    groups = [[r["eval_multistart"] for r in d["A2"] if r["rep"] == rep] for rep in REP_ORDER]
    cols = [S.BLUE] + [S.SKY] * 3
    dotplot(ax, groups, cols, [REP_LABEL[r] for r in REP_ORDER], "", rng_seed=1)
    ax.set_xlabel("State representation (shaped reward)")
    plt.setp(ax.get_yticklabels(), visible=False)
    S.panel_label(ax, "f")

    S.save(fig, ROOT / "results/figures/Figure_3_ProblemA")


if __name__ == "__main__":
    main()
