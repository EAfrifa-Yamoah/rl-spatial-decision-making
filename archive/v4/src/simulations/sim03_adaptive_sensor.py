"""
Simulation 3 — Adaptive sensor placement as sequential spatial decision.

A 30x30 pollution field is generated as a sum of three Gaussian blobs plus
observation noise. Three acquisition strategies choose where to place each
of BUDGET sensors sequentially:

  * random : uniform cell selection
  * greedy : maximise the predicted concentration (exploits previous data)
  * rl     : upper confidence - predicted mean + kappa * sqrt(variance),
             emulating the exploration-exploitation trade-off a learned
             policy would make under information-gain + value rewards.

Posterior predictions come from a light RBF-kernel regressor that is
rebuilt after every new observation. The script compares reconstruction
RMSE across strategies over N_REPLICATES random seeds.

This reproduces Figure 5 of the manuscript (Section 7.2).

Run:
    python -m src.simulations.sim03_adaptive_sensor
    python -m src.simulations.sim03_adaptive_sensor --budget 30 --replicates 20
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_figure, set_seed


def build_field(n: int) -> np.ndarray:
    """Return an n x n pollution field: three blobs + observation noise."""
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    def gaussian(cx, cy, sx, sy, amp):
        return amp * np.exp(-((X - cx) ** 2 / (2 * sx ** 2)
                              + (Y - cy) ** 2 / (2 * sy ** 2)))

    field = (
        gaussian(7, 22, 3.2, 3.0, 1.0)
        + gaussian(20, 8, 4.5, 3.8, 0.85)
        + gaussian(24, 24, 2.5, 2.5, 0.6)
    )
    field += 0.03 * np.random.randn(n, n)
    return field


def rbf_regress(
    sample_coords: List[Tuple[int, int]],
    sample_vals: List[float],
    grid_pts: np.ndarray,
    length: float = 4.0,
    noise: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple RBF regression giving mean and variance over grid_pts."""
    if len(sample_coords) == 0:
        return np.zeros(len(grid_pts)), np.ones(len(grid_pts))

    sc = np.asarray(sample_coords, dtype=float)
    sv = np.asarray(sample_vals, dtype=float)
    n = len(sc)

    def kern(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-d / (2 * length ** 2))

    K = kern(sc, sc) + noise * np.eye(n)
    K_s = kern(grid_pts, sc)
    alpha = np.linalg.solve(K, sv)
    mean = K_s @ alpha
    v = np.linalg.solve(K, K_s.T)
    var = np.clip(1.0 - np.sum(K_s * v.T, axis=1), 1e-6, None)
    return mean, var


def run_strategy(
    strategy: str,
    field: np.ndarray,
    budget: int,
    n: int,
    kappa: float = 1.5,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Run one strategy on one instance. Returns placements and RMSE trace."""
    all_pts = np.stack(
        np.meshgrid(np.arange(n), np.arange(n)), axis=-1
    ).reshape(-1, 2)
    true_flat = field.ravel()
    placements: List[Tuple[int, int]] = []
    vals: List[float] = []
    rmse_trace: List[float] = []

    for _ in range(budget):
        mean, var = rbf_regress(placements, vals, all_pts)

        mask = np.ones(len(all_pts), dtype=bool)
        for p in placements:
            mask[p[1] * n + p[0]] = False
        candidates = np.where(mask)[0]

        if strategy == "random":
            choice = np.random.choice(candidates)
        elif strategy == "greedy":
            scores = mean.copy()
            scores[~mask] = -np.inf
            choice = int(np.argmax(scores))
        elif strategy == "rl":
            scores = mean + kappa * np.sqrt(var)
            scores[~mask] = -np.inf
            choice = int(np.argmax(scores))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        pt = all_pts[choice]
        placements.append((int(pt[0]), int(pt[1])))
        vals.append(float(true_flat[choice] + 0.03 * np.random.randn()))

        mean_after, _ = rbf_regress(placements, vals, all_pts)
        rmse_trace.append(float(np.sqrt(np.mean((mean_after - true_flat) ** 2))))

    return placements, rmse_trace


def plot_figure(
    field: np.ndarray,
    results: Dict[str, np.ndarray],
    last_placements: Dict[str, List[Tuple[int, int]]],
    budget: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 5.5))

    # (a) true field
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(field, cmap="plasma", origin="lower")
    ax1.set_title("(a) True pollution concentration", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax1, fraction=0.045)
    ax1.set_xticks([]); ax1.set_yticks([])

    # (b) placements
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(field, cmap="Greys", origin="lower", alpha=0.4)
    colors = {"random": "#1565c0", "greedy": "#c62828", "rl": "#2e7d32"}
    markers = {"random": "o", "greedy": "s", "rl": "^"}
    for strat, pls in last_placements.items():
        xs_p = [p[0] for p in pls]; ys_p = [p[1] for p in pls]
        ax2.scatter(xs_p, ys_p, c=colors[strat], marker=markers[strat],
                    s=45, edgecolor="white", linewidth=0.8, label=strat.upper())
    ax2.set_title(f"(b) Sensor placements (budget = {budget})",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])

    # (c) RMSE vs budget
    ax3 = plt.subplot(1, 3, 3)
    for strat, rmses in results.items():
        mean = np.mean(rmses, axis=0)
        std = np.std(rmses, axis=0)
        x = np.arange(1, budget + 1)
        ax3.plot(x, mean, color=colors[strat], linewidth=2, label=strat.upper())
        ax3.fill_between(x, mean - std, mean + std, color=colors[strat], alpha=0.15)
    ax3.set_xlabel("Sensors deployed"); ax3.set_ylabel("Reconstruction RMSE")
    ax3.set_title("(c) Field reconstruction error (mean ± σ)",
                  fontsize=11, fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="Upper-confidence coefficient for the RL strategy")
    parser.add_argument("--seed", type=int, default=3,
                        help="Seed used to generate the ground-truth field")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig5_sensor_placement")
    args = parser.parse_args()

    set_seed(args.seed)
    field = build_field(args.grid_size)

    strategies = ["random", "greedy", "rl"]
    results: Dict[str, List[List[float]]] = {s: [] for s in strategies}
    last_placements: Dict[str, List[Tuple[int, int]]] = {}

    for strat in strategies:
        for r in range(args.replicates):
            np.random.seed(r)
            placements, rmses = run_strategy(strat, field, args.budget,
                                             args.grid_size, args.kappa)
            results[strat].append(rmses)
            if r == 0:
                last_placements[strat] = placements
    results_arr = {k: np.asarray(v) for k, v in results.items()}

    fig = plot_figure(field, results_arr, last_placements, args.budget)
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[sim03] Saved figure to {out}")
    print("[sim03] Final RMSE:", {k: float(np.mean(v[:, -1])) for k, v in results_arr.items()})


if __name__ == "__main__":
    main()
