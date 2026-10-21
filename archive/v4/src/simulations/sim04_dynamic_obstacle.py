"""
Simulation 4 — SARSA with a dynamic obstacle.

An agent must navigate an 8x8 grid to reach a fixed goal while a moving
obstacle patrols horizontally along a fixed row. State is augmented with
the obstacle's column and direction so that the problem remains Markov.

Reproduces Figure 6 of the manuscript (Section 6.3). Illustrates how a
dynamic, time-varying environment is handled by expanding the state space
rather than modifying the algorithm.

Reward structure:
  +100   on reaching goal
  -20    on collision with the moving obstacle (terminal crash)
  -0.5   per step otherwise

Run:
    python -m src.simulations.sim04_dynamic_obstacle
    python -m src.simulations.sim04_dynamic_obstacle --episodes 6000 --seed 1
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_figure, set_seed

ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4


class DynamicObstacleEnv:
    """8x8 gridworld with a patrolling obstacle."""

    def __init__(self, size: int = 8, obs_row: int = 4):
        self.size = size
        self.obs_row = obs_row
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

    def state_to_idx(self, s: Tuple[int, int, int, int]) -> int:
        ar, ac, oc, d = s
        return ((ar * self.size + ac) * self.size + oc) * 2 + d

    @property
    def n_states(self) -> int:
        return self.size * self.size * self.size * 2

    def step_obstacle(self, oc: int, d: int) -> Tuple[int, int]:
        noc = oc + (1 if d == 1 else -1)
        nd = d
        if noc < 0 or noc >= self.size:
            nd = 1 - d
            noc = oc + (1 if nd == 1 else -1)
        return noc, nd

    def step(self, s, a):
        ar, ac, oc, d = s
        dr, dc = ACTIONS[a]
        nar, nac = ar + dr, ac + dc
        noc, nd = self.step_obstacle(oc, d)
        if nar < 0 or nar >= self.size or nac < 0 or nac >= self.size:
            nar, nac = ar, ac
        if nar == self.obs_row and nac == noc:
            return (nar, nac, noc, nd), -20, True
        if (nar, nac) == self.goal:
            return (nar, nac, noc, nd), 100, True
        return (nar, nac, noc, nd), -0.5, False


def train_sarsa(
    env: DynamicObstacleEnv,
    n_episodes: int = 4000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    max_steps: int = 80,
) -> Tuple[np.ndarray, List[float]]:
    Q = np.zeros((env.n_states, N_ACTIONS))
    recent: List[int] = []
    success_window: List[float] = []

    for ep in range(n_episodes):
        eps = max(0.05, 1.0 - ep / (n_episodes / 2))
        s = (env.start[0], env.start[1],
             np.random.randint(env.size), np.random.randint(2))
        idx = env.state_to_idx(s)
        a = (np.random.randint(N_ACTIONS) if np.random.rand() < eps
             else int(np.argmax(Q[idx])))

        success = 0
        for _ in range(max_steps):
            ns, r, done = env.step(s, a)
            if done and r > 0:
                success = 1
            ni = env.state_to_idx(ns)
            na = (np.random.randint(N_ACTIONS) if np.random.rand() < eps
                  else int(np.argmax(Q[ni])))
            target = r + (0 if done else gamma * Q[ni, na])
            Q[idx, a] += alpha * (target - Q[idx, a])
            s, idx, a = ns, ni, na
            if done:
                break

        recent.append(success)
        if len(recent) > 100:
            recent.pop(0)
        success_window.append(float(np.mean(recent)))

    return Q, success_window


def rollout_policy(env: DynamicObstacleEnv, Q: np.ndarray, max_steps: int = 30):
    """Greedy rollout of the learned policy. Returns state trajectory."""
    np.random.seed(0)
    s = (env.start[0], env.start[1], 0, 1)
    history = [s]
    for _ in range(max_steps):
        idx = env.state_to_idx(s)
        a = int(np.argmax(Q[idx]))
        ns, _, done = env.step(s, a)
        history.append(ns)
        s = ns
        if done:
            break
    return history


def plot_figure(
    env: DynamicObstacleEnv,
    history: list,
    success_window: List[float],
    snap_times: Tuple[int, int, int, int] = (0, 6, 12, 20),
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.3, 1.6])

    def draw_state(ax, hist_slice, current_s, title):
        env_grid = np.zeros((env.size, env.size))
        ax.imshow(env_grid, cmap="Greys", origin="lower", vmin=0, vmax=1, alpha=0.0)
        ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        for c in range(env.size):
            rect = mpatches.Rectangle((c - 0.5, env.obs_row - 0.5), 1, 1,
                                       facecolor="#fff3e0",
                                       edgecolor="#ffcc80", linewidth=0.3)
            ax.add_patch(rect)
        ax.plot(env.start[1], env.start[0], "o", color="#2e7d32", markersize=13,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(env.goal[1], env.goal[0], "*", color="#c62828", markersize=17,
                markeredgecolor="white", markeredgewidth=1.5)
        if len(hist_slice) > 1:
            xs_t = [h[1] for h in hist_slice]
            ys_t = [h[0] for h in hist_slice]
            ax.plot(xs_t, ys_t, "-", color="#1565c0", linewidth=2, alpha=0.7)
        ax.plot(current_s[1], current_s[0], "o", color="#1565c0",
                markersize=13, markeredgecolor="white", markeredgewidth=1.5)
        oc = current_s[2]
        ax.plot(oc, env.obs_row, "s", color="#d84315", markersize=15,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlim(-0.5, env.size - 0.5); ax.set_ylim(-0.5, env.size - 0.5)

    for i, t in enumerate(snap_times):
        ax = fig.add_subplot(gs[0, i])
        tt = min(t, len(history) - 1)
        draw_state(ax, history[:tt + 1], history[tt], f"t = {tt}")

    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.plot(success_window, color="#0d47a1", linewidth=1.8)
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Rolling success rate (last 100 episodes)")
    ax_curve.set_title(
        "Learning to avoid a moving obstacle — success rate over training",
        fontsize=11, fontweight="bold",
    )
    ax_curve.axhline(0.95, color="green", linestyle="--", linewidth=1,
                     alpha=0.7, label="95% target")
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.grid(alpha=0.3); ax_curve.legend(loc="lower right")

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig6_dynamic_obstacle")
    args = parser.parse_args()

    set_seed(args.seed)
    env = DynamicObstacleEnv()
    Q, success_window = train_sarsa(
        env, n_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma
    )
    history = rollout_policy(env, Q)

    fig = plot_figure(env, history, success_window)
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)

    print(f"[sim04] Saved figure to {out}")
    print(f"[sim04] Final success rate: {success_window[-1]:.2%}")


if __name__ == "__main__":
    main()
