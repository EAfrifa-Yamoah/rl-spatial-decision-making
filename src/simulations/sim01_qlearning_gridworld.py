"""
Simulation 1 — Tabular Q-learning in a gridworld.

Implements the canonical example from Section 3.3 of the manuscript.
A 12 x 12 gridworld with black-box obstacles, a goal, and step/collision costs.
An agent with tabular Q(s, a) learns a near-optimal policy via epsilon-greedy
exploration and Q-learning updates.

Produces Figure 2 of the manuscript:
  (a) the environment layout
  (b) per-episode returns and a 20-episode moving average
  (c) the learned state-value surface V(s) = max_a Q(s, a)
  (d) greedy policy arrows pi*(s) = argmax_a Q(s, a)

Run:
    python -m src.simulations.sim01_qlearning_gridworld
    python -m src.simulations.sim01_qlearning_gridworld --episodes 1000 --seed 7

Design notes:
  - Rewards:  +100 on goal, -5 for bumping obstacle, -1 for wall, -0.1 per step.
  - Exploration: exponential epsilon decay from 1.0 to 0.05 over ~120 episodes.
  - Determinism: call `set_seed(seed)` before constructing the Q-table and
    before the training loop.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.utils import moving_average, save_figure, set_seed

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
State = Tuple[int, int]
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
N_ACTIONS = 4


@dataclass
class GridWorld:
    """A square grid with obstacles, a single start and a single goal."""

    size: int = 12
    start: State = (0, 0)
    goal: State = (11, 11)
    obstacles: Set[State] = field(default_factory=set)

    @classmethod
    def default(cls) -> "GridWorld":
        """The canonical 12x12 layout used in Figure 2 of the manuscript."""
        obs: Set[State] = set()
        for i in range(3, 8):
            obs.add((i, 4))
        for i in range(5, 10):
            obs.add((7, i))
        for i in range(2, 6):
            obs.add((10, i))
        for i in range(0, 3):
            obs.add((4, i + 7))
        return cls(size=12, start=(0, 0), goal=(11, 11), obstacles=obs)

    def step(self, state: State, action: int) -> Tuple[State, float, bool]:
        """Apply an action. Returns (next_state, reward, done).

        Wall bumps and obstacle collisions leave the agent in place but
        incur a cost; reaching the goal terminates the episode.
        """
        r, c = state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            return state, -1.0, False
        if (nr, nc) in self.obstacles:
            return state, -5.0, False
        if (nr, nc) == self.goal:
            return (nr, nc), 100.0, True
        return (nr, nc), -0.1, False


# ----------------------------------------------------------------------
# Q-learning
# ----------------------------------------------------------------------
def train_qlearning(
    env: GridWorld,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_tau: float = 120.0,
    max_steps: int = 200,
) -> Tuple[np.ndarray, List[float], List[int]]:
    """Train a tabular Q-learning agent.

    Returns
    -------
    Q : (size, size, N_ACTIONS) ndarray
    returns_history : per-episode cumulative reward
    steps_history : per-episode step count
    """
    Q = np.zeros((env.size, env.size, N_ACTIONS))
    returns_history: List[float] = []
    steps_history: List[int] = []

    for ep in range(n_episodes):
        eps = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay_tau)
        state = env.start
        total_r = 0.0
        t = 0
        for t in range(max_steps):
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[state[0], state[1]]))
            ns, r, done = env.step(state, a)
            total_r += r
            td_target = r + gamma * (0 if done else np.max(Q[ns[0], ns[1]]))
            Q[state[0], state[1], a] += alpha * (td_target - Q[state[0], state[1], a])
            state = ns
            if done:
                break
        returns_history.append(total_r)
        steps_history.append(t + 1)
    return Q, returns_history, steps_history


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_figure(
    env: GridWorld, Q: np.ndarray, returns_history: List[float]
) -> plt.Figure:
    """Produce the four-panel Figure 2."""
    V = np.max(Q, axis=2)
    policy = np.argmax(Q, axis=2)
    V_disp = V.copy()
    for r, c in env.obstacles:
        V_disp[r, c] = np.nan

    env_grid = np.zeros((env.size, env.size))
    for r, c in env.obstacles:
        env_grid[r, c] = 1

    fig = plt.figure(figsize=(13, 10))

    # (a) environment
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("(a) Gridworld environment", fontsize=12, fontweight="bold")
    ax1.imshow(env_grid, cmap="Greys", origin="lower", vmin=0, vmax=1)
    ax1.plot(env.start[1], env.start[0], "o", color="#2e7d32", markersize=16,
             markeredgecolor="white", markeredgewidth=2, label="Start")
    ax1.plot(env.goal[1], env.goal[0], "*", color="#c62828", markersize=22,
             markeredgecolor="white", markeredgewidth=1.5, label="Goal")
    ax1.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax1.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(loc="upper left", framealpha=0.9)

    # (b) learning curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("(b) Learning dynamics", fontsize=12, fontweight="bold")
    ax2.plot(returns_history, color="#bbdefb", alpha=0.7, linewidth=0.8,
             label="Per-episode return")
    ma = moving_average(returns_history, 20)
    ax2.plot(range(len(ma)), ma, color="#1565c0", linewidth=2.2,
             label="20-episode moving avg")
    ax2.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Cumulative reward")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    # (c) value surface
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("(c) Learned state-value V(s)", fontsize=12, fontweight="bold")
    im = ax3.imshow(V_disp, cmap="viridis", origin="lower")
    for r, c in env.obstacles:
        ax3.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#222222"))
    ax3.plot(env.start[1], env.start[0], "o", color="#81c784", markersize=12,
             markeredgecolor="white", markeredgewidth=1.5)
    ax3.plot(env.goal[1], env.goal[0], "*", color="#ef9a9a", markersize=16,
             markeredgecolor="white", markeredgewidth=1.5)
    ax3.set_xticks([]); ax3.set_yticks([])
    cbar = plt.colorbar(im, ax=ax3, fraction=0.045)
    cbar.set_label("Value")

    # (d) policy arrows
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("(d) Greedy policy π*(s)", fontsize=12, fontweight="bold")
    ax4.imshow(env_grid, cmap="Greys", origin="lower", vmin=0, vmax=1, alpha=0.3)
    # row-1 in data coords is DOWN in plot; row+1 is UP
    arrow_dirs = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles or (r, c) == env.goal:
                continue
            a = policy[r, c]
            dx, dy = arrow_dirs[a]
            ax4.arrow(c, r, dx, dy, head_width=0.18, head_length=0.12,
                      fc="#1565c0", ec="#1565c0", length_includes_head=True,
                      linewidth=0.8)
    for r, c in env.obstacles:
        ax4.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#222222"))
    ax4.plot(env.start[1], env.start[0], "o", color="#2e7d32", markersize=14,
             markeredgecolor="white", markeredgewidth=1.5)
    ax4.plot(env.goal[1], env.goal[0], "*", color="#c62828", markersize=18,
             markeredgecolor="white", markeredgewidth=1.5)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_xlim(-0.6, env.size - 0.4); ax4.set_ylim(-0.6, env.size - 0.4)

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig3_qlearning_grid")
    args = parser.parse_args()

    set_seed(args.seed)
    env = GridWorld.default()
    Q, returns_history, steps_history = train_qlearning(
        env, n_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma
    )

    fig = plot_figure(env, Q, returns_history)
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)

    print(f"[sim01] Saved figure to {out}")
    print(f"[sim01] Final avg return (last 50 ep): "
          f"{np.mean(returns_history[-50:]):.2f}")
    print(f"[sim01] Final avg steps  (last 50 ep): "
          f"{np.mean(steps_history[-50:]):.2f}")


if __name__ == "__main__":
    main()
