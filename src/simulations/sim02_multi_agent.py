"""
Simulation 2 — Independent multi-agent Q-learning.

Two agents with independent Q-functions navigate a shared 10x10 grid
with obstacles, static walls, and collision penalties that couple their
learning. This reproduces Figure 4 of the manuscript (Section 6.2) and
illustrates how decentralized learners can still converge on emergent
coordination when rewards are carefully designed.

Rewards:
  +100   on reaching own goal
  -10    on collision (same-cell or position swap with the other agent)
  -2     on bumping a wall
  -0.2   per step otherwise

Run:
    python -m src.simulations.sim02_multi_agent
    python -m src.simulations.sim02_multi_agent --episodes 1200 --seed 11
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import moving_average, save_figure, set_seed

State = Tuple[int, int]
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
N_ACTIONS = 5  # up, down, left, right, stay


@dataclass
class MultiAgentGrid:
    """Two-agent grid with obstacles."""

    size: int = 10
    obstacles: Set[State] = field(default_factory=set)
    starts: List[State] = field(default_factory=lambda: [(0, 0), (9, 9)])
    goals: List[State] = field(default_factory=lambda: [(9, 0), (0, 9)])

    @classmethod
    def default(cls) -> "MultiAgentGrid":
        obs = {(3, 3), (3, 4), (3, 5), (6, 2), (6, 3),
               (6, 6), (6, 7), (2, 7), (8, 4)}
        return cls(size=10, obstacles=obs,
                   starts=[(0, 0), (9, 9)], goals=[(9, 0), (0, 9)])

    def valid(self, s: State) -> bool:
        r, c = s
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False
        return s not in self.obstacles

    def try_move(self, s: State, a: int) -> State:
        """Return the tentative next state for a single agent."""
        dr, dc = ACTIONS[a]
        ns = (s[0] + dr, s[1] + dc)
        if not self.valid(ns):
            return s
        return ns


def train_independent_qlearners(
    env: MultiAgentGrid,
    n_episodes: int = 800,
    alpha: float = 0.15,
    gamma: float = 0.95,
    max_steps: int = 60,
) -> Tuple[List[np.ndarray], List[float], List, List]:
    """Train two independent tabular Q-learners.

    Returns
    -------
    Q       : list of two (size, size, N_ACTIONS) ndarrays
    joint_returns : per-episode summed reward across both agents
    traj_early, traj_late : trajectory snapshots (episode 20 and final episode)
    """
    Q = [np.zeros((env.size, env.size, N_ACTIONS)) for _ in range(2)]
    joint_returns: List[float] = []
    traj_early: List = []
    traj_late: List = []

    for ep in range(n_episodes):
        eps = max(0.05, 1.0 - ep / 400)
        states = list(env.starts)
        traj = [list(states)]
        total_r = 0.0
        done_flags = [False, False]

        for _ in range(max_steps):
            actions = []
            for i in range(2):
                if done_flags[i]:
                    actions.append(4)  # stay
                    continue
                if np.random.rand() < eps:
                    actions.append(np.random.randint(N_ACTIONS))
                else:
                    actions.append(int(np.argmax(Q[i][states[i][0], states[i][1]])))

            new_states = [env.try_move(states[i], actions[i]) for i in range(2)]
            collision = (
                new_states[0] == new_states[1]
                or (new_states[0] == states[1] and new_states[1] == states[0])
            )

            rewards = [0.0, 0.0]
            for i in range(2):
                if done_flags[i]:
                    continue
                if collision:
                    new_states[i] = states[i]
                    rewards[i] -= 10
                if new_states[i] == env.goals[i]:
                    rewards[i] += 100
                elif (new_states[i] == states[i]
                      and actions[i] != 4 and not collision):
                    rewards[i] -= 2
                else:
                    rewards[i] -= 0.2

            for i in range(2):
                if done_flags[i]:
                    continue
                s = states[i]
                a = actions[i]
                ns = new_states[i]
                done_i = ns == env.goals[i]
                target = rewards[i] + gamma * (
                    0 if done_i else np.max(Q[i][ns[0], ns[1]])
                )
                Q[i][s[0], s[1], a] += alpha * (target - Q[i][s[0], s[1], a])
                if done_i:
                    done_flags[i] = True

            states = new_states
            traj.append(list(states))
            total_r += sum(rewards)
            if all(done_flags):
                break

        joint_returns.append(total_r)
        if ep == 20:
            traj_early = traj
        if ep == n_episodes - 1:
            traj_late = traj

    return Q, joint_returns, traj_early, traj_late


def _draw_env(ax, env: MultiAgentGrid, title: str) -> None:
    grid = np.zeros((env.size, env.size))
    for r, c in env.obstacles:
        grid[r, c] = 1
    ax.imshow(grid, cmap="Greys", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i, col in enumerate(["#1565c0", "#c62828"]):
        ax.plot(env.starts[i][1], env.starts[i][0], "o",
                color=col, markersize=12, markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(env.goals[i][1], env.goals[i][0], "*",
                color=col, markersize=16, markeredgecolor="white", markeredgewidth=1.5)


def _plot_traj(ax, traj, label_prefix: str = "") -> None:
    cols = ["#1565c0", "#c62828"]
    for i in range(2):
        xs = [s[i][1] + 0.05 * (-1) ** i for s in traj]
        ys = [s[i][0] + 0.05 * (-1) ** i for s in traj]
        ax.plot(xs, ys, "-", color=cols[i], linewidth=2.2, alpha=0.85,
                label=f"{label_prefix}Agent {i + 1}")


def plot_figure(
    env: MultiAgentGrid,
    traj_early,
    traj_late,
    joint_returns: List[float],
    final_ep: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    _draw_env(axes[0], env, "(a) Early training (episode 20)")
    _plot_traj(axes[0], traj_early)
    axes[0].legend(loc="upper right", fontsize=9)

    _draw_env(axes[1], env, f"(b) After learning (episode {final_ep})")
    _plot_traj(axes[1], traj_late)
    axes[1].legend(loc="upper right", fontsize=9)

    ax2 = axes[2]
    ax2.plot(joint_returns, color="#bbdefb", alpha=0.6, linewidth=0.7,
             label="Episode joint return")
    ma = moving_average(joint_returns, 30)
    ax2.plot(range(len(ma)), ma, color="#0d47a1", linewidth=2,
             label="30-ep moving avg")
    ax2.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Joint cumulative reward")
    ax2.set_title("(c) Joint learning dynamics", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig4_multi_agent")
    args = parser.parse_args()

    set_seed(args.seed)
    env = MultiAgentGrid.default()
    Q, joint_returns, traj_early, traj_late = train_independent_qlearners(
        env, n_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma
    )

    fig = plot_figure(env, traj_early, traj_late, joint_returns, args.episodes)
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[sim02] Saved figure to {out}")
    print(f"[sim02] Final avg joint return (last 50 ep): "
          f"{np.mean(joint_returns[-50:]):.2f}")


if __name__ == "__main__":
    main()
