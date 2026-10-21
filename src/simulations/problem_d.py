"""Problem D: crossing a patrolling obstacle (SARSA), held-out geography.

An 8 x 8 grid; the agent starts at (0, 0) and must reach (7, 7). One
obstacle patrols a single row, bouncing between the left and right edges.
Moving into the obstacle's cell, or swapping cells with it, is a collision:
the episode ends with -20 (failure). Reaching the goal ends it with +100.
Every step costs -0.5. Episodes time out after 80 steps (failure).

2 x 2 design, 30 SARSA seeds per cell:
  state     row-blind: (agent row, agent col, obstacle col, direction), 1,024 states
            row-aware: adds the obstacle row, 8,192 states
  training  fixed: obstacle always on row 4
            varied: obstacle row drawn uniformly from {2, 3, 4} each episode

Greedy evaluation is exhaustive: for each obstacle row 1-6 every one of the
16 initial (column, direction) configurations is rolled out once, so each
seed's success rate per row is exact. A compositional test places two
obstacles at once (rows 2 and 4, both inside the varied training range);
the state then reports the obstacle on the nearest row at or above the
agent (the next one to cross).
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.utils import rng, argmax_random

GRID = 8
START, GOAL = (0, 0), (7, 7)
ACTIONS = ((1, 0), (-1, 0), (0, -1), (0, 1))  # up, down, left, right
GAMMA, ALPHA = 0.95, 0.1
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.9995
MAX_STEPS = 80
BUDGET = 6000
GOAL_R, COLLISION_R, STEP_R = 100.0, -20.0, -0.5
N_SEEDS = 30
TEST_ROWS = (1, 2, 3, 4, 5, 6)
TRAIN_ROWS = {"fixed": (4,), "varied": (2, 3, 4)}
STATES = ("blind", "aware")
TWO_OBSTACLE_ROWS = (2, 4)


def bounce(col, d):
    """d = 0 moving right, 1 moving left."""
    if d == 0:
        return (col - 1, 1) if col == GRID - 1 else (col + 1, 0)
    return (col + 1, 0) if col == 0 else (col - 1, 1)


def sid(state_kind, agent, obs):
    """obs = (row, col, dir) of the observed obstacle."""
    r, c = agent
    orow, ocol, od = obs
    if state_kind == "blind":
        return ((r * GRID + c) * GRID + ocol) * 2 + od
    return (((r * GRID + c) * GRID + orow) * GRID + ocol) * 2 + od


def n_states(state_kind):
    return GRID ** 3 * 2 if state_kind == "blind" else GRID ** 4 * 2


def observed(agent, obstacles):
    """Obstacle on the nearest row at or above the agent, else the last one."""
    ahead = [o for o in obstacles if o[0] >= agent[0]]
    if ahead:
        return min(ahead, key=lambda o: o[0])
    return max(obstacles, key=lambda o: o[0])


def env_step(agent, a, obstacles):
    """Return (agent', obstacles', reward, done, success)."""
    dr, dc = ACTIONS[a]
    nr, nc = agent[0] + dr, agent[1] + dc
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        nr, nc = agent
    new_obs = []
    collided = False
    for (orow, ocol, od) in obstacles:
        ncol, nd = bounce(ocol, od)
        new_obs.append((orow, ncol, nd))
        if (nr, nc) == (orow, ncol):
            collided = True
        if agent == (orow, ncol) and (nr, nc) == (orow, ocol):  # swap
            collided = True
    if collided:
        return (nr, nc), new_obs, STEP_R + COLLISION_R, True, False
    if (nr, nc) == GOAL:
        return (nr, nc), new_obs, STEP_R + GOAL_R, True, True
    return (nr, nc), new_obs, STEP_R, False, False


def greedy_success(Q, state_kind, obstacles, record=False):
    agent = START
    traj = [(agent, list(obstacles))]
    for _ in range(MAX_STEPS):
        s = sid(state_kind, agent, observed(agent, obstacles))
        a = int(np.argmax(Q[s]))
        agent, obstacles, r, done, ok = env_step(agent, a, obstacles)
        if record:
            traj.append((agent, list(obstacles)))
        if done:
            return (ok, traj) if record else ok
    return (False, traj) if record else False


def evaluate(Q, state_kind):
    per_row = {}
    for row in TEST_ROWS:
        res = [greedy_success(Q, state_kind, [(row, c, d)]) for c in range(GRID) for d in (0, 1)]
        per_row[str(row)] = float(np.mean(res))
    two = []
    for c1 in range(GRID):
        for d1 in (0, 1):
            for c2 in range(GRID):
                for d2 in (0, 1):
                    two.append(greedy_success(Q, state_kind, [(TWO_OBSTACLE_ROWS[0], c1, d1),
                                                              (TWO_OBSTACLE_ROWS[1], c2, d2)]))
    return per_row, float(np.mean(two))


def run(seed: int, state_kind: str, regime: str) -> dict:
    ci = STATES.index(state_kind) * 2 + list(TRAIN_ROWS).index(regime)
    g = rng("D", ci, seed)
    g_env = rng("D", 50 + list(TRAIN_ROWS).index(regime), seed)  # same episode sequence for both states
    Q = np.zeros((n_states(state_kind), len(ACTIONS)))
    rows = TRAIN_ROWS[regime]
    eps = EPS_START
    succ = []

    def choose(s):
        if g.random() < eps:
            return int(g.integers(len(ACTIONS)))
        return argmax_random(Q[s].tolist(), g)

    for ep in range(BUDGET):
        row = rows[int(g_env.integers(len(rows)))]
        obstacles = [(row, int(g_env.integers(GRID)), int(g_env.integers(2)))]
        agent = START
        s = sid(state_kind, agent, observed(agent, obstacles))
        a = choose(s)
        ok = False
        for _ in range(MAX_STEPS):
            agent, obstacles, r, done, ok = env_step(agent, a, obstacles)
            if done:
                Q[s, a] += ALPHA * (r - Q[s, a])
                break
            s2 = sid(state_kind, agent, observed(agent, obstacles))
            a2 = choose(s2)
            Q[s, a] += ALPHA * (r + GAMMA * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
        succ.append(bool(ok))
        eps = max(EPS_END, eps * EPS_DECAY)
    per_row, two = evaluate(Q, state_kind)
    out = {"seed": seed, "state": state_kind, "regime": regime,
           "train_success": succ, "eval_rows": per_row, "eval_two_obstacles": two}
    if seed == 0:
        ok, traj = greedy_success(Q, state_kind, [(4, 3, 0)], record=True)
        out["rollout_row4"] = {"success": ok,
                               "traj": [[list(a_), [list(o) for o in obs]] for a_, obs in traj]}
    return out


def _job(args):
    return run(*args)


def main(out_dir: Path, workers: int = 2):
    jobs = [(s, st, rg) for st in STATES for rg in TRAIN_ROWS for s in range(N_SEEDS)]
    with Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=2)
    out = {"config": {"grid": GRID, "start": START, "goal": GOAL, "rewards": [GOAL_R, COLLISION_R, STEP_R],
                      "gamma": GAMMA, "alpha": ALPHA, "eps": [EPS_START, EPS_END, EPS_DECAY],
                      "max_steps": MAX_STEPS, "budget": BUDGET, "test_rows": TEST_ROWS,
                      "train_rows": TRAIN_ROWS, "two_obstacle_rows": TWO_OBSTACLE_ROWS,
                      "n_seeds": N_SEEDS},
           "runs": res}
    (out_dir / "problem_d.json").write_text(json.dumps(out))
    return out
