"""Problem A: single agent navigation on a 12 x 12 gridworld (Q-learning).

Two factorial ablations, 30 independent training seeds each:

  A1  reward specification, algorithm and state fixed
      (tabular Q-learning, alpha = 0.1 fixed a priori for all rewards)
  A2  state representation, algorithm and reward fixed
      (Q-learning with the shaped reward; learning rate tuned separately for
      every representation on 20 tuning seeds that are disjoint from the 30
      reporting seeds, which removes the learning-rate confound of v5)

Coordinates are (row, col) with row 0 at the bottom of every figure.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.utils import rng, argmax_random

GRID = 12
START = (0, 0)
GOAL = (11, 11)
OBSTACLES = frozenset({
    (2, 1), (2, 2), (2, 3), (2, 4),
    (4, 3), (4, 4), (4, 5), (4, 6),
    (6, 5), (6, 6),
    (8, 2), (8, 3), (8, 8), (8, 9),
    (10, 4), (10, 5),
})
ACTIONS = ((1, 0), (-1, 0), (0, -1), (0, 1))  # up, down, left, right
AWAY_ORDER = (1, 0, 2, 3)  # tie-break order whose first action points away from the goal
GAMMA = 0.95
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.99
MAX_STEPS = 200
BUDGET = 500
EVAL_EVERY = 10
N_SEEDS = 30
TUNE_SEEDS = range(100, 120)
ALPHA_A1 = 0.1
ALPHA_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

FREE = [(r, c) for r in range(GRID) for c in range(GRID)
        if (r, c) not in OBSTACLES and (r, c) != GOAL]


def step(pos, a):
    dr, dc = ACTIONS[a]
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        return pos, "wall"
    if (nr, nc) in OBSTACLES:
        return pos, "obstacle"
    if (nr, nc) == GOAL:
        return (nr, nc), "goal"
    return (nr, nc), "move"


def manhattan(p):
    return abs(p[0] - GOAL[0]) + abs(p[1] - GOAL[1])


# ------------------------------------------------------------------ rewards
def make_reward(name: str, seed: int):
    """Return r(prev, new, event). All rewards pay +100 on reaching the goal."""
    if name == "sparse":
        return lambda prev, new, ev: 100.0 if ev == "goal" else 0.0
    if name == "step_cost":
        return lambda prev, new, ev: 100.0 if ev == "goal" else -0.1
    if name == "shaped":
        table = {"goal": 100.0, "obstacle": -5.0, "wall": -1.0, "move": -0.1}
        return lambda prev, new, ev: table[ev]
    if name == "pbrs":
        # potential-based shaping (Ng et al., 1999), phi = -Manhattan distance;
        # phi(goal) = 0 so the shaping term vanishes at termination
        return lambda prev, new, ev: ((100.0 if ev == "goal" else 0.0)
                                      + GAMMA * (-manhattan(new)) - (-manhattan(prev)))
    if name == "random_dense":
        g = rng("A", 9, seed)
        noise = g.uniform(-1.0, 1.0, size=(GRID, GRID))
        noise -= noise.mean()
        return lambda prev, new, ev: (100.0 if ev == "goal" else 0.0) + float(noise[new])
    raise ValueError(name)


REWARDS = ("sparse", "step_cost", "shaped", "pbrs", "random_dense")


# ------------------------------------------------------- representations
def _obstacle_or_wall(r, c):
    return not (0 <= r < GRID and 0 <= c < GRID) or (r, c) in OBSTACLES


def feat_onehot(p):
    v = np.zeros(GRID * GRID)
    v[p[0] * GRID + p[1]] = 1.0
    return v


def feat_rowcol(p):
    v = np.zeros(2 * GRID + 1)
    v[p[0]] = 1.0
    v[GRID + p[1]] = 1.0
    v[-1] = 1.0
    return v


def feat_window(p):
    """3 x 3 occupancy window around the agent (8 cells), goal offset, bias."""
    r, c = p
    occ = [float(_obstacle_or_wall(r + dr, c + dc))
           for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    return np.array(occ + [(GOAL[0] - r) / GRID, (GOAL[1] - c) / GRID, 1.0])


def feat_hand(p):
    """Signed and absolute row and column offsets to the goal, and distance to
    the nearest obstacle or edge in each compass direction, plus bias (9 values)."""
    r, c = p
    dists = []
    for dr, dc in ACTIONS:
        k = 1
        while not _obstacle_or_wall(r + k * dr, c + k * dc):
            k += 1
        dists.append(k / GRID)
    dr_g, dc_g = (GOAL[0] - r) / GRID, (GOAL[1] - c) / GRID
    return np.array([dr_g, dc_g, abs(dr_g), abs(dc_g)] + dists + [1.0])


REPRESENTATIONS = {
    "tabular": feat_onehot,
    "rowcol": feat_rowcol,
    "window": feat_window,
    "hand": feat_hand,
}
FEATURE_CACHE = {name: {p: f(p) for p in FREE + [GOAL]} for name, f in REPRESENTATIONS.items()}


# ---------------------------------------------------------------- learning
def greedy_reaches(qfun, start) -> bool:
    pos = start
    for _ in range(MAX_STEPS):
        q = qfun(pos)
        a = int(np.argmax(q))  # deterministic tie-break at evaluation
        pos, ev = step(pos, a)
        if ev == "goal":
            return True
    return False


def run(seed: int, reward: str = "shaped", rep: str = "tabular",
        alpha: float = ALPHA_A1, budget: int = BUDGET, cond_id: int = 0,
        tiebreak: str = "random") -> dict:
    """Train one Q-learning agent; return learning and evaluation metrics."""
    g = rng("A", cond_id, seed)
    rfun = make_reward(reward, seed)
    feats = FEATURE_CACHE[rep]
    dim = len(feats[START])
    W = np.zeros((dim, len(ACTIONS)))

    def q(p):
        return feats[p] @ W

    eps = EPS_START
    first_goal = None
    diverged = False
    returns, greedy_curve = [], []
    for ep in range(budget):
        pos, total = START, 0.0
        for _ in range(MAX_STEPS):
            x = feats[pos]
            qs = x @ W
            if not np.all(np.isfinite(qs)) or np.abs(qs).max() > 1e6:
                diverged = True
                break
            if g.random() < eps:
                a = int(g.integers(len(ACTIONS)))
            elif tiebreak == "random":
                a = argmax_random(qs.tolist(), g)
            elif tiebreak == "first_toward":  # first index; action 0 (up) points toward the goal
                a = int(np.argmax(qs))
            else:  # "first_away": first index in the order (down, up, left, right), as in v5
                a = AWAY_ORDER[int(np.argmax(qs[list(AWAY_ORDER)]))]
            new, ev = step(pos, a)
            r = rfun(pos, new, ev)
            total += r
            target = r if ev == "goal" else r + GAMMA * float(np.max(feats[new] @ W))
            W[:, a] += alpha * (target - qs[a]) * x
            pos = new
            if ev == "goal":
                if first_goal is None:
                    first_goal = ep + 1  # 1-based episode index
                break
        if diverged:
            break
        returns.append(total)
        if (ep + 1) % EVAL_EVERY == 0:
            greedy_curve.append(greedy_reaches(q, START))
        eps = max(EPS_END, eps * EPS_DECAY)

    if diverged:  # weights blew up: count as failure everywhere
        W[:] = 0.0
        multistart, from_start = 0.0, False
    else:
        multistart = float(np.mean([greedy_reaches(q, s) for s in FREE]))
        from_start = greedy_reaches(q, START)
    out = {
        "seed": seed, "reward": reward, "rep": rep, "alpha": alpha, "diverged": diverged,
        "tiebreak": tiebreak,
        "first_goal": first_goal,
        "eval_from_start": from_start,
        "eval_multistart": multistart,
        "returns": returns,
        "greedy_curve": greedy_curve,
    }
    if rep == "tabular" and seed == 0:
        V = np.full((GRID, GRID), np.nan)
        policy = {}
        for p in FREE:
            qs = q(p)
            V[p] = float(np.max(qs))
            policy[f"{p[0]},{p[1]}"] = int(np.argmax(qs))
        V[GOAL] = 0.0
        out["value_map"] = V.tolist()
        out["policy"] = policy
    return out


def _job(args):
    return run(**args)


def main(out_dir: Path, workers: int = 2):
    jobs = []
    # Ablation A1: reward specification
    for ci, rw in enumerate(REWARDS):
        for s in range(N_SEEDS):
            jobs.append(dict(seed=s, reward=rw, rep="tabular", alpha=ALPHA_A1, cond_id=10 + ci))
    # Sensitivity: deterministic first-index tie-breaking during training, with the
    # first action pointing toward the goal or away from it (v5 used the latter)
    for tb in ("first_toward", "first_away"):
        for ci, rw in enumerate(REWARDS):
            for s in range(N_SEEDS):
                jobs.append(dict(seed=s, reward=rw, rep="tabular", alpha=ALPHA_A1, cond_id=10 + ci,
                                 tiebreak=tb))
    # Ablation A2 tuning: every representation over the alpha grid on tuning seeds
    tune_jobs = []
    for ri, rep in enumerate(REPRESENTATIONS):
        for ai, al in enumerate(ALPHA_GRID):
            for s in TUNE_SEEDS:
                tune_jobs.append(dict(seed=s, reward="shaped", rep=rep, alpha=al,
                                      cond_id=100 + 10 * ri + ai))
    with Pool(workers) as pool:
        a1_all = pool.map(_job, jobs, chunksize=2)
        tune = pool.map(_job, tune_jobs, chunksize=2)
    a1 = [r for r in a1_all if r["tiebreak"] == "random"]
    a1_first = [r for r in a1_all if r["tiebreak"] != "random"]
    best_alpha = {}
    tuning_table = {}
    for rep in REPRESENTATIONS:
        scores = {}
        for al in ALPHA_GRID:
            vals = [t["eval_multistart"] for t in tune if t["rep"] == rep and t["alpha"] == al]
            scores[al] = float(np.mean(vals))
        tuning_table[rep] = {str(k): v for k, v in scores.items()}
        best = max(scores.values())
        best_alpha[rep] = min(al for al, v in scores.items() if v == best)  # smallest alpha among ties
    jobs2 = []
    for ri, rep in enumerate(REPRESENTATIONS):
        for s in range(N_SEEDS):
            jobs2.append(dict(seed=s, reward="shaped", rep=rep, alpha=best_alpha[rep],
                              cond_id=200 + ri))
    with Pool(workers) as pool:
        a2 = pool.map(_job, jobs2, chunksize=2)
    res = {
        "config": {"grid": GRID, "start": START, "goal": GOAL, "obstacles": sorted(OBSTACLES),
                   "gamma": GAMMA, "eps": [EPS_START, EPS_END, EPS_DECAY], "max_steps": MAX_STEPS,
                   "budget": BUDGET, "alpha_A1": ALPHA_A1, "alpha_grid": ALPHA_GRID,
                   "tune_seeds": list(TUNE_SEEDS), "n_seeds": N_SEEDS, "n_free_cells": len(FREE)},
        "A1": a1, "A1_tiebreak_first": a1_first, "A2_tuning": tuning_table, "A2_best_alpha": best_alpha, "A2": a2,
    }
    (out_dir / "problem_a.json").write_text(json.dumps(res))
    return res
