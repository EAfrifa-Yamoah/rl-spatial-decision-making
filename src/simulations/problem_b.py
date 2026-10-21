"""Problem B: two agents swapping sides through a one-lane corridor.

Redesign of the v5 map, whose shortest paths were disjoint. Here the only
route between the two rooms is a three-cell, one-lane corridor, the agents
start in opposite rooms with head-on goals, so at least one agent must yield.

Three tabular learners with identical reward, budget and exploration:
  iql_own    independent Q-learners, state = own position only
  iql_joint  independent Q-learners, state = (own position, other position)
  jal        one centralised joint-action learner over (pos1, pos2) and the
             25 joint actions, trained on the summed reward

Contrast iql_own vs iql_joint isolates the state specification with the
algorithm fixed; iql_joint vs jal isolates the learning architecture with
the state fixed. Evaluation is greedy (no exploration) from 50 start pairs
per seed that are drawn once and shared by the three learners.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.utils import rng, argmax_random

ROWS, COLS = 5, 11
CORRIDOR_ROW = 2
WALLS = frozenset({(r, c) for c in (4, 5, 6) for r in range(ROWS) if r != CORRIDOR_ROW})
GOALS = ((CORRIDOR_ROW, 10), (CORRIDOR_ROW, 0))  # agent 0 goes right, agent 1 goes left
ROOM = (
    [(r, c) for r in range(ROWS) for c in range(0, 4) if (r, c) != GOALS[1]],   # left room
    [(r, c) for r in range(ROWS) for c in range(7, 11) if (r, c) != GOALS[0]],  # right room
)
CELLS = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in WALLS]
CELL_ID = {p: i for i, p in enumerate(CELLS)}
NC = len(CELLS)
DONE_ID = NC  # "other agent has finished" token for the joint state
ACTIONS = ((1, 0), (-1, 0), (0, -1), (0, 1), (0, 0))  # up, down, left, right, stay
NA = len(ACTIONS)
GAMMA = 0.95
ALPHA = 0.1
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.998
MAX_STEPS = 50
BUDGET = 3000
GOAL_R, COLLISION_R, STEP_R = 100.0, -10.0, -0.1
N_SEEDS = 30
N_EVAL = 50
LEARNERS = ("iql_own", "iql_joint", "jal")


def propose(p, a):
    dr, dc = ACTIONS[a]
    q = (p[0] + dr, p[1] + dc)
    if not (0 <= q[0] < ROWS and 0 <= q[1] < COLS) or q in WALLS:
        return p
    return q


def joint_step(pos, done, acts):
    """Resolve one joint move. Finished agents leave the grid."""
    prop = [pos[i] if done[i] else propose(pos[i], acts[i]) for i in range(2)]
    collided = False
    if not done[0] and not done[1]:
        same = prop[0] == prop[1]
        swap = prop[0] == pos[1] and prop[1] == pos[0]
        if same or swap:
            collided = True
            prop = list(pos)
    rew = [0.0, 0.0]
    new_done = list(done)
    for i in range(2):
        if done[i]:
            continue
        rew[i] += STEP_R + (COLLISION_R if collided else 0.0)
        if prop[i] == GOALS[i]:
            rew[i] += GOAL_R
            new_done[i] = True
    return prop, new_done, rew, collided


def state_ids(learner, pos, done):
    if learner == "iql_own":
        return [CELL_ID[pos[0]], CELL_ID[pos[1]]]
    other = [DONE_ID if done[1] else CELL_ID[pos[1]], DONE_ID if done[0] else CELL_ID[pos[0]]]
    return [CELL_ID[pos[0]] * (NC + 1) + other[0], CELL_ID[pos[1]] * (NC + 1) + other[1]]


def sample_starts(g):
    a = ROOM[0][int(g.integers(len(ROOM[0])))]
    b = ROOM[1][int(g.integers(len(ROOM[1])))]
    return [a, b]


class Agents:
    def __init__(self, learner):
        self.learner = learner
        if learner == "iql_own":
            self.Q = [np.zeros((NC, NA)), np.zeros((NC, NA))]
        elif learner == "iql_joint":
            self.Q = [np.zeros((NC * (NC + 1), NA)), np.zeros((NC * (NC + 1), NA))]
        else:
            self.Q = np.zeros(((NC + 1) * (NC + 1), NA * NA))

    def jal_state(self, pos, done):
        a = DONE_ID if done[0] else CELL_ID[pos[0]]
        b = DONE_ID if done[1] else CELL_ID[pos[1]]
        return a * (NC + 1) + b

    def act(self, pos, done, eps, g, greedy=False):
        if self.learner == "jal":
            s = self.jal_state(pos, done)
            if not greedy and g.random() < eps:
                ja = int(g.integers(NA * NA))
            else:
                qs = self.Q[s].tolist()
                ja = int(np.argmax(qs)) if greedy else argmax_random(qs, g)
            return [ja // NA, ja % NA], s
        ids = state_ids(self.learner, pos, done)
        acts = []
        for i in range(2):
            if not greedy and g.random() < eps:
                acts.append(int(g.integers(NA)))
            else:
                qs = self.Q[i][ids[i]].tolist()
                acts.append(int(np.argmax(qs)) if greedy else argmax_random(qs, g))
        return acts, ids

    def update(self, s, acts, rew, pos2, done_prev, done2):
        if self.learner == "jal":
            ja = acts[0] * NA + acts[1]
            r = rew[0] + rew[1]
            terminal = done2[0] and done2[1]
            s2 = self.jal_state(pos2, done2)
            target = r if terminal else r + GAMMA * float(self.Q[s2].max())
            self.Q[s, ja] += ALPHA * (target - self.Q[s, ja])
            return
        ids2 = state_ids(self.learner, pos2, done2)
        for i in range(2):
            if done_prev[i]:
                continue
            target = rew[i] if done2[i] else rew[i] + GAMMA * float(self.Q[i][ids2[i]].max())
            self.Q[i][s[i], acts[i]] += ALPHA * (target - self.Q[i][s[i], acts[i]])


def rollout(agents, starts, g=None, eps=0.0, greedy=True, learn=False, record=False):
    pos, done = list(starts), [False, False]
    ret, collisions, steps = 0.0, 0, MAX_STEPS
    traj = [list(pos)]
    for t in range(MAX_STEPS):
        acts, s = agents.act(pos, done, eps, g, greedy=greedy)
        pos2, done2, rew, col = joint_step(pos, done, acts)
        if learn:
            agents.update(s, acts, rew, pos2, done, done2)
        ret += rew[0] + rew[1]
        collisions += int(col)
        pos, done = pos2, done2
        if record:
            traj.append(list(pos))
        if done[0] and done[1]:
            steps = t + 1
            break
    out = {"return": ret, "collisions": collisions, "success": done[0] and done[1],
           "makespan": steps if (done[0] and done[1]) else None}
    if record:
        out["trajectory"] = [[list(p) for p in step] for step in traj]
    return out


def run(seed: int, learner: str) -> dict:
    li = LEARNERS.index(learner)
    g = rng("B", li, seed)
    g_start = rng("B", 50, seed)  # training start pairs: identical sequence for all learners
    agents = Agents(learner)
    eps = EPS_START
    curve_ret, curve_col, curve_succ = [], [], []
    for ep in range(BUDGET):
        starts = sample_starts(g_start)
        o = rollout(agents, starts, g, eps=eps, greedy=False, learn=True)
        curve_ret.append(o["return"])
        curve_col.append(o["collisions"])
        curve_succ.append(o["success"])
        eps = max(EPS_END, eps * EPS_DECAY)
    g_eval = rng("B", 90, seed)  # evaluation start pairs shared across learners
    evals = [rollout(agents, sample_starts(g_eval)) for _ in range(N_EVAL)]
    canonical = rollout(agents, [(CORRIDOR_ROW, 1), (CORRIDOR_ROW, 9)], record=True)
    return {
        "seed": seed, "learner": learner,
        "train_return": curve_ret, "train_collisions": curve_col, "train_success": curve_succ,
        "eval_success": float(np.mean([e["success"] for e in evals])),
        "eval_collisions": float(np.mean([e["collisions"] for e in evals])),
        "eval_makespan_median": (float(np.median([e["makespan"] for e in evals if e["success"]]))
                                 if any(e["success"] for e in evals) else None),
        "canonical": canonical,
    }


def _job(args):
    return run(*args)


def main(out_dir: Path, workers: int = 2):
    jobs = [(s, l) for l in LEARNERS for s in range(N_SEEDS)]
    with Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = {
        "config": {"rows": ROWS, "cols": COLS, "walls": sorted(WALLS), "goals": GOALS,
                   "corridor_row": CORRIDOR_ROW, "rewards": [GOAL_R, COLLISION_R, STEP_R],
                   "gamma": GAMMA, "alpha": ALPHA, "eps": [EPS_START, EPS_END, EPS_DECAY],
                   "max_steps": MAX_STEPS, "budget": BUDGET, "n_eval": N_EVAL,
                   "n_seeds": N_SEEDS, "n_cells": NC},
        "runs": res,
    }
    (out_dir / "problem_b.json").write_text(json.dumps(out))
    return out
