"""Problem C: adaptive sensor placement on a 30 x 30 field.

Belief-state MDP. The state is the observation history (equivalently the
Gaussian process posterior under fixed hyperparameters), the action is an
unsensed cell, the horizon is T = 20 placements, gamma = 1. Two families of
decision procedures act on exactly the same posterior:

Heuristic acquisition rules (no learning)
  random, greedy (posterior mean, kappa = 0), UCB (kappa = 1.5),
  max posterior variance, integrated variance reduction (IVR; Cohn 1996),
  space filling (farthest point). Under fixed GP hyperparameters the
  posterior variance does not depend on the observed values, so max
  variance, IVR and space filling produce one fixed, non-adaptive design;
  only greedy and UCB (and the learned policies) react to observations.

Learned RL policies (2 x 2 factorial, 30 independent training runs each)
  algorithm  REINFORCE (linear softmax policy gradient) or
             linear Q-learning (semi-gradient, epsilon greedy)
  reward     'rmse': per-step reduction in whole-field RMSE (telescopes to
                     the terminal objective)
             'conc': the measured concentration at the chosen cell
                     (a hotspot-seeking objective)
  Both learners score each unsensed cell from the same four standardised
  features: posterior mean, posterior standard deviation, distance to the
  nearest existing sensor, distance to the grid edge. This class nests the
  greedy, max variance, UCB and space-filling rules as special cases.

Training fields and test fields come from disjoint random streams. All
methods are evaluated on the same 30 held-out test fields with the same
pre-drawn observation noise, so comparisons are paired by field. A second
set of 30 shifted fields (five narrower sources) probes distribution shift.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.utils import rng

N = 30
NCELL = N * N
T = 20
LENGTH = 4.0
NOISE_VAR = 0.05
FIELD_NOISE_SD = 0.03
KAPPA = 1.5
KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
TOP_FRACTION = 0.10
N_TEST = 30
N_RUNS = 30
TRAIN_EPISODES = 1500
HEURISTICS = ("random", "greedy", "ucb", "maxvar", "ivr", "spacefill")
LEARNERS = tuple(f"{alg}_{rw}" for alg in ("reinforce", "qlearn") for rw in ("rmse", "conc"))

YY, XX = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
COORDS = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(float)  # (x, y), index = y * N + x
D2 = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
KERNEL = np.exp(-D2 / (2 * LENGTH ** 2))
DIST = np.sqrt(D2)
CENTRE_DIST = np.sqrt(((COORDS - (N - 1) / 2) ** 2).sum(1))
EDGE_DIST = np.minimum.reduce([COORDS[:, 0], COORDS[:, 1], N - 1 - COORDS[:, 0], N - 1 - COORDS[:, 1]])
N_TOP = int(round(NCELL * TOP_FRACTION))


# -------------------------------------------------------------------- fields
def make_field(g: np.random.Generator, kind: str = "standard"):
    """Return (true field, observation noise for every cell)."""
    if kind == "standard":
        n_src, w_lo, w_hi = 3, 2.5, 4.5
    elif kind == "shift":
        n_src, w_lo, w_hi = 5, 1.5, 2.5
    else:
        raise ValueError(kind)
    f = np.zeros(NCELL)
    for _ in range(n_src):
        cx, cy = g.uniform(2, N - 3, size=2)
        sx, sy = g.uniform(w_lo, w_hi, size=2)
        amp = g.uniform(0.6, 1.0)
        f += amp * np.exp(-((COORDS[:, 0] - cx) ** 2 / (2 * sx ** 2)
                            + (COORDS[:, 1] - cy) ** 2 / (2 * sy ** 2)))
    f += FIELD_NOISE_SD * g.standard_normal(NCELL)
    noise = np.sqrt(NOISE_VAR) * g.standard_normal(NCELL)
    return f, noise


def test_fields(kind: str):
    stream = {"standard": 2, "shift": 3}[kind]
    return [make_field(rng("C", stream, j), kind) for j in range(N_TEST)]


# ------------------------------------------------------------- GP posterior
class Posterior:
    """GP posterior with fixed hyperparameters, updated one observation at a
    time (Sigma_t = K - C C^T)."""

    def __init__(self):
        self.mu = np.zeros(NCELL)
        self.var = np.ones(NCELL)
        self.C = np.zeros((NCELL, T))
        self.t = 0
        self.sensed = np.zeros(NCELL, bool)
        self.dmin = np.full(NCELL, np.inf)

    def add(self, c: int, y: float):
        col = KERNEL[:, c] - self.C[:, :self.t] @ self.C[c, :self.t]
        s = col[c] + NOISE_VAR
        self.mu = self.mu + col * (y - self.mu[c]) / s
        newc = col / np.sqrt(s)
        self.var = np.maximum(self.var - newc ** 2, 1e-10)
        self.C[:, self.t] = newc
        self.t += 1
        self.sensed[c] = True
        self.dmin = np.minimum(self.dmin, DIST[:, c])

    def full_cov(self):
        return KERNEL - self.C[:, :self.t] @ self.C[:, :self.t].T


def rmse(mu, f):
    return float(np.sqrt(np.mean((mu - f) ** 2)))


def recall(mu, f):
    true_top = set(np.argsort(f)[-N_TOP:].tolist())
    pred_top = set(np.argsort(mu)[-N_TOP:].tolist())
    return len(true_top & pred_top) / N_TOP


def pick(score, mask):
    """Argmax over unsensed cells; ties go to the cell nearest the centre."""
    s = np.where(mask, score, -np.inf)
    m = s.max()
    cand = np.flatnonzero(s >= m - 1e-12)
    if len(cand) == 1:
        return int(cand[0])
    return int(cand[np.argmin(CENTRE_DIST[cand])])


# ------------------------------------------------------------ heuristics
def fixed_design(rule: str):
    """Designs that do not depend on observed values."""
    post = Posterior()
    design = []
    for _ in range(T):
        mask = ~post.sensed
        if rule == "maxvar":
            c = pick(post.var, mask)
        elif rule == "ivr":
            S = post.full_cov()
            score = (S ** 2).sum(0) / (np.diag(S) + NOISE_VAR)
            c = pick(score, mask)
        elif rule == "spacefill":
            score = -CENTRE_DIST if post.t == 0 else post.dmin
            c = pick(score, mask)
        else:
            raise ValueError(rule)
        design.append(c)
        post.add(c, 0.0)  # value is irrelevant to the variance
    return design


def run_policy(field, noise, chooser):
    """Run one episode; chooser(post, mask, t) -> cell index."""
    post = Posterior()
    traj = [rmse(post.mu, field)]
    for t in range(T):
        c = chooser(post, ~post.sensed, t)
        post.add(c, field[c] + noise[c])
        traj.append(rmse(post.mu, field))
    return {"rmse_traj": traj, "rmse": traj[-1], "recall": recall(post.mu, field),
            "placements": np.flatnonzero(post.sensed).tolist(), "order": None}


def heuristic_chooser(rule, kappa=KAPPA, design=None, g=None, select="argmax"):
    def choose(post, mask, t):
        if design is not None:
            return design[t]
        if rule == "random":
            return int(g.choice(np.flatnonzero(mask)))
        score = post.mu + kappa * np.sqrt(post.var)
        if select == "argmax":
            return pick(score, mask)
        idx = np.flatnonzero(mask)
        if select == "softmax":
            z = score[idx] / 0.5
            p = np.exp(z - z.max())
            return int(idx[g.choice(len(idx), p=p / p.sum())])
        if select == "epsilon":
            if g.random() < 0.1:
                return int(g.choice(idx))
            return pick(score, mask)
        raise ValueError(select)
    return choose


# ------------------------------------------------------------ learned policies
def features(post: Posterior):
    """Standardised features for every cell (rows) -> (NCELL, 4)."""
    mask = ~post.sensed
    d = post.dmin if post.t > 0 else np.zeros(NCELL)
    F = np.stack([post.mu, np.sqrt(post.var), d, EDGE_DIST], axis=1)
    m = F[mask].mean(0)
    sd = F[mask].std(0)
    sd[sd < 1e-12] = np.inf
    return (F - m) / sd


class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999):
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.lr, self.b1, self.b2, self.k = lr, b1, b2, 0

    def step(self, grad):
        """Return an update for gradient *descent* on `grad`."""
        self.k += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.k)
        vh = self.v / (1 - self.b2 ** self.k)
        return -self.lr * mh / (np.sqrt(vh) + 1e-8)


def reward_fn(kind, field, post_before_mu, post_after_mu, c, y):
    if kind == "rmse":
        return rmse(post_before_mu, field) - rmse(post_after_mu, field)
    return y  # 'conc': measured concentration


def reward_scale(kind, g):
    """1 / mean |return| of a uniformly random policy on 20 training fields;
    positive scaling leaves the optimal policy unchanged."""
    tot = []
    for _ in range(20):
        f, e = make_field(g)
        post = Posterior()
        G = 0.0
        for t in range(T):
            c = int(g.choice(np.flatnonzero(~post.sensed)))
            mu0 = post.mu
            post.add(c, f[c] + e[c])
            G += reward_fn(kind, f, mu0, post.mu, c, f[c] + e[c])
        tot.append(abs(G))
    return 1.0 / float(np.mean(tot))


def train_reinforce(run: int, kind: str, lr: float = 0.05):
    g = rng("C", 10 + ("rmse", "conc").index(kind), run)
    gf = rng("C", 1, run)  # training fields: same sequence for both algorithms
    scale = reward_scale(kind, rng("C", 20, run))
    theta = np.zeros(4)
    opt = Adam(4, lr)
    base = np.zeros(T)
    curve = []
    for ep in range(TRAIN_EPISODES):
        f, e = make_field(gf)
        post = Posterior()
        grads, rews = [], []
        for t in range(T):
            Phi = features(post)
            idx = np.flatnonzero(~post.sensed)
            z = Phi[idx] @ theta
            p = np.exp(z - z.max())
            p /= p.sum()
            k = int(g.choice(len(idx), p=p))
            c = int(idx[k])
            grads.append(Phi[c] - p @ Phi[idx])
            mu0 = post.mu
            y = f[c] + e[c]
            post.add(c, y)
            rews.append(scale * reward_fn(kind, f, mu0, post.mu, c, y))
        G = np.cumsum(rews[::-1])[::-1]
        adv = G - base
        base += 0.05 * (G - base)
        grad = sum(a * gr for a, gr in zip(adv, grads))
        theta += opt.step(-grad)  # ascent
        curve.append(rmse(post.mu, f))
    return theta, curve


def train_qlearn(run: int, kind: str, lr: float = 0.01):
    g = rng("C", 30 + ("rmse", "conc").index(kind), run)
    gf = rng("C", 1, run)
    scale = reward_scale(kind, rng("C", 20, run))
    w = np.zeros(6)
    opt = Adam(6, lr)
    curve = []

    def psi(Phi, t):
        extra = np.tile([1.0, (T - t) / T], (NCELL, 1))
        return np.hstack([Phi, extra])

    n_decay = int(0.8 * TRAIN_EPISODES)
    for ep in range(TRAIN_EPISODES):
        eps = max(0.05, 1.0 - 0.95 * ep / n_decay)
        f, e = make_field(gf)
        post = Posterior()
        Psi = psi(features(post), 0)
        for t in range(T):
            mask = ~post.sensed
            q = Psi @ w
            if g.random() < eps:
                c = int(g.choice(np.flatnonzero(mask)))
            else:
                c = pick(q, mask)
            x = Psi[c].copy()
            mu0 = post.mu
            y = f[c] + e[c]
            post.add(c, y)
            r = scale * reward_fn(kind, f, mu0, post.mu, c, y)
            if t + 1 < T:
                Psi = psi(features(post), t + 1)
                q2 = np.where(~post.sensed, Psi @ w, -np.inf).max()
                target = r + q2
            else:
                target = r
            td = target - x @ w
            w += opt.step(-td * x)
        curve.append(rmse(post.mu, f))
    return w, curve


def learned_chooser(alg, params):
    def choose(post, mask, t):
        Phi = features(post)
        if alg == "reinforce":
            score = Phi @ params
        else:
            score = np.hstack([Phi, np.tile([1.0, (T - t) / T], (NCELL, 1))]) @ params
        return pick(score, mask)
    return choose


def run_learner(args):
    name, run = args
    alg, kind = name.split("_")
    if alg == "reinforce":
        params, curve = train_reinforce(run, kind)
    else:
        params, curve = train_qlearn(run, kind)
    ch = learned_chooser(alg, params)
    ev = {}
    for set_name, fields in (("standard", TEST["standard"]), ("shift", TEST["shift"])):
        ev[set_name] = [run_policy(f, e, ch) for f, e in fields]
        for o in ev[set_name]:
            o.pop("placements")
            o.pop("order")
    return {"learner": name, "run": run, "params": params.tolist(),
            "train_curve": curve, "eval": ev}


TEST = {}


def _init():
    TEST["standard"] = test_fields("standard")
    TEST["shift"] = test_fields("shift")


def placement_order(rule_or_learner, field_idx=0, params=None, designs=None):
    f, e = TEST["standard"][field_idx]
    if rule_or_learner in ("maxvar", "ivr", "spacefill"):
        ch = heuristic_chooser(rule_or_learner, design=designs[rule_or_learner])
    elif rule_or_learner in ("random", "greedy", "ucb"):
        ch = heuristic_chooser("random" if rule_or_learner == "random" else "ucb",
                               kappa=0.0 if rule_or_learner == "greedy" else KAPPA,
                               g=rng("C", 40, field_idx))
    else:
        alg = rule_or_learner.split("_")[0]
        ch = learned_chooser(alg, np.asarray(params))
    post = Posterior()
    order = []
    for t in range(T):
        c = ch(post, ~post.sensed, t)
        order.append(c)
        post.add(c, f[c] + e[c])
    return order


def main(out_dir: Path, workers: int = 2):
    _init()
    res = {"config": {"N": N, "T": T, "length": LENGTH, "noise_var": NOISE_VAR,
                      "field_noise_sd": FIELD_NOISE_SD, "kappa": KAPPA, "kappas": KAPPAS,
                      "n_test": N_TEST, "n_runs": N_RUNS, "train_episodes": TRAIN_EPISODES,
                      "top_fraction": TOP_FRACTION}}
    designs = {r: fixed_design(r) for r in ("maxvar", "ivr", "spacefill")}
    res["fixed_designs"] = designs
    for set_name, so in (("standard", 0), ("shift", 100)):
        fields = TEST[set_name]
        block = {}
        for rule in HEURISTICS:
            rows = []
            for j, (f, e) in enumerate(fields):
                ch = heuristic_chooser(
                    "random" if rule == "random" else "ucb",
                    kappa=0.0 if rule == "greedy" else KAPPA,
                    design=designs.get(rule), g=rng("C", 40 + so, j))
                o = run_policy(f, e, ch)
                o.pop("placements")
                o.pop("order")
                rows.append(o)
            block[rule] = rows
        # kappa sweep
        for k in KAPPAS:
            block[f"kappa_{k}"] = [run_policy(f, e, heuristic_chooser("ucb", kappa=k))["rmse"]
                                   for f, e in fields]
        # action-selection sensitivity over the same UCB score
        for sel, st in (("softmax", 60), ("epsilon", 70)):
            block[f"select_{sel}"] = [
                run_policy(f, e, heuristic_chooser("ucb", g=rng("C", st + so, j), select=sel))["rmse"]
                for j, (f, e) in enumerate(fields)]
        res[f"heuristics_{set_name}"] = block
    jobs = [(name, r) for name in LEARNERS for r in range(N_RUNS)]
    with Pool(workers, initializer=_init) as pool:
        learned = pool.map(run_learner, jobs, chunksize=1)
    res["learned"] = learned
    # placement orders on test field 0 for the figure
    orders = {r: placement_order(r, designs=designs) for r in HEURISTICS}
    for name in LEARNERS:
        p0 = next(l["params"] for l in learned if l["learner"] == name and l["run"] == 0)
        orders[name] = placement_order(name, params=p0)
    res["orders_field0"] = orders
    res["field0"] = TEST["standard"][0][0].tolist()
    (out_dir / "problem_c.json").write_text(json.dumps(res))
    return res
