"""Shared utilities: random number streams and statistics.

Every random draw in the v6 code base comes from `rng(...)`, which derives an
independent NumPy Generator from a tuple of integers through
`numpy.random.SeedSequence`. The tuple always starts with a problem code, so
streams for different problems, conditions, seeds and purposes never overlap
and adding a condition never perturbs another condition's stream.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

ROOT_ENTROPY = 20261021  # fixed project entropy; change only with a new release

PROBLEM_CODE = {"A": 1, "B": 2, "C": 3, "D": 4}


def rng(problem: str, *keys: int) -> np.random.Generator:
    """Independent generator for (problem, *keys)."""
    ss = np.random.SeedSequence(ROOT_ENTROPY, spawn_key=(PROBLEM_CODE[problem], *keys))
    return np.random.default_rng(ss)


def argmax_random(values, gen: np.random.Generator) -> int:
    """Argmax with uniform random tie-breaking (used during training)."""
    m = max(values)
    ties = [i for i, v in enumerate(values) if v == m]
    if len(ties) == 1:
        return ties[0]
    return ties[int(gen.integers(len(ties)))]


# ---------------------------------------------------------------- statistics
N_BOOT = 10_000


def boot_ci(x: Sequence[float], stat=np.median, n_boot: int = N_BOOT,
            alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over seeds."""
    x = np.asarray(x, dtype=float)
    g = np.random.default_rng(np.random.SeedSequence(ROOT_ENTROPY, spawn_key=(99, seed)))
    idx = g.integers(0, len(x), size=(n_boot, len(x)))
    est = stat(x[idx], axis=1)
    return float(np.quantile(est, alpha / 2)), float(np.quantile(est, 1 - alpha / 2))


def summarise(x: Sequence[float], stat: str = "median") -> dict:
    """Point estimate, bootstrap CI and IQR for a seed-level metric."""
    x = np.asarray(x, dtype=float)
    f = np.median if stat == "median" else np.mean
    lo, hi = boot_ci(x, stat=f)
    return {
        "stat": stat,
        "estimate": float(f(x)),
        "ci95": [lo, hi],
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "n": int(len(x)),
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def proportion(flags: Sequence[bool]) -> dict:
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "p": k / n, "ci95": [lo, hi]}


def paired_test(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sided paired Wilcoxon signed-rank test of x versus y.

    Returns W (the smaller of the signed-rank sums, scipy convention), p,
    the matched-pairs rank-biserial correlation r = (R+ - R-) / (R+ + R-)
    (positive when x tends to exceed y), and the median paired difference
    x - y with a percentile bootstrap CI.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    nz = d[d != 0]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz))}
    lo, hi = boot_ci(d, stat=np.median, seed=7)
    out["median_diff"] = float(np.median(d))
    out["median_diff_ci95"] = [lo, hi]
    if len(nz) == 0:
        out.update({"W": None, "p": 1.0, "r_rb": 0.0})
        return out
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nz))
    r_plus = float(ranks[nz > 0].sum())
    r_minus = float(ranks[nz < 0].sum())
    out.update({
        "W": float(res.statistic),
        "p": float(res.pvalue),
        "r_rb": (r_plus - r_minus) / (r_plus + r_minus),
    })
    return out


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p values (monotone, capped at 1).

    Undefined p values (NaN, e.g. a test between two constant samples) are
    excluded from the family and returned as NaN.
    """
    p = np.asarray(pvals, float)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def family_of_tests(pairs: dict[str, tuple[Sequence[float], Sequence[float]]]) -> dict:
    """Run paired tests for a named family and add Holm-adjusted p values."""
    res = {name: paired_test(a, b) for name, (a, b) in pairs.items()}
    names = list(res)
    adj = holm([res[n]["p"] for n in names])
    for n, a in zip(names, adj):
        res[n]["p_holm"] = a
    return res


def unpaired_test(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U test for independent training runs.

    Effect size: Cliff's delta = P(x > y) - P(x < y) (equal to the
    rank-biserial correlation for two independent samples), and the
    difference in medians x - y with a percentile bootstrap CI.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    gt = (x[:, None] > y[None, :]).mean()
    lt = (x[:, None] < y[None, :]).mean()
    g = np.random.default_rng(np.random.SeedSequence(ROOT_ENTROPY, spawn_key=(98,)))
    bx = x[g.integers(0, len(x), size=(N_BOOT, len(x)))]
    by = y[g.integers(0, len(y), size=(N_BOOT, len(y)))]
    d = np.median(bx, axis=1) - np.median(by, axis=1)
    return {"U": float(res.statistic), "p": float(res.pvalue), "cliffs_delta": float(gt - lt),
            "median_diff": float(np.median(x) - np.median(y)),
            "median_diff_ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "n_x": int(len(x)), "n_y": int(len(y))}


def fisher_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    res = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return {"p": float(res.pvalue), "k1": k1, "n1": n1, "k2": k2, "n2": n2}


def holm_family(tests: dict) -> dict:
    names = list(tests)
    adj = holm([tests[n]["p"] for n in names])
    for n, a in zip(names, adj):
        tests[n]["p_holm"] = a
    return tests
