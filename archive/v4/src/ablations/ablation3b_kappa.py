"""
Ablation 3b: Sweep the exploration coefficient kappa in the UCB acquisition on
Problem C. This tests the claim that reward specification (kappa) drives the
ranking, not algorithm choice.

Algorithm fixed: argmax UCB. State fixed: posterior mean + variance from RBF.
Kappa in {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}. 30 seeds. Budget 20.
"""
import numpy as np
from scipy import stats
import json

N = 30; BUDGET = 20; N_SEEDS = 30

def make_field(seed):
    rng = np.random.RandomState(seed)
    xs = np.arange(N); ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys)
    def g(cx, cy, sx, sy, amp):
        return amp * np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2)))
    f = g(7, 22, 3.2, 3.0, 1.0) + g(20, 8, 4.5, 3.8, 0.85) + g(24, 24, 2.5, 2.5, 0.6)
    f += 0.03 * rng.randn(N, N)
    return f.ravel(), np.stack([X.ravel(), Y.ravel()], axis=1)

def rbf_regress(sample_coords, sample_vals, grid_pts, length=4.0, noise=0.05):
    sample_coords = np.asarray(sample_coords, dtype=float)
    sample_vals = np.asarray(sample_vals, dtype=float)
    n = len(sample_coords)
    if n == 0:
        return np.zeros(len(grid_pts)), np.ones(len(grid_pts))
    def kern(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :])**2, axis=2)
        return np.exp(-d / (2 * length**2))
    K = kern(sample_coords, sample_coords) + noise * np.eye(n)
    K_s = kern(grid_pts, sample_coords)
    alpha = np.linalg.solve(K, sample_vals)
    mean = K_s @ alpha
    v = np.linalg.solve(K, K_s.T)
    var = np.clip(1.0 - np.sum(K_s * v.T, axis=1), 1e-6, None)
    return mean, var

def run_kappa(kappa, seed):
    rng = np.random.RandomState(seed)
    true_flat, all_pts = make_field(seed)
    placements = []; vals = []
    for _ in range(BUDGET):
        mean, var = rbf_regress(placements, vals, all_pts)
        mask = np.ones(len(all_pts), dtype=bool)
        for p in placements:
            mask[p[1] * N + p[0]] = False
        score = mean + kappa * np.sqrt(var)
        score[~mask] = -np.inf
        choice = int(np.argmax(score))
        pt = all_pts[choice]
        placements.append((int(pt[0]), int(pt[1])))
        vals.append(float(true_flat[choice] + 0.03 * rng.randn()))
    m, _ = rbf_regress(placements, vals, all_pts)
    return float(np.sqrt(np.mean((m - true_flat) ** 2)))

def bootstrap_ci(vals, n=1000, ci=0.95):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(0)
    boot = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)])
    return float(np.median(vals)), float(np.percentile(boot, 100*(1-ci)/2)), float(np.percentile(boot, 100*(1+ci)/2))

if __name__ == "__main__":
    kappas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = {}
    for k in kappas:
        rmses = [run_kappa(k, s) for s in range(N_SEEDS)]
        m, lo, hi = bootstrap_ci(rmses)
        results[f"kappa_{k}"] = {'kappa': k, 'rmse_median': m, 'rmse_ci_lo': lo, 'rmse_ci_hi': hi, 'raw': rmses}
        print(f"kappa={k}: RMSE median={m:.4f} CI=[{lo:.4f}, {hi:.4f}]")

    # Wilcoxon: kappa=0 (pure greedy) vs each other kappa
    print("\nWilcoxon tests vs kappa=0 (greedy baseline):")
    for k in kappas[1:]:
        stat, p = stats.wilcoxon(results[f"kappa_0.0"]['raw'], results[f"kappa_{k}"]['raw'])
        print(f"  kappa=0 vs kappa={k}: W={stat:.2f}  p={p:.4f}")

    with open('ablation3b_kappa.json', 'w') as f:
        json.dump(results, f, indent=2)
