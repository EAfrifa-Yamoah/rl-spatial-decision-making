"""
Ablation 3: Algorithm on Problem C (adaptive sensor placement).

Fix reward (upper confidence: mean + 1.5 * sqrt(var), which is a proxy for
information gain weighted by predicted value) and state representation
(current posterior mean and variance over the grid), and vary the algorithm
that picks the next cell:

  A. argmax_ucb:  greedy argmax over UCB score (deterministic policy)
  B. softmax:     sample cell with probability proportional to exp(UCB / T)
                  with T = 0.5 (soft policy)
  C. epsilon_greedy: argmax with eps=0.1 probability of random cell

The three are formal equivalents of value-based greedy, policy-gradient-style
stochastic policy, and eps-greedy exploration, all operating on the SAME
reward and SAME state representation. If reviewer 2's alternative framing is
correct, these three should perform similarly.

30 seeds, budget = 20 sensors, 30x30 field.
Reports final reconstruction RMSE (median with bootstrap 95% CI) and pairwise
Wilcoxon signed-rank tests.
"""
import numpy as np
from scipy import stats
import json

N = 30
BUDGET = 20
N_SEEDS = 30
KAPPA = 1.5
TEMP = 0.5
EPS = 0.1

# --- Fixed ground truth field: three Gaussian blobs ---
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

def run_policy(policy, seed):
    """Run one seed with the given policy on a freshly generated field."""
    rng = np.random.RandomState(seed)
    true_flat, all_pts = make_field(seed)
    placements = []; vals = []
    for _ in range(BUDGET):
        mean, var = rbf_regress(placements, vals, all_pts)
        mask = np.ones(len(all_pts), dtype=bool)
        for p in placements:
            mask[p[1] * N + p[0]] = False
        candidates = np.where(mask)[0]
        # UCB score
        score = mean + KAPPA * np.sqrt(var)
        score[~mask] = -np.inf

        if policy == 'argmax_ucb':
            choice = int(np.argmax(score))
        elif policy == 'softmax':
            # softmax over UCB score with temperature TEMP
            probs = np.exp(score / TEMP - np.max(score / TEMP))
            probs[~mask] = 0
            probs = probs / probs.sum()
            choice = int(rng.choice(len(all_pts), p=probs))
        elif policy == 'epsilon_greedy':
            if rng.rand() < EPS:
                choice = int(rng.choice(candidates))
            else:
                choice = int(np.argmax(score))

        pt = all_pts[choice]
        placements.append((int(pt[0]), int(pt[1])))
        vals.append(float(true_flat[choice] + 0.03 * rng.randn()))

    mean_final, _ = rbf_regress(placements, vals, all_pts)
    rmse = float(np.sqrt(np.mean((mean_final - true_flat) ** 2)))
    return rmse

def bootstrap_ci(vals, n=1000, ci=0.95):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(0)
    boot = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)])
    return float(np.median(vals)), float(np.percentile(boot, 100*(1-ci)/2)), float(np.percentile(boot, 100*(1+ci)/2))

if __name__ == "__main__":
    results = {}
    rmse_by_policy = {}
    for policy in ['argmax_ucb', 'softmax', 'epsilon_greedy']:
        print(f"Running {policy}...")
        rmses = [run_policy(policy, s) for s in range(N_SEEDS)]
        rmse_by_policy[policy] = rmses
        m, lo, hi = bootstrap_ci(rmses)
        results[policy] = {'rmse_median': m, 'rmse_ci_lo': lo, 'rmse_ci_hi': hi, 'raw': rmses}
        print(f"  RMSE median={m:.4f} CI=[{lo:.4f}, {hi:.4f}]")

    # Pairwise Wilcoxon signed-rank tests
    print("\nPairwise Wilcoxon signed-rank tests (two-sided):")
    pairs = [('argmax_ucb', 'softmax'), ('argmax_ucb', 'epsilon_greedy'), ('softmax', 'epsilon_greedy')]
    tests = {}
    for a, b in pairs:
        stat, pval = stats.wilcoxon(rmse_by_policy[a], rmse_by_policy[b])
        tests[f"{a}_vs_{b}"] = {'W': float(stat), 'p_value': float(pval)}
        print(f"  {a} vs {b}: W={stat:.3f}  p={pval:.4f}")
    results['wilcoxon'] = tests

    with open('ablation3_algorithm.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved ablation3_algorithm.json")
