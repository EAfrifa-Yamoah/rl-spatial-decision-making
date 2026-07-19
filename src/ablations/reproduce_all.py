"""
Reproduce all reviewer-requested ablations (v4 revision).

Runs six ablation scripts in sequence, saves JSON results under
results/ablations/, and prints a summary table matching the numbers
reported in Section 4 and the response-to-reviewers letter.

Usage:
    python -m src.ablations.reproduce_all

Wall-clock: ~5 minutes on a laptop (single-threaded).
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ablations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("ablation1_reward",       "Reward function on Problem A (Q-learning, tabular)"),
    ("ablation2_state",        "State representation on Problem A (shaped reward)"),
    ("ablation3_algorithm",    "Algorithm on Problem C (UCB reward + posterior state)"),
    ("ablation3b_kappa",       "Kappa sweep on Problem C (reward specification)"),
    ("problem_c_30seeds",      "Problem C main comparison, 30 seeds"),
    ("holdout_geography_v2",   "Held-out geography on Problem D"),
]


def run_one(name: str, label: str) -> None:
    print("=" * 72)
    print(f"[{name}] {label}")
    print("=" * 72)
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    # Each ablation script writes its own JSON in the CWD; run from RESULTS_DIR
    # so outputs land in the results directory.
    import os
    old_cwd = os.getcwd()
    os.chdir(RESULTS_DIR)
    try:
        runpy.run_path(str(module_path), run_name="__main__")
    finally:
        os.chdir(old_cwd)
    print()


def print_summary() -> None:
    print("=" * 72)
    print("SUMMARY OF ABLATION RESULTS")
    print("=" * 72)
    print(
        "All results below use 30 random seeds. Confidence intervals are 95%\n"
        "bootstrap intervals from 1,000 resamples. Wilcoxon signed-rank tests\n"
        "are two-sided.\n"
    )

    # Ablation 1
    a1 = json.loads((RESULTS_DIR / "ablation1_reward.json").read_text())
    print("\n### Ablation 1: Reward function on Problem A (Q-learning, tabular)")
    for k, v in a1.items():
        print(
            f"  {k:15s}  first-goal ep median={v['first_goal_median']:>4.0f}  "
            f"CI=[{v['first_goal_ci_lo']:>3.0f}, {v['first_goal_ci_hi']:>3.0f}]  "
            f"converged={v['n_converged']:>2d}/{v['n_seeds']}"
        )

    # Ablation 2
    a2 = json.loads((RESULTS_DIR / "ablation2_state.json").read_text())
    print("\n### Ablation 2: State representation on Problem A (shaped reward)")
    for k, v in a2.items():
        print(
            f"  {k:10s}  first-goal ep median={v['first_goal_median']:>4.0f}  "
            f"CI=[{v['first_goal_ci_lo']:>3.0f}, {v['first_goal_ci_hi']:>3.0f}]  "
            f"converged={v['n_converged']:>2d}/{v['n_seeds']}"
        )

    # Ablation 3
    a3 = json.loads((RESULTS_DIR / "ablation3_algorithm.json").read_text())
    print("\n### Ablation 3: Algorithm on Problem C (UCB reward + posterior state)")
    for k in ["argmax_ucb", "softmax", "epsilon_greedy"]:
        v = a3[k]
        print(
            f"  {k:18s}  RMSE median={v['rmse_median']:.4f}  "
            f"CI=[{v['rmse_ci_lo']:.4f}, {v['rmse_ci_hi']:.4f}]"
        )
    for k, t in a3.get("wilcoxon", {}).items():
        print(f"    Wilcoxon {k}: W={t['W']:.1f}  p={t['p_value']:.4f}")

    # Ablation 3b
    a3b = json.loads((RESULTS_DIR / "ablation3b_kappa.json").read_text())
    print("\n### Ablation 3b: Kappa sweep on Problem C (argmax UCB)")
    for k, v in a3b.items():
        print(
            f"  kappa={v['kappa']:>3.1f}  RMSE median={v['rmse_median']:.4f}  "
            f"CI=[{v['rmse_ci_lo']:.4f}, {v['rmse_ci_hi']:.4f}]"
        )

    # Problem C main
    pc = json.loads((RESULTS_DIR / "problem_c_30seeds.json").read_text())
    print("\n### Problem C main comparison (30-seed rerun)")
    for k in ["random", "greedy", "ucb"]:
        v = pc[k]
        print(
            f"  {k:8s}  RMSE median={v['rmse_median']:.4f}  "
            f"CI=[{v['rmse_ci_lo']:.4f}, {v['rmse_ci_hi']:.4f}]"
        )
    for k, v in pc.items():
        if k.startswith("wilcoxon"):
            print(f"    {k}: W={v['W']:.1f}  p={v['p_value']:.6f}")

    # Held-out geography
    ho = json.loads((RESULTS_DIR / "holdout_geography_v2.json").read_text())
    print("\n### Held-out geography (Problem D SARSA, trained on row 4)")
    for k, v in ho.items():
        print(
            f"  {k:22s}  success={v['median']:.2%}  "
            f"CI=[{v['ci_lo']:.2%}, {v['ci_hi']:.2%}]"
        )

    print()
    print("Raw JSON results in:", RESULTS_DIR)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    only_summary = "--summary-only" in argv
    if not only_summary:
        for name, label in SCRIPTS:
            run_one(name, label)
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
