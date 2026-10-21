"""
Smoke tests for the v4 ablation scripts.

These tests import each ablation module and verify basic invariants
(function signatures, small-N execution). Full 30-seed runs are the
job of ``src.ablations.reproduce_all``.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_ablation1_reward_train_runs():
    """One seed, one reward key, few episodes -> should return a dict."""
    from src.ablations import ablation1_reward as a1

    # Monkey-patch the module's episode budget for speed
    original = a1.N_EPISODES
    a1.N_EPISODES = 20
    try:
        result = a1.train("shaped", seed=0)
    finally:
        a1.N_EPISODES = original

    assert isinstance(result, dict)
    assert "first_goal_ep" in result
    assert "converged" in result


def test_ablation2_state_reps():
    """Three state reps should be trainable at small N."""
    from src.ablations import ablation2_state as a2

    original = a2.N_EPISODES
    a2.N_EPISODES = 20
    try:
        r_tab = a2.train_tabular(seed=0)
        r_lin = a2.train_linear(seed=0)
        r_conv = a2.train_conv(seed=0)
    finally:
        a2.N_EPISODES = original

    for r in (r_tab, r_lin, r_conv):
        assert "first_goal_ep" in r
        assert "converged" in r


def test_ablation3_algorithm_ucb():
    """All three UCB variants should produce a finite RMSE."""
    from src.ablations import ablation3_algorithm as a3

    for policy in ("argmax_ucb", "softmax", "epsilon_greedy"):
        rmse = a3.run_policy(policy, seed=0)
        assert np.isfinite(rmse)
        assert 0.0 <= rmse < 1.0


def test_ablation3b_kappa_sweep_single():
    """A single kappa value should return finite RMSE at seed 0."""
    from src.ablations import ablation3b_kappa as a3b

    rmse = a3b.run_kappa(1.5, seed=0)
    assert np.isfinite(rmse)
    assert 0.0 <= rmse < 1.0


def test_holdout_geography_train_small():
    """A short SARSA train should produce a Q table of the right shape."""
    from src.ablations import holdout_geography_v2 as ho

    original = ho.N_EPISODES_TRAIN
    ho.N_EPISODES_TRAIN = 100
    try:
        Q = ho.train_sarsa(seed=0)
    finally:
        ho.N_EPISODES_TRAIN = original

    # (agent_r, agent_c, obs_col, obs_dir) -> (SIZE^3 * 2) states, 4 actions
    expected_n_states = ho.SIZE * ho.SIZE * ho.SIZE * 2
    assert Q.shape == (expected_n_states, ho.N_ACTIONS)


def test_reference_classification_totals():
    """The classifier should return a stable count on stable input."""
    from docs.reference_classification import CLASSIFICATION

    total = len(CLASSIFICATION)
    assert total == 98, f"Expected 98 refs in v4 classification, got {total}"

    from collections import Counter
    counts = Counter(CLASSIFICATION.values())
    applied = total - counts["foundational"]
    foundational = counts["foundational"]
    assert applied == 41
    assert foundational == 57
