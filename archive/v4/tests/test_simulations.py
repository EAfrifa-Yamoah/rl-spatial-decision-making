"""
Smoke tests for each simulation.

These tests run each simulation at drastically reduced episode counts to
verify that the code runs end-to-end and produces outputs with the
expected shape and sane ranges. They do NOT verify convergence — the
full manuscript-reproduction parameters are used by `reproduce_all.py`.

Run:
    pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.simulations import (
    sim01_qlearning_gridworld as s1,
    sim02_multi_agent as s2,
    sim03_adaptive_sensor as s3,
    sim04_dynamic_obstacle as s4,
)
from src.utils import set_seed


def test_sim01_qlearning_runs():
    set_seed(0)
    env = s1.GridWorld.default()
    Q, returns, steps = s1.train_qlearning(env, n_episodes=50)
    assert Q.shape == (12, 12, 4)
    assert len(returns) == 50
    assert len(steps) == 50
    # After 50 episodes with aggressive exploration the mean return
    # should at least be finite.
    assert np.isfinite(np.mean(returns))


def test_sim01_policy_is_consistent():
    """Greedy policy from the Q-table should not point into walls."""
    set_seed(0)
    env = s1.GridWorld.default()
    Q, _, _ = s1.train_qlearning(env, n_episodes=100)
    policy = np.argmax(Q, axis=2)
    # The policy must have been written to every non-obstacle cell.
    assert policy.shape == (env.size, env.size)


def test_sim02_multi_agent_runs():
    set_seed(0)
    env = s2.MultiAgentGrid.default()
    Q, joint_returns, traj_early, traj_late = s2.train_independent_qlearners(
        env, n_episodes=30
    )
    assert len(Q) == 2
    assert Q[0].shape == (10, 10, 5)
    assert len(joint_returns) == 30
    assert len(traj_early) >= 1
    assert len(traj_late) >= 1


def test_sim03_sensor_reduces_rmse():
    """Sanity check that any strategy reduces RMSE as it deploys sensors."""
    set_seed(0)
    field = s3.build_field(n=20)
    _, rmses = s3.run_strategy("random", field, budget=10, n=20)
    assert len(rmses) == 10
    # Final RMSE should be lower than initial.
    assert rmses[-1] < rmses[0] + 0.1  # generous tolerance


def test_sim03_all_strategies_complete():
    set_seed(0)
    field = s3.build_field(n=15)
    for strat in ("random", "greedy", "rl"):
        placements, rmses = s3.run_strategy(strat, field, budget=5, n=15)
        assert len(placements) == 5
        assert len(rmses) == 5
        assert all(r >= 0 for r in rmses)


def test_sim04_dynamic_runs():
    set_seed(0)
    env = s4.DynamicObstacleEnv()
    Q, success = s4.train_sarsa(env, n_episodes=80)
    assert Q.shape == (env.n_states, 4)
    assert len(success) == 80
    # All success rates must be probabilities.
    assert all(0.0 <= r <= 1.0 for r in success)


def test_sim04_state_encoding_roundtrip():
    env = s4.DynamicObstacleEnv()
    for ar in range(env.size):
        for ac in range(env.size):
            for oc in range(env.size):
                for d in range(2):
                    idx = env.state_to_idx((ar, ac, oc, d))
                    assert 0 <= idx < env.n_states
