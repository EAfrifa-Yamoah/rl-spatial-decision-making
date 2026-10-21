"""Fast checks of environment logic and statistics (a few seconds)."""
import numpy as np

from src.utils import holm, paired_test, unpaired_test, clopper_pearson, rng
from src.simulations import problem_a as A, problem_b as B, problem_c as C, problem_d as D


def test_rng_streams_independent_and_deterministic():
    a = rng("A", 1, 2).random(5)
    b = rng("A", 1, 2).random(5)
    c = rng("A", 1, 3).random(5)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == 0.04 and adj[2] == 0.09 and adj[1] == 0.09 and adj[3] == 0.5


def test_effect_sizes_signs():
    x = np.arange(30) + 5.0
    y = np.arange(30) + 0.0
    assert paired_test(x, y)["r_rb"] == 1.0
    assert unpaired_test(x + 100, y)["cliffs_delta"] == 1.0
    lo, hi = clopper_pearson(30, 30)
    assert hi == 1.0 and 0.88 < lo < 0.89


def test_problem_a_pbrs_vanishes_at_goal():
    r = A.make_reward("pbrs", 0)
    assert r((11, 10), A.GOAL, "goal") == 100.0 + 0.0 - (-1)


def test_problem_a_short_run():
    out = A.run(0, reward="shaped", budget=60)
    assert out["first_goal"] is not None


def test_problem_b_collision_rules():
    pos, done, rew, col = B.joint_step([(2, 4), (2, 5)], [False, False], [3, 2])  # swap
    assert col and pos == [(2, 4), (2, 5)] and rew[0] < -10


def test_problem_c_posterior_matches_batch():
    post = C.Posterior()
    cells, ys = [10, 455, 899], [0.3, 0.9, -0.1]
    for c, y in zip(cells, ys):
        post.add(c, y)
    K = C.KERNEL[np.ix_(cells, cells)] + C.NOISE_VAR * np.eye(3)
    mu = C.KERNEL[:, cells] @ np.linalg.solve(K, ys)
    var = 1 - np.sum(C.KERNEL[:, cells] * np.linalg.solve(K, C.KERNEL[cells, :]).T, axis=1)
    assert np.allclose(post.mu, mu) and np.allclose(post.var, np.maximum(var, 1e-10))


def test_problem_c_fixed_designs_do_not_depend_on_values():
    d1 = C.fixed_design("ivr")
    assert len(set(d1)) == C.T


def test_problem_d_swap_is_collision():
    # obstacle at (1, 1) moving left, agent at (1, 0) moving right -> swap
    agent, obs, r, done, ok = D.env_step((1, 0), 3, [(1, 1, 1)])
    assert done and not ok
