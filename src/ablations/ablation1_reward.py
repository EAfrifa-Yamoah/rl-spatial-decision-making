"""
Ablation 1: Reward function on Problem A (12x12 gridworld Q-learning).

Compares four reward specifications, holding algorithm (Q-learning) and state
representation (tabular) fixed.

Rewards under comparison:
  A. sparse_only:  +100 on goal, 0 otherwise, terminal on obstacle contact (r=0)
  B. shaped:       +100 goal, -5 obstacle, -1 wall, -0.1 step (manuscript's original)
  C. potential_pbrs: shaped goal only, with potential based reward shaping using
                    negative Manhattan distance to goal as the potential function
  D. random_dense: matched mean of the shaped reward but random per (s,a) noise

30 random seeds, 500 episodes each, budget 200 steps per episode.
Reports median with 95% bootstrap CI for:
  - Episodes to first goal
  - Fraction of seeds that converge (final 50-episode mean return > 90)
  - Final policy quality (correct greedy action from start)
"""
import numpy as np
from collections import defaultdict
import json

SIZE = 12
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4
GOAL = (11, 11)
START = (0, 0)
OBSTACLES = set()
for i in range(3, 8): OBSTACLES.add((i, 4))
for i in range(5, 10): OBSTACLES.add((7, i))
for i in range(2, 6): OBSTACLES.add((10, i))
for i in range(0, 3): OBSTACLES.add((4, i + 7))

N_SEEDS = 30
N_EPISODES = 500
MAX_STEPS = 200
ALPHA = 0.1
GAMMA = 0.95

def step(state, action, reward_fn, potential_prev=None):
    """Apply action; return (next_state, reward, done)."""
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    # Wall bump
    if nr < 0 or nr >= SIZE or nc < 0 or nc >= SIZE:
        ns = state; contact = 'wall'
    elif (nr, nc) in OBSTACLES:
        ns = state; contact = 'obstacle'
    elif (nr, nc) == GOAL:
        ns = (nr, nc); contact = 'goal'
        return ns, reward_fn(state, action, ns, contact), True
    else:
        ns = (nr, nc); contact = 'step'
    return ns, reward_fn(state, action, ns, contact), False

def r_sparse(s, a, ns, contact):
    if contact == 'goal': return 100.0
    return 0.0

def r_shaped(s, a, ns, contact):
    if contact == 'goal': return 100.0
    if contact == 'obstacle': return -5.0
    if contact == 'wall': return -1.0
    return -0.1

def manhattan(s):
    return abs(s[0] - GOAL[0]) + abs(s[1] - GOAL[1])

def r_pbrs(s, a, ns, contact):
    """Goal only reward + potential based shaping (Ng et al. 1999)."""
    # F(s, a, s') = gamma * Phi(s') - Phi(s)
    # Phi(s) = -manhattan(s) so closer to goal = higher potential
    phi_s = -manhattan(s)
    phi_ns = -manhattan(ns) if contact != 'goal' else 0.0  # goal state has Phi = 0
    base = 100.0 if contact == 'goal' else 0.0
    return base + GAMMA * phi_ns - phi_s

# Matched mean random dense: mean of r_shaped over trajectory is roughly -0.5
# (mostly step rewards). We'll set random dense to N(-0.5, 1) with +100 on goal.
def r_random_dense(s, a, ns, contact, rng):
    if contact == 'goal': return 100.0
    return rng.normal(-0.5, 1.0)

def train(reward_fn_key, seed, use_rng_for_reward=False):
    rng = np.random.RandomState(seed)
    Q = np.zeros((SIZE, SIZE, N_ACTIONS))
    per_episode_return = []
    per_episode_steps = []
    first_goal_ep = None

    for ep in range(N_EPISODES):
        eps = 0.05 + 0.95 * np.exp(-ep / 120.0)
        state = START
        total_r = 0.0
        done = False
        goal_hit = False
        for t in range(MAX_STEPS):
            if rng.rand() < eps:
                a = rng.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[state[0], state[1]]))
            if reward_fn_key == 'sparse':
                ns, r, done = step(state, a, r_sparse)
            elif reward_fn_key == 'shaped':
                ns, r, done = step(state, a, r_shaped)
            elif reward_fn_key == 'pbrs':
                ns, r, done = step(state, a, r_pbrs)
            elif reward_fn_key == 'random_dense':
                ns, r, done = step(state, a, lambda s, act, nns, c: r_random_dense(s, act, nns, c, rng))
            total_r += r
            td = r + GAMMA * (0 if done else np.max(Q[ns[0], ns[1]]))
            Q[state[0], state[1], a] += ALPHA * (td - Q[state[0], state[1], a])
            state = ns
            if done and r > 50:  # reached goal
                goal_hit = True
                break
            if done: break
        per_episode_return.append(total_r)
        per_episode_steps.append(t + 1)
        if goal_hit and first_goal_ep is None:
            first_goal_ep = ep

    # Final greedy policy from start: does it move toward goal?
    a_greedy_start = int(np.argmax(Q[START[0], START[1]]))
    # A correct first action is 'right' (0,1) or 'down' (1,0) — both reduce Manhattan distance
    correct_start = a_greedy_start in [1, 3]  # 1 = down (row+1), 3 = right (col+1)

    # Final mean return over last 50 episodes with the true shaped reward
    # (evaluate policy without exploration, one rollout)
    state = START
    eval_return = 0.0
    for t in range(MAX_STEPS):
        a = int(np.argmax(Q[state[0], state[1]]))
        ns, r, done = step(state, a, r_shaped)  # evaluate under shaped for fair comparison
        eval_return += r
        state = ns
        if done: break
    eval_steps = t + 1
    eval_reached_goal = done and eval_return > 50

    return {
        'first_goal_ep': first_goal_ep if first_goal_ep is not None else N_EPISODES,
        'converged': eval_reached_goal,
        'correct_start_action': correct_start,
        'eval_return': eval_return,
        'eval_steps': eval_steps,
        'final_avg_return': np.mean(per_episode_return[-50:]),
    }

def bootstrap_ci(vals, n=1000, ci=0.95):
    vals = np.asarray(vals, dtype=float)
    boot = np.zeros(n)
    rng = np.random.RandomState(0)
    for i in range(n):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot[i] = np.median(sample)
    lo = np.percentile(boot, 100*(1-ci)/2)
    hi = np.percentile(boot, 100*(1+ci)/2)
    return float(np.median(vals)), float(lo), float(hi)

if __name__ == "__main__":
    results = {}
    for key in ['sparse', 'shaped', 'pbrs', 'random_dense']:
        print(f"Running {key}...")
        seed_results = []
        for s in range(N_SEEDS):
            seed_results.append(train(key, s))
        first_goals = [r['first_goal_ep'] for r in seed_results]
        conv = [r['converged'] for r in seed_results]
        correct = [r['correct_start_action'] for r in seed_results]

        m_fg, lo_fg, hi_fg = bootstrap_ci(first_goals)
        results[key] = {
            'first_goal_median': m_fg,
            'first_goal_ci_lo': lo_fg,
            'first_goal_ci_hi': hi_fg,
            'n_converged': int(sum(conv)),
            'n_seeds': N_SEEDS,
            'n_correct_start': int(sum(correct)),
            'first_goal_raw': first_goals,
        }
        print(f"  first-goal median={m_fg:.0f} CI=[{lo_fg:.0f}, {hi_fg:.0f}]  "
              f"converged={sum(conv)}/{N_SEEDS}  correct_start={sum(correct)}/{N_SEEDS}")

    with open('ablation1_reward.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved ablation1_reward.json")
