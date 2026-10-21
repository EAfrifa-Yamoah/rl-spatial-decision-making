"""
Ablation 2: State representation on Problem A.

Fix algorithm (Q-learning family) and reward (shaped), vary state representation:
  A. tabular:  Q(s,a) table indexed by (row, col)
  B. linear:   Q(s,a) = w_a . phi(s), where phi(s) is a concatenation of row
               one-hot and column one-hot (24-dim features)
  C. small_conv: Q(s,a) computed by a tiny convolutional network with a 3x3
               receptive field around the agent, hand-coded feature: (goal_dr,
               goal_dc, nearest_obstacle_dr, nearest_obstacle_dc, wall_flags)
               with linear final layer

Reports sample complexity via episodes-to-first-goal and episodes-to-converge
(final policy successfully reaches goal in evaluation).

30 seeds each.
"""
import numpy as np
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
GAMMA = 0.95

# --- Environment (shared) ---
def env_step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= SIZE or nc < 0 or nc >= SIZE:
        return state, -1.0, False
    if (nr, nc) in OBSTACLES:
        return state, -5.0, False
    if (nr, nc) == GOAL:
        return (nr, nc), 100.0, True
    return (nr, nc), -0.1, False

# --- Representations ---
def phi_linear(s):
    """Row indicator concatenated with column indicator (24-dim)."""
    v = np.zeros(2 * SIZE)
    v[s[0]] = 1.0
    v[SIZE + s[1]] = 1.0
    return v

def phi_conv(s):
    """Small hand-coded features simulating a 3x3 convolutional receptive field.
    Encodes: normalized position, direction to goal, nearest obstacle distance
    in each cardinal direction. 8-dim feature vector.
    """
    r, c = s
    goal_dr = (GOAL[0] - r) / SIZE
    goal_dc = (GOAL[1] - c) / SIZE
    # nearest obstacle in each direction within receptive field
    obs_up = obs_down = obs_left = obs_right = 3.0  # cap at 3 cells
    for i in range(1, 4):
        if (r-i, c) in OBSTACLES: obs_up = min(obs_up, i); break
    for i in range(1, 4):
        if (r+i, c) in OBSTACLES: obs_down = min(obs_down, i); break
    for i in range(1, 4):
        if (r, c-i) in OBSTACLES: obs_left = min(obs_left, i); break
    for i in range(1, 4):
        if (r, c+i) in OBSTACLES: obs_right = min(obs_right, i); break
    at_wall_up = 1.0 if r == 0 else 0.0
    at_wall_left = 1.0 if c == 0 else 0.0
    return np.array([goal_dr, goal_dc, obs_up/3, obs_down/3, obs_left/3, obs_right/3,
                     at_wall_up, at_wall_left])

# --- Q-learning with three representations ---
def train_tabular(seed):
    rng = np.random.RandomState(seed)
    Q = np.zeros((SIZE, SIZE, N_ACTIONS))
    first_goal_ep = None
    per_ep_return = []
    for ep in range(N_EPISODES):
        eps = 0.05 + 0.95 * np.exp(-ep / 120.0)
        state = START
        total_r = 0.0
        for t in range(MAX_STEPS):
            if rng.rand() < eps:
                a = rng.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[state[0], state[1]]))
            ns, r, done = env_step(state, a)
            total_r += r
            td = r + GAMMA * (0 if done else np.max(Q[ns[0], ns[1]]))
            Q[state[0], state[1], a] += 0.1 * (td - Q[state[0], state[1], a])
            state = ns
            if done: break
        per_ep_return.append(total_r)
        if total_r > 50 and first_goal_ep is None:
            first_goal_ep = ep
    # eval
    state = START; eval_return = 0
    for t in range(MAX_STEPS):
        a = int(np.argmax(Q[state[0], state[1]]))
        ns, r, done = env_step(state, a)
        eval_return += r; state = ns
        if done: break
    return {
        'first_goal_ep': first_goal_ep if first_goal_ep is not None else N_EPISODES,
        'converged': eval_return > 50,
        'eval_return': eval_return,
    }

def train_linear(seed):
    """Linear Q(s,a) = w_a . phi(s)."""
    rng = np.random.RandomState(seed)
    d = 2 * SIZE
    W = np.zeros((N_ACTIONS, d))
    alpha = 0.05  # smaller LR for FA to avoid divergence
    first_goal_ep = None
    for ep in range(N_EPISODES):
        eps = 0.05 + 0.95 * np.exp(-ep / 120.0)
        state = START
        total_r = 0.0
        for t in range(MAX_STEPS):
            phi_s = phi_linear(state)
            qs = W @ phi_s
            if rng.rand() < eps:
                a = rng.randint(N_ACTIONS)
            else:
                a = int(np.argmax(qs))
            ns, r, done = env_step(state, a)
            total_r += r
            phi_ns = phi_linear(ns)
            q_next = 0 if done else np.max(W @ phi_ns)
            td_err = r + GAMMA * q_next - qs[a]
            W[a] += alpha * td_err * phi_s
            state = ns
            if done: break
        if total_r > 50 and first_goal_ep is None:
            first_goal_ep = ep
    state = START; eval_return = 0
    for t in range(MAX_STEPS):
        phi_s = phi_linear(state)
        a = int(np.argmax(W @ phi_s))
        ns, r, done = env_step(state, a)
        eval_return += r; state = ns
        if done: break
    return {
        'first_goal_ep': first_goal_ep if first_goal_ep is not None else N_EPISODES,
        'converged': eval_return > 50,
        'eval_return': eval_return,
    }

def train_conv(seed):
    """Small conv-style features (8-dim) with linear head."""
    rng = np.random.RandomState(seed)
    d = 8
    W = np.zeros((N_ACTIONS, d))
    alpha = 0.05
    first_goal_ep = None
    for ep in range(N_EPISODES):
        eps = 0.05 + 0.95 * np.exp(-ep / 120.0)
        state = START
        total_r = 0.0
        for t in range(MAX_STEPS):
            phi_s = phi_conv(state)
            qs = W @ phi_s
            if rng.rand() < eps:
                a = rng.randint(N_ACTIONS)
            else:
                a = int(np.argmax(qs))
            ns, r, done = env_step(state, a)
            total_r += r
            phi_ns = phi_conv(ns)
            q_next = 0 if done else np.max(W @ phi_ns)
            td_err = r + GAMMA * q_next - qs[a]
            W[a] += alpha * td_err * phi_s
            state = ns
            if done: break
        if total_r > 50 and first_goal_ep is None:
            first_goal_ep = ep
    state = START; eval_return = 0
    for t in range(MAX_STEPS):
        phi_s = phi_conv(state)
        a = int(np.argmax(W @ phi_s))
        ns, r, done = env_step(state, a)
        eval_return += r; state = ns
        if done: break
    return {
        'first_goal_ep': first_goal_ep if first_goal_ep is not None else N_EPISODES,
        'converged': eval_return > 50,
        'eval_return': eval_return,
    }

def bootstrap_ci(vals, n=1000, ci=0.95):
    vals = np.asarray(vals, dtype=float)
    boot = np.zeros(n)
    rng = np.random.RandomState(0)
    for i in range(n):
        boot[i] = np.median(rng.choice(vals, size=len(vals), replace=True))
    return float(np.median(vals)), float(np.percentile(boot, 100*(1-ci)/2)), float(np.percentile(boot, 100*(1+ci)/2))

if __name__ == "__main__":
    results = {}
    for name, fn in [('tabular', train_tabular), ('linear', train_linear), ('conv', train_conv)]:
        print(f"Running {name}...")
        seed_results = [fn(s) for s in range(N_SEEDS)]
        fg = [r['first_goal_ep'] for r in seed_results]
        conv = [r['converged'] for r in seed_results]
        m, lo, hi = bootstrap_ci(fg)
        results[name] = {
            'first_goal_median': m,
            'first_goal_ci_lo': lo,
            'first_goal_ci_hi': hi,
            'n_converged': int(sum(conv)),
            'n_seeds': N_SEEDS,
            'first_goal_raw': fg,
        }
        print(f"  first-goal median={m:.0f} CI=[{lo:.0f}, {hi:.0f}]  converged={sum(conv)}/{N_SEEDS}")

    with open('ablation2_state.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved ablation2_state.json")
