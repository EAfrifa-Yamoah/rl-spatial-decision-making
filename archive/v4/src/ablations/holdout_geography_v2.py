"""
Held-out geography protocol v2. The v1 shifts were too easy. Here we test
harder OOD shifts:
  1. In-distribution:      horizontal sweep, obstacle at row 4 (as trained)
  2. Row shift:            obstacle at row 2 instead of row 4
  3. Two obstacles:        two simultaneous horizontal obstacles at rows 3 and 5

These are more realistic "held-out geographies" — same problem class, different
spatial parameters.
"""
import numpy as np
import json

SIZE = 8
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
OBS_ROW_TRAIN = 4

N_SEEDS = 30
N_EPISODES_TRAIN = 4000
MAX_STEPS = 80
ALPHA = 0.1
GAMMA = 0.95

def obs_step(oc, d):
    noc = oc + (1 if d == 1 else -1)
    nd = d
    if noc < 0 or noc >= SIZE:
        nd = 1 - d
        noc = oc + (1 if nd == 1 else -1)
    return noc, nd

def state_idx(ar, ac, oc, dr):
    return ((ar * SIZE + ac) * SIZE + oc) * 2 + dr

def train_sarsa(seed):
    rng = np.random.RandomState(seed)
    n_states = SIZE * SIZE * SIZE * 2
    Q = np.zeros((n_states, N_ACTIONS))
    for ep in range(N_EPISODES_TRAIN):
        eps = max(0.05, 1.0 - ep / (N_EPISODES_TRAIN / 2))
        ar, ac = START
        oc = rng.randint(SIZE); d = rng.randint(2)
        s_idx = state_idx(ar, ac, oc, d)
        a = rng.randint(N_ACTIONS) if rng.rand() < eps else int(np.argmax(Q[s_idx]))
        for t in range(MAX_STEPS):
            dr, dc = ACTIONS[a]
            nar, nac = ar + dr, ac + dc
            noc, nd = obs_step(oc, d)
            if nar < 0 or nar >= SIZE or nac < 0 or nac >= SIZE:
                nar, nac = ar, ac
            done = False; r = -0.5
            if nar == OBS_ROW_TRAIN and nac == noc:
                r = -20; done = True
            elif (nar, nac) == GOAL:
                r = 100; done = True
            ns_idx = state_idx(nar, nac, noc, nd)
            na = rng.randint(N_ACTIONS) if rng.rand() < eps else int(np.argmax(Q[ns_idx]))
            target = r + (0 if done else GAMMA * Q[ns_idx, na])
            Q[s_idx, a] += ALPHA * (target - Q[s_idx, a])
            s_idx = ns_idx; a = na; ar, ac = nar, nac; oc = noc; d = nd
            if done: break
    return Q

def eval_at_row(Q, seed, obs_row):
    """Evaluate with obstacle at a different row than trained on.

    The agent's Q table is indexed by (agent_r, agent_c, obs_col, obs_dir) but
    NOT by obs_row — the agent implicitly assumes obs_row = trained row. When
    we move the obstacle to a different row at evaluation time, the agent
    'sees' the state as if the obstacle is still at OBS_ROW_TRAIN.
    """
    rng = np.random.RandomState(seed + 10000)
    successes = 0; n_trials = 50
    for tr in range(n_trials):
        ar, ac = START
        oc = rng.randint(SIZE); d = rng.randint(2)
        for t in range(MAX_STEPS):
            # Agent's state uses the trained row implicitly (state doesn't encode obs_row)
            s_idx = state_idx(ar, ac, oc, d)
            a = int(np.argmax(Q[s_idx]))
            dr, dc = ACTIONS[a]
            nar, nac = ar + dr, ac + dc
            noc, nd = obs_step(oc, d)
            if nar < 0 or nar >= SIZE or nac < 0 or nac >= SIZE:
                nar, nac = ar, ac
            # BUT the actual obstacle is at obs_row (different from trained row)
            if nar == obs_row and nac == noc:
                break
            if (nar, nac) == GOAL:
                successes += 1; break
            ar, ac = nar, nac; oc = noc; d = nd
    return successes / n_trials

def eval_two_obstacles(Q, seed, obs_rows=(3, 5)):
    """Two simultaneous horizontal obstacles."""
    rng = np.random.RandomState(seed + 20000)
    successes = 0; n_trials = 50
    for tr in range(n_trials):
        ar, ac = START
        # Two independent obstacles
        oc1 = rng.randint(SIZE); d1 = rng.randint(2)
        oc2 = rng.randint(SIZE); d2 = rng.randint(2)
        for t in range(MAX_STEPS):
            # Agent sees only the first obstacle in its state encoding
            s_idx = state_idx(ar, ac, oc1, d1)
            a = int(np.argmax(Q[s_idx]))
            dr, dc = ACTIONS[a]
            nar, nac = ar + dr, ac + dc
            noc1, nd1 = obs_step(oc1, d1)
            noc2, nd2 = obs_step(oc2, d2)
            if nar < 0 or nar >= SIZE or nac < 0 or nac >= SIZE:
                nar, nac = ar, ac
            # Collision check against BOTH obstacles
            if (nar == obs_rows[0] and nac == noc1) or (nar == obs_rows[1] and nac == noc2):
                break
            if (nar, nac) == GOAL:
                successes += 1; break
            ar, ac = nar, nac; oc1 = noc1; d1 = nd1; oc2 = noc2; d2 = nd2
    return successes / n_trials

def bootstrap_ci(vals, n=1000):
    vals = np.asarray(vals)
    rng = np.random.RandomState(0)
    boot = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)])
    return float(np.median(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

if __name__ == "__main__":
    print(f"Training {N_SEEDS} SARSA agents on obstacle at row {OBS_ROW_TRAIN}...")
    id_rates, row2_rates, row6_rates, two_rates = [], [], [], []
    for s in range(N_SEEDS):
        Q = train_sarsa(s)
        id_rates.append(eval_at_row(Q, s, OBS_ROW_TRAIN))
        row2_rates.append(eval_at_row(Q, s, 2))
        row6_rates.append(eval_at_row(Q, s, 6))
        two_rates.append(eval_two_obstacles(Q, s))
        if s % 5 == 0:
            print(f"  seed {s+1}/{N_SEEDS}: ID={id_rates[-1]:.2f} row2={row2_rates[-1]:.2f} "
                  f"row6={row6_rates[-1]:.2f} two={two_rates[-1]:.2f}")

    results = {}
    for name, rates in [('in_distribution', id_rates),
                        ('shift_obstacle_row_2', row2_rates),
                        ('shift_obstacle_row_6', row6_rates),
                        ('shift_two_obstacles', two_rates)]:
        m, lo, hi = bootstrap_ci(rates)
        results[name] = {'median': m, 'ci_lo': lo, 'ci_hi': hi, 'raw': rates}

    print("\n=== HELD-OUT GEOGRAPHY RESULTS ===")
    for name, res in results.items():
        print(f"{name:35s} median={res['median']:.2%} CI=[{res['ci_lo']:.2%}, {res['ci_hi']:.2%}]")

    with open('holdout_geography_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
