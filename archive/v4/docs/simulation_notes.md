# Simulation notes

Companion notes to the four simulation modules. Each section explains the MDP formulation, hyperparameter choices, expected behaviour, and pointers for extension.

---

## Sim 01 — Tabular Q-learning in a gridworld (Fig. 2)

**MDP**
- States: cells in a 12×12 grid.
- Actions: {up, down, left, right}.
- Transitions: deterministic; moves into walls or obstacles leave the agent in place.
- Rewards: +100 on goal, −5 on obstacle bump, −1 on wall bump, −0.1 per step.
- Horizon: max 200 steps per episode.

**Algorithm.** Tabular Q-learning (Watkins & Dayan, 1992) with α = 0.1, γ = 0.95, ε-greedy exploration decaying from 1.0 to 0.05 with time-constant τ = 120 episodes.

**What to look for.** The 20-episode moving-average return climbs from ~−80 to ~97 within 150 episodes. The value surface forms a smooth gradient toward the goal, and the greedy policy's arrows flow around obstacles.

**Extensions.**
- Replace ε-greedy with optimistic initialisation (all Q(s, a) = 100) and observe faster initial exploration.
- Add stochastic transitions (5% chance of slipping) and observe whether SARSA outperforms Q-learning near obstacles.
- Scale to 50×50 and watch tabular approaches struggle; motivates DQN.

---

## Sim 02 — Independent multi-agent Q-learning (Fig. 4)

**MDP.** Two-agent stochastic game on a 10×10 grid with obstacles. Each agent has its own 5-action space (incl. "stay"); rewards couple via a −10 collision penalty on same-cell co-location or position swap.

**Algorithm.** Two independent Q-learners. Each treats the other as part of the environment — a deliberately simple baseline that illustrates emergent coordination despite non-stationarity.

**What to look for.** Early trajectories overlap chaotically; after ~400 episodes agents settle into non-interfering corridors. Joint return stabilises around 196 (theoretical ceiling ≈ 200 for two collision-free minimum-length paths).

**Extensions.**
- Add more agents (3–5). Independent learners degrade with agent count; this motivates CTDE methods (MADDPG, QMIX).
- Add partial observability (agents see only a 3×3 window). This motivates recurrent policies.
- Introduce heterogeneous agents with different goals to observe competitive equilibria.

---

## Sim 03 — Adaptive sensor placement (Fig. 5)

**Problem.** Given a true 2-D pollution field (mixture of Gaussians), sequentially place B sensors to minimise reconstruction RMSE. The posterior is modelled with RBF regression; three acquisition strategies are compared.

**Acquisition strategies.**
- `random`: uniform.
- `greedy`: argmax of predicted concentration.
- `rl`: argmax of mean + κ·√variance (upper confidence). This mimics the acquisition function a learned policy would converge to under an information-gain + value reward.

**What to look for.** The RL strategy achieves the lowest final RMSE (~0.10) by balancing coverage (via variance) with concentration (via mean). Greedy collapses to hotspots and misses the rest of the field (RMSE ~0.26). Random is surprisingly competitive (~0.13) because it covers the field by construction.

**Extensions.**
- Replace RBF regression with a Gaussian process using `scikit-learn`.
- Add mobile sensors with a path-length budget (transitions between placements cost).
- Introduce time-varying fields and observe whether static acquisition still works.

---

## Sim 04 — SARSA with a dynamic obstacle (Fig. 6)

**MDP.** 8×8 gridworld with a horizontally patrolling obstacle on a fixed row. State is (agent_row, agent_col, obstacle_col, obstacle_direction) — a 4-tuple that preserves Markovness by embedding obstacle dynamics.

**Algorithm.** On-policy SARSA with α = 0.1, γ = 0.95. The linear decay of ε from 1.0 to 0.05 over half the training run strikes a practical balance for this problem.

**What to look for.** Rolling success rate climbs from near 0% to 100% after ~1,800 episodes. The learned policy exhibits a "wait-and-pass" pattern: the agent enters the obstacle's row only when the obstacle is on the far side.

**Extensions.**
- Move the obstacle vertically as well (add obstacle_row to state).
- Make the obstacle stochastic (probabilistic direction changes). SARSA's on-policy nature becomes important here.
- Increase the grid size and observe the curse of dimensionality in the state encoding.
