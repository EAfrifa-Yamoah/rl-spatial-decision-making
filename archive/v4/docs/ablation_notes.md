# Ablation notes (v4 revision)

This document explains the reviewer requested ablations added in v4 and how they map to the manuscript.

## Rationale

Reviewers 1 and 2 both asked for stronger empirical evidence to support the claim that problem specification (state and reward) matters more than algorithm choice in the four canonical problems. In response, v4 adds three factorial ablations plus a held out geography protocol, all with 30 seed replication and paired Wilcoxon signed rank tests.

## The six ablation scripts

### `ablation1_reward.py`
Reward function on Problem A. Q learning and tabular state are fixed. Four rewards are compared: sparse goal only, shaped, potential based shaping with negative Manhattan distance potential (Ng et al. 1999), and a random dense reward with matched mean.

- Sparse only: 0 of 30 seeds converge (median first goal episode 500, hit the budget cap)
- Shaped: 30 of 30, median first goal 8, CI [5, 12]
- Potential based shaping: 29 of 30, median 6, CI [4, 8]. Reaches the same greedy policy — confirms shaping accelerates without distorting the optimum
- Random dense: 30 of 30, median 9, CI [6, 12]. Caveat: works only because the 12 by 12 grid is small enough that the +100 goal signal dominates the added noise. Will not hold at scale.

### `ablation2_state.py`
State representation on Problem A. Q learning and shaped reward are fixed. Three representations are compared: tabular Q table, linear FA with row and column indicator features (24 dim), small hand engineered features (8 dim) emulating a 3 by 3 conv receptive field.

- Tabular: 30 of 30 converged at eval, median first goal 23, CI [18, 28]
- Linear FA: 17 of 30, median 26, CI [16, 32]. Finds the goal at comparable speed but eval time policy is unstable
- Small conv: 6 of 30, median 20, CI [15, 24]. Fastest to first goal but least stable at eval

Finding: eval stability, not sample complexity in the first goal sense, is what separates these representations at this scale.

### `ablation3_algorithm.py`
Algorithm on Problem C. The UCB reward (kappa 1.5) and the posterior state (mean + variance) are fixed. Three policies are compared: argmax UCB, softmax over UCB with temperature 0.5, epsilon greedy over UCB with eps 0.1.

- argmax UCB: RMSE median 0.110, CI [0.093, 0.114]
- softmax: RMSE 0.124, CI [0.111, 0.139]. Worse than argmax (p 0.0002)
- epsilon greedy: RMSE 0.102, CI [0.090, 0.116]. Indistinguishable from argmax (p 0.32)

Nuanced finding: when policies act on the same well specified score with only minor stochasticity differences, the algorithm label does not matter; softmax adds harmful noise on top of an already informative score.

### `ablation3b_kappa.py`
Kappa sweep on Problem C. Algorithm (argmax UCB) and state representation are fixed. Kappa is swept across {0, 0.5, 1, 1.5, 2, 3, 5}.

- kappa 0 (pure greedy): RMSE 0.256
- kappa 0.5: 0.210
- kappa 1.0: 0.125
- kappa 1.5 (paper default): 0.110
- kappa 2.0 (optimum): 0.100
- kappa 3.0: 0.109
- kappa 5.0: 0.124

Clean U shape. Wilcoxon: every kappa above 0 is significantly better than kappa 0 (p < 0.001). This is the strongest single piece of evidence that reward specification drives the ranking, not algorithm label.

### `problem_c_30seeds.py`
Rerun of the main Problem C comparison with 30 seeds and Wilcoxon tests. Numbers match the manuscript's Figure 5.

- Random: RMSE 0.127
- Greedy: RMSE 0.256
- UCB (kappa 1.5): RMSE 0.110

All pairwise Wilcoxon differences highly significant.

### `holdout_geography_v2.py`
Held out geography protocol demonstration on Problem D. The SARSA agent is trained on horizontal obstacle sweeps at row 4, then evaluated on three shifts:

- In distribution (row 4): 100 percent
- Row shift to row 2: 69 percent, CI [52 percent, 98 percent]
- Row shift to row 6: 100 percent (harmless: sits below start goal path)
- Two simultaneous obstacles at rows 3 and 5: 74 percent, CI [58 percent, 82 percent]

Finding: shift severity is asymmetric and only a systematic held out protocol will surface which shifts hurt. Illustrates the six step protocol in Section 6.3.1 of the manuscript.

## Where these results appear in the manuscript

| Ablation | Referenced in |
|---|---|
| Reward function | 4.1 Problem A prose, revised Figure 3d panel, 4.5 What the evidence tells us |
| State representation | 4.1 Problem A prose, supplementary Figure S1, 4.5 |
| Algorithm | 4.3 Problem C prose, 4.5 |
| Kappa sweep | 4.3 Problem C prose, supplementary Figure S2, abstract |
| Problem C 30 seed rerun | 4.3 Problem C, Figure 5, Table 2 row C |
| Held out geography | 6.3.1 A minimum viable held out geography protocol, 4.5, abstract |

## Reproducing the results

```bash
# All six ablations end to end (~5 min single threaded)
python -m src.ablations.reproduce_all

# Fast preview from saved JSON (immediate)
python -m src.ablations.reproduce_all --summary-only

# Individual ablations
python -m src.ablations.ablation1_reward
python -m src.ablations.ablation2_state
python -m src.ablations.ablation3_algorithm
python -m src.ablations.ablation3b_kappa
python -m src.ablations.problem_c_30seeds
python -m src.ablations.holdout_geography_v2
```

JSON outputs land in `results/ablations/`. The `reproduce_all` runner writes there by default.

## Deterministic seeding

Every ablation script sets a `numpy.random.RandomState(seed)` at the top of every function that uses randomness. The 30 seed batch uses seeds 0 through 29. Results are bit reproducible on the same NumPy version.
