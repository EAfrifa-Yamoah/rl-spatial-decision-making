# Reinforcement Learning for Spatial Decision Making

Reproducible Python code for the simulations, ablations, and figures in the review manuscript **"Reinforcement Learning for Spatial Decision Making: Methodological Gaps in GIScience and a Research Agenda"** (v4 revision).

[![CI](https://github.com/<user>/rl-spatial-decision-making/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/rl-spatial-decision-making/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The repository produces (i) the eight figures used in the manuscript, (ii) the six reviewer requested factorial ablations added in v4, and (iii) the reference classification behind Figure 8.

Everything runs on pure NumPy, SciPy, and Matplotlib. No deep learning framework is required, by design: the goal is pedagogical reproducibility, not raw performance.

## v4 revision summary

The v4 revision addresses reviewer feedback with three sets of code changes over v3. Revision documents (response letter, addendum, changelog) are separately committed:

- [`CHANGELOG.md`](CHANGELOG.md) — What changed between v3 and v4
- Manuscript v4 (`RL_Spatial_Decision_Making_v4.docx`) and the addendum with new content blocks (`Revision_Addendum.docx`) are distributed with the manuscript submission, not in this repository
- Reviewer response letter (`Response_to_Reviewers.docx`) is likewise distributed with the manuscript submission

The v4 revision addresses reviewer feedback with three sets of code changes over v3.

**Ablation experiments (new in v4).** Every simulation is re run over 30 seeds with bootstrap 95 percent confidence intervals and paired Wilcoxon signed rank tests. Three factorial ablations are added:

| Ablation | Fixed | Varied | Manuscript section |
|---|---|---|---|
| 1. Reward function | Q learning, tabular state | Sparse, shaped, potential based, random dense | 4.1 Problem A |
| 2. State representation | Q learning, shaped reward | Tabular, linear FA, small conv features | 4.1 Problem A |
| 3. Algorithm | UCB reward (kappa 1.5), posterior state | argmax UCB, softmax UCB, epsilon greedy | 4.3 Problem C |
| 3b. Kappa sweep | argmax UCB algorithm | kappa in {0, 0.5, 1, 1.5, 2, 3, 5} | 4.3 Problem C |

Plus a held out geography protocol demonstration on Problem D (train on obstacle row 4, test on rows 2 and 6 and on two simultaneous obstacles).

**Corpus expansion (new in v4).** The reference bibliography grows from 80 to 98 entries. Figure 8 and the classifier in `docs/reference_classification.py` are regenerated. New categories: Transportation and traffic MARL, Remote sensing RL.

**Section renumbering (new in v4).** A new Section 2 (Review Methodology) is inserted, so v3 Sections 2 through 7 become v4 Sections 3 through 8. Cross references throughout the manuscript are updated.

## Figures

| # | Script | Figure | Type | What it shows |
|---|--------|--------|------|---------------|
| 1 | `illustrations/fig01_mdp_framework.py`     | Fig. 1 | Schematic  | Spatial MDP and three state representations |
| 2 | `illustrations/fig03_taxonomy.py`          | Fig. 2 | Schematic  | Taxonomy of RL methods and augmentations |
| 3 | `simulations/sim01_qlearning_gridworld.py` | Fig. 3 | Simulation | Q learning convergence in a 12 by 12 gridworld |
| 4 | `simulations/sim02_multi_agent.py`         | Fig. 4 | Simulation | Two independent Q learners with collision penalties |
| 5 | `simulations/sim03_adaptive_sensor.py`     | Fig. 5 | Simulation | Adaptive sensor placement vs. greedy and random |
| 6 | `simulations/sim04_dynamic_obstacle.py`    | Fig. 6 | Simulation | SARSA avoiding a patrolling dynamic obstacle |
| 7 | `illustrations/fig07_flowchart.py`         | Fig. 7 | Schematic  | Practitioner decision flowchart |
| 8 | `illustrations/fig08_applications.py`      | Fig. 8 | Data       | Application domain landscape and corpus composition |

Note: v4 swaps figures 2 and 3 relative to v3, matching the corrected cross references in the revised manuscript. The taxonomy is now Figure 2 (it introduces algorithm families before the Q learning gridworld demonstrates them in Figure 3).

## Quick start

```bash
git clone https://github.com/<user>/rl-spatial-decision-making.git
cd rl-spatial-decision-making

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Reproduce every figure at 400 DPI (default) in ~2-3 minutes
python -m src.reproduce_all

# Reproduce every ablation with 30 seeds each (~5 minutes)
python -m src.ablations.reproduce_all

# Fast preview: fewer episodes, 200 DPI, ~30 seconds
python -m src.reproduce_all --quick --dpi 200
```

All figures land in `figures/` as PNG files. All ablation JSON results land in `results/ablations/`.

## Running individual scripts

### Figures

```bash
# Gridworld Q-learning with more episodes and a different seed
python -m src.simulations.sim01_qlearning_gridworld \
    --episodes 1000 --seed 7 --dpi 400

# Adaptive sensor placement with a larger budget and more replicates
python -m src.simulations.sim03_adaptive_sensor \
    --budget 40 --replicates 20 --kappa 2.0

# Only the conceptual flowchart
python -m src.illustrations.fig07_flowchart --dpi 300
```

### Ablations

Each ablation writes a JSON summary in the current working directory:

```bash
# All six ablations in sequence with a final summary print
python -m src.ablations.reproduce_all

# Just print the summary from previously saved JSON
python -m src.ablations.reproduce_all --summary-only

# One ablation at a time
python -m src.ablations.ablation1_reward       # Reward on Problem A
python -m src.ablations.ablation2_state        # State rep on Problem A
python -m src.ablations.ablation3_algorithm    # Algorithm on Problem C
python -m src.ablations.ablation3b_kappa       # Kappa sweep on Problem C
python -m src.ablations.problem_c_30seeds      # Problem C main 30 seed rerun
python -m src.ablations.holdout_geography_v2   # Held out geography on Problem D
```

All scripts accept `--dpi` and `--out` (output filename stem). Simulations also accept `--episodes`, `--alpha`, `--gamma`, and `--seed` where applicable.

## Repository layout

```
.
├── src/
│   ├── utils.py                          # Seeding, plot style, save helper
│   ├── reproduce_all.py                  # One command figure regeneration
│   ├── simulations/
│   │   ├── sim01_qlearning_gridworld.py
│   │   ├── sim02_multi_agent.py
│   │   ├── sim03_adaptive_sensor.py
│   │   └── sim04_dynamic_obstacle.py
│   ├── ablations/                        # v4 addition
│   │   ├── ablation1_reward.py           # 4 reward functions on Problem A
│   │   ├── ablation2_state.py            # 3 state representations on Problem A
│   │   ├── ablation3_algorithm.py        # 3 algorithms on Problem C
│   │   ├── ablation3b_kappa.py           # 7 point kappa sweep on Problem C
│   │   ├── problem_c_30seeds.py          # 30 seed rerun of Problem C main
│   │   ├── holdout_geography_v2.py       # OOD evaluation on Problem D
│   │   └── reproduce_all.py              # Runner and summary printer
│   └── illustrations/
│       ├── fig01_mdp_framework.py
│       ├── fig03_taxonomy.py             # v4: renumbered as Figure 2
│       ├── fig07_flowchart.py
│       └── fig08_applications.py         # v4: regenerated with 98 refs
├── results/
│   └── ablations/                        # v4 addition: JSON outputs
├── tests/
│   └── test_simulations.py               # Smoke tests (pytest)
├── figures/                              # Generated outputs (PNG, 400 DPI)
├── notebooks/                            # Optional Jupyter demos
├── docs/
│   ├── reference_classification.py       # v4: 98 refs, 12 domains
│   └── simulation_notes.md
├── .github/workflows/ci.yml
├── requirements.txt
├── setup.py
├── CITATION.cff
├── LICENSE
└── README.md
```

## Expected results

On a modern laptop with no GPU, `python -m src.reproduce_all` completes in 2 to 4 minutes. The simulations will converge to:

| Simulation | Reported outcome |
|------------|------------------|
| `sim01_qlearning_gridworld` | Final mean return around 97 over last 50 episodes (target around 98) |
| `sim02_multi_agent`         | Final joint return around 196; collision free trajectories |
| `sim03_adaptive_sensor`     | RMSE (median of 30 seeds): random around 0.13, greedy around 0.26, UCB around 0.11 |
| `sim04_dynamic_obstacle`    | 100 percent success rate after around 1,800 episodes |

### Expected ablation results

`python -m src.ablations.reproduce_all` takes around 5 minutes and prints:

**Reward on Problem A** (30 seeds, Q learning, tabular):
```
sparse       first-goal ep median=500  CI=[ 12, 500]  converged= 0/30
shaped       first-goal ep median=  8  CI=[  5,  12]  converged=30/30
pbrs         first-goal ep median=  6  CI=[  4,   8]  converged=29/30
random_dense first-goal ep median=  9  CI=[  6,  12]  converged=30/30
```

**State representation on Problem A** (30 seeds, shaped reward):
```
tabular  first-goal ep median= 23  converged=30/30
linear   first-goal ep median= 26  converged=17/30
conv     first-goal ep median= 20  converged= 6/30
```

**Algorithm on Problem C** (30 seeds, UCB reward + posterior state):
```
argmax_ucb      RMSE median=0.1095  CI=[0.0932, 0.1137]
softmax         RMSE median=0.1240  CI=[0.1114, 0.1386]
epsilon_greedy  RMSE median=0.1015  CI=[0.0903, 0.1162]
  Wilcoxon argmax_vs_softmax:         p=0.0002
  Wilcoxon argmax_vs_epsilon_greedy:  p=0.32
```

**Kappa sweep** (30 seeds, argmax UCB): median RMSE ranges from 0.256 at kappa 0 to 0.100 at kappa 2.0 and back up to 0.124 at kappa 5.0. All differences vs kappa 0 are highly significant (p < 0.001).

**Held out geography on Problem D** (30 SARSA seeds, trained on row 4):
```
in_distribution        success=100.00%  CI=[100.00%, 100.00%]
shift_obstacle_row_2   success= 69.00%  CI=[ 52.00%,  98.00%]
shift_obstacle_row_6   success=100.00%  CI=[100.00%, 100.00%]
shift_two_obstacles    success= 74.00%  CI=[ 58.00%,  82.00%]
```

Seeds are set explicitly in each script so these numbers are deterministic across runs on the same machine.

## Reference classification

Figure 8 uses reference counts derived from classifying the manuscript's 98 references. Each reference is assigned to exactly one primary domain based on its citation context in the manuscript.

```bash
python -m docs.reference_classification
```

Prints:
```
Total references classified: 98

By domain (descending count):
  foundational           57
  conservation_wildlife   6
  sensor_placement        5
  gis_rl                  5
  autonomous_vehicles     4
  transportation          4
  robot_navigation        3
  forestry_wildfire       3
  water_reservoir         3
  remote_sensing          3
  env_monitoring          2
  urban_planning          2
  uav_swarm               1

Applied domain refs:       41  (42%)
Foundational refs:         57  (58%)
```

If you disagree with a classification, edit the `CLASSIFICATION` dictionary at the top of `docs/reference_classification.py` and rerun `python -m src.illustrations.fig08_applications` to refresh Figure 8.

## Testing

```bash
pytest -q
```

The test suite runs each simulation at drastically reduced episode counts (seconds, not minutes) and verifies end to end execution, output shape, and sanity bounds. It does not verify convergence. The full reproduction is `python -m src.reproduce_all`.

## Citation

If this code is useful to your work, please cite the accompanying manuscript. See [`CITATION.cff`](CITATION.cff) for a machine readable version.

```bibtex
@article{rl_spatial_review_2026,
  title   = {Reinforcement Learning for Spatial Decision Making:
             Methodological Gaps in GIScience and a Research Agenda},
  author  = {Afrifa-Yamoah, E. and Bradshaw, S. and Awuah-Mensah, Y. K.
             and Nuakoh, B. W. and Tan, W. H. and Fouedjio, F. and Arya, E.},
  journal = {Transactions in GIS},
  year    = {2026},
  note    = {Code: https://github.com/<user>/rl-spatial-decision-making}
}
```

## License

MIT. See [`LICENSE`](LICENSE). You are free to use, modify, and redistribute, including for commercial purposes, with attribution.

## Contributing

Pull requests are welcome. For non trivial changes, please open an issue first to discuss what you would like to change. New simulations that illustrate additional clusters from the manuscript (hierarchical RL for long horizon tasks, transfer learning across maps) are particularly welcome.

## Acknowledgements

The simulations intentionally use small tabular problems to keep every piece of behaviour auditable. The algorithmic templates follow the canonical expositions in Sutton and Barto (2018), with adaptations described in the relevant sections of the manuscript. The v4 revision was informed by careful and constructive reviewer feedback.
