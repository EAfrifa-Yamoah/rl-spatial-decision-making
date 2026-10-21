# rl-spatial-decision-making (v6)

Code, raw results and figure scripts for *Reinforcement Learning for Spatial
Decision Making: Methodological Gaps in GIScience and a Research Agenda*
(Afrifa-Yamoah et al.).

v6 is a rebuild, not an edit, of the v5 bundle. In v5 the README stated that the
code did not reproduce the manuscript's numbers. In v6 **every number, table and
figure in the manuscript and the Supplementary Material is generated from the
raw outputs of this repository** through one file, `results/manifest.json`.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10 or later
python -m pytest tests -q                # a few seconds
python -m src.reproduce_all              # all four problems, about 15 min on 2 cores
python -m tools.build_manifest           # every reported number -> results/manifest.json
python -m src.figures.make_all           # every figure -> results/figures/ (PNG 600 dpi and PDF)
```

`python -m src.reproduce_all A C` runs a subset. Runtimes on the reference
machine are recorded in `results/raw/environment.json`.

## What each problem does

| Problem | Module | Design (30 independent seeds per condition) |
|---|---|---|
| A | `src/simulations/problem_a.py` | Q-learning, 12 x 12 grid. A1: five rewards with algorithm and tabular state fixed. A1 sensitivity: deterministic tie-breaking with the first action toward or away from the goal. A2: four state representations, learning rate tuned per representation on 20 separate seeds. |
| B | `src/simulations/problem_b.py` | Two agents swap rooms through a one-lane corridor. Independent Q-learners with own-position state, independent Q-learners with joint state, centralised joint-action learner. Greedy evaluation on 50 shared start pairs per seed. |
| C | `src/simulations/problem_c.py` | 20 sensor placements on a 30 x 30 Gaussian process field. Six heuristic rules and a 2 x 2 factorial of learned policies (REINFORCE or linear Q-learning; RMSE-reduction or measured-concentration reward), 1,500 training fields per run, 30 held-out test fields and 30 shifted fields shared by all methods. Kappa sweep and action-selection sensitivity. |
| D | `src/simulations/problem_d.py` | SARSA, 8 x 8 grid, one patrolling obstacle, collisions terminal. Row-blind or row-aware state, trained on row 4 or rows 2 to 4; exhaustive greedy evaluation on rows 1 to 6 and a two-obstacle test. |

Full specifications are in the module docstrings and Supplementary Appendix B.

## From manuscript item to code

| Manuscript item | Script | Manifest keys |
|---|---|---|
| Figure 1, 2, 8 | `src/figures/fig_concepts.py` | none (conceptual) |
| Figure 3, Tables S3 to S5 | `src/figures/fig_problem_a.py` | `A.*` |
| Figure 4, Table S6 | `src/figures/fig_problem_b.py` | `B.*` |
| Figure 5, Figures S1 and S2, Tables S7 to S11 | `src/figures/fig_problem_c.py` | `C.*` |
| Figure 6, Table S12 | `src/figures/fig_problem_d.py` | `D.*` |
| Figure 7 | `src/figures/fig_references.py` | `docs/coding_sheet.csv` |
| Table 3 and all numbers in Section 4 | `tools/build_manifest.py` | all |

`python -m tools.check_consistency MANUSCRIPT.docx SUPPLEMENT.docx [RESPONSE.docx]`
re-reads the documents and reports
any number that does not match the manifest, any figure, table, section or
appendix reference without a target, and any citation without a reference
entry (and vice versa).

## Random numbers and reproducibility

All randomness comes from `numpy.random.SeedSequence(20261021, spawn_key=(problem, condition, seed, ...))`
(see `src/utils/__init__.py`), so every condition and seed has its own
independent stream and adding a condition never changes another. Exact
numerical reproduction was verified with the versions pinned in
`requirements.txt` (Python 3.13, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2)
on Linux. Other versions or BLAS libraries may change the last digits of
Gaussian process computations and, through rare ties, individual
placements in Problem C; qualitative results should not change. If they do,
please open an issue.

## Statistics

Medians with 95% percentile bootstrap intervals (10,000 resamples) for
continuous seed- or field-level metrics; means with bootstrap intervals for
success rates averaged over evaluation rollouts; exact Clopper-Pearson
intervals for counts of seeds; Mann-Whitney U with Cliff's delta (Fisher's
exact test for counts) for independent training runs; Wilcoxon signed-rank
with matched-pairs rank-biserial r for comparisons on the same test fields;
Holm adjustment within each named family. A non-significant test is never
read as evidence of equivalence.

## Figure colours

One colour, one meaning, across all figures (Okabe and Ito palette, checked
for colour-vision deficiency): dark blue, value-based learner with a full
state; light blue, value-based learner with a restricted or compressed state;
orange, policy-gradient learner; green, model-based methods; purple, RL
augmentations; vermilion, hazards and safety; greys, non-learning baselines
and environment structure. See `src/figures/style.py`.

## Licence and citation

MIT. Please cite the manuscript and the archived release (see `CITATION.cff`).
