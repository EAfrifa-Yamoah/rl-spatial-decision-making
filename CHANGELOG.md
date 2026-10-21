# Changelog

## v6.0.0

Rebuild of the experiment code so that the manuscript, supplement and
repository agree. Every reported number is now computed from
`results/raw/*.json` by `tools/build_manifest.py`.

### Corrections of v5 behaviour that changed results
* Problem A: training now breaks ties between equal action values at
  random. v5 broke ties by action index with the first action pointing
  away from the goal, which made the sparse reward appear never to
  converge; both deterministic orders are retained as a sensitivity
  analysis.
* Problem A, state ablation: the learning rate is tuned separately for each
  representation on 20 tuning seeds (v5 used 0.1 versus 0.05, a confound).
* Problem C: values quoted for the kappa sweep in the previous manuscript and
  response letter (0.256, 0.110, 0.100) are not produced by the archived v5
  code and could not be reproduced; all v6 analyses use one field generator.
  Source locations are now fully random rather than
  jittered around fixed centres; space filling starts at the centre (v5
  started in a corner because of a sign error); IVR is exact over all
  candidates (v5 sampled 100).
* Problem D: collisions with the obstacle, including swaps, end the episode
  as a failure (v5 penalised them but let the episode continue, so
  evaluation counted collided runs as successes); evaluation enumerates
  every initial obstacle state instead of sampling.

### New designs
* Problem B: one-lane corridor map that forces yielding; own-position and
  joint-state independent learners and a centralised joint-action learner.
* Problem C: learned RL policies (REINFORCE and linear Q-learning) trained
  with two rewards and evaluated on held-out and shifted fields.
* Problem D: row-blind versus row-aware state crossed with fixed versus
  varied training rows; held-out rows and a two-obstacle test.

### Other
* `src.utils.holm` excludes undefined (NaN) p values from a family; the
  degenerate in-distribution test (all values 1.0) was removed from the
  Problem D family.
* Problem C manifest records each method against the zero-sensor prior on
  every field (`vs_prior`).
* `docs/coding_sheet_v5_stub.csv` removed; `docs/coding_sheet.csv` is the only
  classification record.
* Single seeding scheme (`numpy.random.SeedSequence`) for all problems.
* Statistics: Holm within named families, Cliff's delta or matched-pairs
  rank-biserial r, bootstrap intervals for paired differences.
* All figures redrawn with a fixed colour semantics and no embedded titles.
* `tools/check_consistency.py` audits numbers, cross references and citations.
