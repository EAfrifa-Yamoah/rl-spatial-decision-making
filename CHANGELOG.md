# Changelog

All notable changes to this repository are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with dates in ISO 8601.

## [4.0.0] — 2026-07-08

### Added
- New `src/ablations/` package with six reviewer requested ablation scripts:
  - `ablation1_reward.py` — Four reward functions on Problem A (sparse, shaped, potential based, random dense)
  - `ablation2_state.py` — Three state representations on Problem A (tabular, linear FA, small conv features)
  - `ablation3_algorithm.py` — Three algorithms on Problem C (argmax UCB, softmax, epsilon greedy)
  - `ablation3b_kappa.py` — Seven point kappa sweep on Problem C
  - `problem_c_30seeds.py` — 30 seed rerun of the Problem C main comparison
  - `holdout_geography_v2.py` — Held out geography protocol demonstration on Problem D
- `src/ablations/reproduce_all.py` — Runner with `--summary-only` mode for quick verification from committed JSON
- `results/ablations/` — Committed JSON results and `summary.txt` for immediate inspection without rerunning
- `tests/test_ablations.py` — Six smoke tests for the ablation modules
- `docs/ablation_notes.md` — Documentation of every ablation, findings, and where they appear in the manuscript
- `docs/__init__.py` — Makes `docs/` a proper Python package for module invocation
- New reference categories in `docs/reference_classification.py`:
  - `transportation` (4 references)
  - `remote_sensing` (3 references)
- `CHANGELOG.md` — This file

### Changed
- **Corpus expansion**: reference count grows from 80 (v3) to 98 (v4), a 22 percent increase, in response to reviewer 1 comment 2
  - 18 new references added across GIScience, remote sensing, transportation MARL, spatial resource allocation, and OOD generalisation
  - Applied to foundational ratio: 41 / 57 (42 percent / 58 percent)
- **Figure 8 regenerated** with the 98 reference distribution. New leaders: Conservation and wildlife protection (6), Sensor placement (5), GIS-RL integration (5), Autonomous vehicles (4), Transportation and traffic MARL (4)
- **Manuscript title** updated in `README.md` and `CITATION.cff` from "A Methodological Review and Practitioner's Guide" to "Methodological Gaps in GIScience and a Research Agenda"
- **Author list** filled in `CITATION.cff` with the full seven author affiliations
- `setup.py` version bumped to 4.0.0
- CI workflow adds two verification steps: `docs.reference_classification` and `src.ablations.reproduce_all --summary-only`

### Fixed
- Figure numbering consistency: v4 correctly orders Fig. 2 as the taxonomy and Fig. 3 as the Q learning gridworld, matching the revised manuscript cross references
- README's figure table updated to reflect the corrected numbering

### Notes
- All 13 tests pass (7 existing simulation smoke tests + 6 new ablation smoke tests)
- Ablation runtime: about 5 minutes single threaded on a modern laptop
- Full reproduction of figures and ablations: about 10 minutes total

## [3.0.0] — 2026-04-30

### Added
- Expanded reference corpus with ecology, conservation, water management, forestry, and GIScience literatures (from 60 to 80 references)
- New citations: Lapeyrolerie et al. 2022 (Methods in Ecology and Evolution), Marrec & Boettiger 2023, Subramanian & Crowley 2018, Zhang et al. 2024 (GIScience spatial resource allocation survey), and 11 others

### Changed
- Manuscript title updated to include "GIScience" framing
- Section 5.5 corpus paragraph and Figure 8 regenerated with honest applied-to-foundational ratio

## [2.0.0] — 2026-04-24

### Added
- Five synthesis tables integrated into the manuscript
- Ecology and GIS rebalance across canonical problems and gap discussions

## [1.0.0] — 2026-04-23

### Added
- Initial release
- Four canonical simulations (Q learning gridworld, multi agent grid, adaptive sensor, dynamic obstacle)
- Four illustrations (MDP framework, taxonomy, flowchart, applications)
- Reproducible one command build via `src.reproduce_all`
- CI workflow, tests, and package installation
