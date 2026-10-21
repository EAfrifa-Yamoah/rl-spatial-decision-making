# Contributing

Thank you for considering a contribution. This repository is the reproducibility companion to a methodological review, so contributions should preserve two properties above all:

1. **Reproducibility.** Every figure must regenerate deterministically from a fresh clone + `pip install` + `python -m src.reproduce_all`.
2. **Pedagogical clarity.** The simulations are intentionally small and auditable. Changes that dramatically increase runtime, add heavy dependencies (PyTorch, TensorFlow, Gym), or obscure the algorithm with abstraction are unlikely to be merged.

## Development setup

```bash
git clone https://github.com/<user>/rl-spatial-decision-making.git
cd rl-spatial-decision-making
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pytest -q
```

## Code style

- Python 3.9+ syntax.
- Type hints on public functions.
- Docstrings on every module and public function; the module docstring should explain what figure in the manuscript the file reproduces.
- Keep line length ≤ 100 characters.
- Use `src.utils.set_seed()` at the start of every stochastic run rather than `np.random.seed()` directly.
- Save figures via `src.utils.save_figure()`, not `plt.savefig()`.

## Adding a new simulation

If you add a simulation that illustrates a cluster not currently covered (e.g. hierarchical RL, transfer learning, POMDPs):

1. Create `src/simulations/simNN_<short_name>.py` following the structure of `sim01_qlearning_gridworld.py`: module docstring → env dataclass → training function → plotting function → `main()` with argparse.
2. Add a matching smoke test in `tests/test_simulations.py`.
3. Register the module in `src/reproduce_all.py`.
4. Update the README's figure table.

## Pull request checklist

- [ ] `pytest -q` passes.
- [ ] `python -m src.reproduce_all --quick` runs without errors.
- [ ] New figures, if any, are added to `figures/`.
- [ ] README updated if user-facing behaviour changes.
- [ ] Commit messages are descriptive; avoid "wip" / "fix stuff".
