"""Regenerate every raw result used in the manuscript and supplement.

Usage (from the repository root):
    python -m src.reproduce_all            # all four problems
    python -m src.reproduce_all A C        # a subset

Writes results/raw/problem_{a,b,c,d}.json and results/raw/environment.json.
Then run
    python -m tools.build_manifest         # every reported number -> results/manifest.json
    python -m src.figures.make_all         # every figure -> results/figures/
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy
import scipy

from src.simulations import problem_a, problem_b, problem_c, problem_d

OUT = Path(__file__).resolve().parent.parent / "results" / "raw"
MODULES = {"A": problem_a, "B": problem_b, "C": problem_c, "D": problem_d}


def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "timing_s": {}}
    env_path = OUT / "environment.json"
    if env_path.exists():
        env["timing_s"] = json.loads(env_path.read_text()).get("timing_s", {})
    for key in which:
        t0 = time.perf_counter()
        print(f"Problem {key} ...", flush=True)
        MODULES[key].main(OUT)
        env["timing_s"][key] = round(time.perf_counter() - t0, 1)
        print(f"Problem {key} done in {env['timing_s'][key]} s", flush=True)
        env_path.write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODULES))
