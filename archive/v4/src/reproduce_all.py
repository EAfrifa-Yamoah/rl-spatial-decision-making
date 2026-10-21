"""
Reproduce all eight figures from the manuscript in a single command.

Runs each simulation / illustration in sequence with its default
hyperparameters, producing `figures/fig{1..8}_*.png` at 400 DPI.

Usage:
    python -m src.reproduce_all
    python -m src.reproduce_all --dpi 200        # faster, screen-quality
    python -m src.reproduce_all --quick          # fewer episodes

Environment:
    Requires numpy, scipy, matplotlib. See requirements.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Tuple

from src.illustrations import (
    fig01_mdp_framework,
    fig03_taxonomy,
    fig07_flowchart,
    fig08_applications,
)
from src.simulations import (
    sim01_qlearning_gridworld,
    sim02_multi_agent,
    sim03_adaptive_sensor,
    sim04_dynamic_obstacle,
)


def _run(label: str, argv: List[str], entry: Callable[[], None]) -> float:
    """Run one module's main() with argv injected into sys.argv."""
    print(f"\n[{label}] starting")
    t0 = time.time()
    old_argv = sys.argv
    try:
        sys.argv = [f"{label}"] + argv
        entry()
    finally:
        sys.argv = old_argv
    dt = time.time() - t0
    print(f"[{label}] done in {dt:.1f}s")
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=400,
                        help="Output DPI passed to every figure (default: 400)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced episode counts for a faster run")
    args = parser.parse_args()

    dpi_args = ["--dpi", str(args.dpi)]
    sim1_args = dpi_args + (["--episodes", "200"] if args.quick else [])
    sim2_args = dpi_args + (["--episodes", "300"] if args.quick else [])
    sim3_args = dpi_args + (["--replicates", "3"] if args.quick else [])
    sim4_args = dpi_args + (["--episodes", "1500"] if args.quick else [])

    jobs: List[Tuple[str, List[str], Callable[[], None]]] = [
        ("fig01", dpi_args, fig01_mdp_framework.main),
        ("sim01", sim1_args, sim01_qlearning_gridworld.main),
        ("fig03", dpi_args, fig03_taxonomy.main),
        ("sim02", sim2_args, sim02_multi_agent.main),
        ("sim03", sim3_args, sim03_adaptive_sensor.main),
        ("sim04", sim4_args, sim04_dynamic_obstacle.main),
        ("fig07", dpi_args, fig07_flowchart.main),
        ("fig08", dpi_args, fig08_applications.main),
    ]

    total = 0.0
    for label, argv, entry in jobs:
        total += _run(label, argv, entry)

    print(f"\nAll figures produced in {total:.1f}s total. "
          f"See the `figures/` directory.")


if __name__ == "__main__":
    main()
