"""Figure 7: composition of the reference list, from docs/coding_sheet.csv."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]


def main():
    S.apply()
    rows = list(csv.DictReader(open(ROOT / "docs/coding_sheet.csv")))
    applied = [r for r in rows if r["category"] == "applied"]
    c = Counter(r["primary_domain"] for r in applied)
    items = sorted(c.items(), key=lambda kv: (kv[1], kv[0]))
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    y = range(len(items))
    ax.barh(list(y), [v for _, v in items], color=S.GREY_DARK, height=0.62)
    for yi, (_, v) in zip(y, items):
        ax.text(v + 0.12, yi, str(v), va="center", ha="left", fontsize=6.5, color=S.INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([k for k, _ in items])
    ax.set_xlabel(f"Number of references (applied domains: {len(applied)} of {len(rows)})")
    ax.set_xlim(0, max(c.values()) + 1.2)
    ax.grid(axis="x", color=S.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    S.save(fig, ROOT / "results/figures/Figure_7_References")
    return len(rows), len(applied), dict(c)


if __name__ == "__main__":
    print(main())
