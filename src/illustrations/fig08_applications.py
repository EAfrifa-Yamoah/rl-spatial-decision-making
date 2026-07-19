"""
Figure 8 — Application landscape of references cited in this review.

Two-panel summary computed from the manuscript's reference list:
  (a) horizontal bar chart of cited references per application domain.
      Each of the 98 references is assigned to exactly one primary
      domain (its most natural citation context in the manuscript).
  (b) donut chart showing the split between applied-domain references
      and foundational/methodological references (RL algorithms, theory,
      and general reviews that are not tied to any single application).

These counts are derived from this manuscript's reference list itself,
not from external publication databases. The classification is reproducible
via docs/reference_classification.py.

v4 (revision) expands the corpus from 80 to 98 references to address
reviewer 1's request for stronger coverage of GIScience, remote sensing,
transportation MARL, and spatial resource allocation.

Run:
    python -m src.illustrations.fig08_applications
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_figure

# ---- Data (derived from classifying the manuscript's 98 references) ----
# Panel (a): applied-domain reference counts.
# Classification is one-reference-to-one-domain (primary citation context).
DOMAIN_LABELS = [
    "Conservation &\nwildlife protection",
    "Sensor placement &\nadaptive sensing",
    "GIS-RL\nintegration",
    "Autonomous\nvehicles",
    "Transportation &\ntraffic MARL",
    "Robot\nnavigation",
    "Forestry &\nwildfire management",
    "Water & reservoir\nmanagement",
    "Remote sensing\nRL",
    "Environmental\nmonitoring",
    "Urban / facility\nplanning",
    "UAV / swarm\ncoordination",
]
DOMAIN_COUNTS = [6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 1]

TOTAL_REFS = 98
APPLIED_TOTAL = sum(DOMAIN_COUNTS)         # 41
FOUNDATIONAL_TOTAL = TOTAL_REFS - APPLIED_TOTAL  # 57
SPLIT_LABELS = [
    f"Applied-domain\nreferences (n = {APPLIED_TOTAL})",
    f"Foundational /\nmethodological\nreferences (n = {FOUNDATIONAL_TOTAL})",
]
SPLIT_VALUES = [APPLIED_TOTAL, FOUNDATIONAL_TOTAL]
SPLIT_COLORS = ["#1565c0", "#9e9e9e"]


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(14, 6.5))

    # (a) bar chart of applied-domain counts
    ax1 = plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(DOMAIN_LABELS)))
    bars = ax1.barh(range(len(DOMAIN_LABELS)), DOMAIN_COUNTS,
                    color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_yticks(range(len(DOMAIN_LABELS)))
    ax1.set_yticklabels(DOMAIN_LABELS, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of cited references", fontsize=10)
    ax1.set_title(f"(a) Applied-domain references (n = {APPLIED_TOTAL} of {TOTAL_REFS})",
                  fontsize=11.5, fontweight="bold")
    ax1.set_xlim(0, max(DOMAIN_COUNTS) + 2)
    ax1.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, DOMAIN_COUNTS):
        ax1.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=9.5, color="#333",
                 fontweight="bold")

    # (b) donut: applied vs foundational
    ax2 = plt.subplot(1, 2, 2)
    wedges, texts, autotexts = ax2.pie(
        SPLIT_VALUES,
        labels=SPLIT_LABELS,
        colors=SPLIT_COLORS,
        startangle=90,
        counterclock=False,
        autopct="%1.0f%%",
        pctdistance=0.78,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(13)
    ax2.text(0, 0, f"n = {TOTAL_REFS}", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#333")
    ax2.set_title("(b) Composition of the reference corpus",
                  fontsize=11.5, fontweight="bold")

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", type=str, default="fig8_applications")
    args = parser.parse_args()

    fig = build_figure()
    out = save_figure(fig, args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig08] Saved figure to {out}")
    print(f"[fig08] Applied-domain total: {APPLIED_TOTAL}")
    print(f"[fig08] Foundational total:   {FOUNDATIONAL_TOTAL}")


if __name__ == "__main__":
    main()
