"""
plot_figure5_clean.py

Generate the task-level F1-score improvement figure reported in the manuscript.

Output formats
--------------
PNG
PDF
SVG
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Output directory
# =========================

OUTPUT_DIR = Path("outputs") / "figs_plosone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Figure style
# =========================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)


# =========================
# Experimental results
# =========================

tasks = [
    "Commonsense",
    "Deontology",
    "Justice",
    "Virtue",
    "Utilitarianism",
]

improvement = np.array(
    [
        9.6,
        10.6,
        9.7,
        12.3,
        13.3,
    ],
    dtype=float,
)

# Display the largest improvement at the top.
tasks = tasks[::-1]
improvement = improvement[::-1]


# =========================
# Create figure
# =========================

figure, axis = plt.subplots(
    figsize=(7.0, 4.5)
)

colors = plt.cm.Blues(
    np.linspace(
        0.50,
        0.90,
        len(tasks),
    )
)

bars = axis.barh(
    tasks,
    improvement,
    color=colors,
    edgecolor="black",
    linewidth=0.6,
)


# =========================
# Value labels
# =========================

for bar, value in zip(
    bars,
    improvement,
):
    axis.text(
        value + 0.20,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        fontsize=10,
    )


# =========================
# Axis settings
# =========================

axis.set_xlabel(
    "Relative F1-score Improvement (%)"
)

axis.set_ylabel(
    "Ethical Task"
)

axis.set_xlim(
    0,
    15,
)

axis.grid(
    axis="x",
    linestyle="--",
    alpha=0.35,
)


# =========================
# Spine style
# =========================

axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)

axis.spines["left"].set_linewidth(1.0)
axis.spines["bottom"].set_linewidth(1.0)


# =========================
# Save figure
# =========================

figure.tight_layout()

for extension in (
    "png",
    "pdf",
    "svg",
):
    figure.savefig(
        OUTPUT_DIR
        / f"Figure5_clean.{extension}",
        bbox_inches="tight",
        dpi=600 if extension == "png" else None,
    )

plt.close(figure)

print(
    f"Figure saved successfully: {OUTPUT_DIR}"
)
