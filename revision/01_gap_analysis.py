"""
Script 01: Information Coverage Heatmap (Gap Analysis)

Generates a publication-quality heatmap showing which analytical capabilities
are available in each source dataset (PUMS, RECS, ATUS, NSRDB) versus the
Unified synthetic population. All data is hardcoded -- no shard loading needed.

Output: revision/figures/fig_gap_analysis_heatmap.pdf (300 DPI)
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication-quality rc settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Use plain-text symbols that render in all serif fonts
# (Unicode checkmark/ballot-x may be absent from DejaVu Serif)

# ---------------------------------------------------------------------------
# Data: rows = analytical capabilities, columns = data sources
# Values: 1.0 = available, 0.5 = partial, 0.0 = unavailable
# ---------------------------------------------------------------------------
capabilities = [
    "Household demographics\n(age, sex, income, education)",
    "Building envelope\n(sqft, insulation, HVAC)",
    "Energy consumption\n(kWh, cost, end-use)",
    "Minute-level activities\n(time-use diaries)",
    "Hourly weather\n(temp, solar, wind)",
    "Household coordination\n(childcare, meals, sleep)",
    "Income-stratified\nenergy burden",
    "Activity-aware\noccupancy profiles",
    "Weather-behavior\ninteraction",
    "Cross-domain\ncouplings analysis",
    "EV adoption by\ndemographic profile",
    "Population scale\n(>1M buildings)",
]

sources = ["PUMS", "RECS", "ATUS", "NSRDB", "Unified"]

# fmt: off
matrix = np.array([
    # PUMS  RECS  ATUS  NSRDB  Unified
    [ 1.0,  0.5,  1.0,  0.0,   1.0],  # Household demographics
    [ 0.0,  1.0,  0.0,  0.0,   1.0],  # Building envelope
    [ 0.0,  1.0,  0.0,  0.0,   1.0],  # Energy consumption
    [ 0.0,  0.0,  1.0,  0.0,   1.0],  # Minute-level activities
    [ 0.0,  0.0,  0.0,  1.0,   1.0],  # Hourly weather
    [ 0.0,  0.0,  0.0,  0.0,   1.0],  # Household coordination
    [ 0.0,  0.5,  0.0,  0.0,   1.0],  # Income-stratified energy burden
    [ 0.0,  0.0,  0.5,  0.0,   1.0],  # Activity-aware occupancy
    [ 0.0,  0.0,  0.0,  0.0,   1.0],  # Weather-behavior interaction
    [ 0.0,  0.0,  0.0,  0.0,   1.0],  # Cross-domain couplings
    [ 0.5,  0.0,  0.0,  0.0,   1.0],  # EV adoption
    [ 1.0,  0.0,  0.0,  1.0,   1.0],  # Population scale
])
# fmt: on

# Annotation symbols -- plain ASCII to avoid missing-glyph warnings
symbol_map = {1.0: "Y", 0.5: "~", 0.0: "N"}

# ---------------------------------------------------------------------------
# Colormap: red/gray -> amber/yellow -> green
# ---------------------------------------------------------------------------
cmap = mcolors.ListedColormap(["#D9534F", "#F0AD4E", "#5CB85C"])
bounds = [-0.25, 0.25, 0.75, 1.25]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

# Annotate each cell
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        sym = symbol_map[val]
        # Use white text on dark cells, black on yellow
        color = "white" if val in (0.0, 1.0) else "black"
        fontweight = "bold" if j == len(sources) - 1 else "normal"
        ax.text(j, i, sym, ha="center", va="center",
                fontsize=14, fontweight=fontweight, color=color)

# Vertical separator line before the "Unified" column
sep_x = len(sources) - 1.5
ax.axvline(x=sep_x, color="black", linewidth=2.0, linestyle="-")

# Horizontal dashed line separating source-available from integration-only
# capabilities (between row 4 "Hourly weather" and row 5 "Household coordination")
ax.axhline(y=4.5, color="black", linewidth=1.2, linestyle="--")

# Highlight the Unified column with a subtle border
for i in range(matrix.shape[0]):
    rect = Rectangle((len(sources) - 1.5, i - 0.5), 1, 1,
                      linewidth=0.8, edgecolor="#333333",
                      facecolor="none", zorder=3)
    ax.add_patch(rect)

# Axis labels
ax.set_xticks(range(len(sources)))
ax.set_xticklabels(sources, fontsize=12, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

ax.set_yticks(range(len(capabilities)))
ax.set_yticklabels(capabilities, fontsize=10)

# Title
ax.set_title("Analytical Capability Coverage by Data Source",
             fontsize=14, fontweight="bold", pad=18)

# Remove spines for cleanliness
for spine in ax.spines.values():
    spine.set_visible(False)

# Grid lines between cells
ax.set_xticks([x - 0.5 for x in range(1, len(sources))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(capabilities))], minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#5CB85C", edgecolor="gray", label="Available (Y)"),
    Patch(facecolor="#F0AD4E", edgecolor="gray", label="Partial (~)"),
    Patch(facecolor="#D9534F", edgecolor="gray", label="Unavailable (N)"),
]
ax.legend(handles=legend_elements, loc="lower center",
          bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10,
          frameon=True, fancybox=True, shadow=False)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path("d:/PUMS_Enrichment/revision/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_gap_analysis_heatmap.pdf"
fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"[DONE] Saved: {out_path}")
