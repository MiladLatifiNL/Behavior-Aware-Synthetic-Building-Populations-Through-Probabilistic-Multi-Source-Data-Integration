"""
Script 02: Structural Fidelity -- Age Pyramid & Income Distribution

Streams ALL 874 Phase 4 shards to produce two publication-quality figures:
  A) Age Pyramid  -- horizontal population pyramid by sex and 5-year age bands
  B) Income Distribution -- histogram of household income (HINCP)

Memory-efficient: accumulates counts in fixed-size arrays / dicts, never stores
all person records simultaneously.

Outputs (300 DPI PDF):
  revision/figures/fig_age_pyramid.pdf
  revision/figures/fig_income_distribution.pdf
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import pickle
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Publication-quality rc settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

OUT_DIR = Path("d:/PUMS_Enrichment/revision/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("d:/PUMS_Enrichment/data/processed/phase4_shards/manifest.json")

# ---------------------------------------------------------------------------
# Shard iterator
# ---------------------------------------------------------------------------
def iterate_shards():
    """Yield each shard DataFrame."""
    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest_dir = MANIFEST.parent
    for fp in manifest["files"]:
        p = Path(fp)
        if not p.exists():
            p = manifest_dir / p.name
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                yield df
        except Exception:
            continue

# ---------------------------------------------------------------------------
# Age band helpers
# ---------------------------------------------------------------------------
AGE_BANDS = [(i, i + 4) for i in range(0, 85, 5)] + [(85, 120)]
AGE_LABELS = [f"{lo}-{hi}" if hi < 120 else "85+" for lo, hi in AGE_BANDS]
N_BANDS = len(AGE_BANDS)

def age_to_band(age):
    """Return band index for a given age."""
    if age is None or np.isnan(age):
        return -1
    age = int(age)
    if age < 0:
        return -1
    if age >= 85:
        return N_BANDS - 1
    return age // 5

# ---------------------------------------------------------------------------
# Income bin helpers
# ---------------------------------------------------------------------------
INCOME_EDGES = list(range(0, 210_000, 10_000)) + [np.inf]
INCOME_LABELS = [f"${e//1000}k-${(e+10_000)//1000}k" for e in range(0, 200_000, 10_000)]
INCOME_LABELS.append(">$200k")

# ---------------------------------------------------------------------------
# Stream over shards and accumulate
# ---------------------------------------------------------------------------
print("Streaming 874 shards ...")

# Accumulators
male_counts = np.zeros(N_BANDS, dtype=np.int64)
female_counts = np.zeros(N_BANDS, dtype=np.int64)
income_counts = np.zeros(len(INCOME_EDGES) - 1, dtype=np.int64)

n_shards = 0
n_persons_total = 0
n_buildings_total = 0

for df in iterate_shards():
    n_shards += 1
    n_buildings_total += len(df)

    # --- Person data (age pyramid) ---
    for persons_list in df["persons"]:
        if not isinstance(persons_list, list):
            continue
        for person in persons_list:
            n_persons_total += 1
            agep = person.get("AGEP")
            sex = person.get("SEX")
            if agep is None or sex is None:
                continue
            try:
                age_val = float(agep)
                sex_val = int(sex)
            except (ValueError, TypeError):
                continue
            band = age_to_band(age_val)
            if band < 0:
                continue
            if sex_val == 1:
                male_counts[band] += 1
            elif sex_val == 2:
                female_counts[band] += 1

    # --- Household income (income distribution) ---
    if "HINCP" in df.columns:
        incomes = pd.to_numeric(df["HINCP"], errors="coerce").dropna()
        # Clip negative incomes to 0 for binning
        incomes = incomes.clip(lower=0)
        hist, _ = np.histogram(incomes.values, bins=INCOME_EDGES)
        income_counts += hist.astype(np.int64)

    # Progress
    if n_shards % 100 == 0:
        print(f"  processed {n_shards} shards, {n_persons_total:,} persons, "
              f"{n_buildings_total:,} buildings ...")
    del df
    gc.collect()

print(f"Finished streaming: {n_shards} shards, "
      f"{n_buildings_total:,} buildings, {n_persons_total:,} persons.")

# ===================================================================
# FIGURE A: Age Pyramid
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 6))

y_pos = np.arange(N_BANDS)
bar_height = 0.8

# Males on the left (negative), females on the right (positive)
ax.barh(y_pos, -male_counts, height=bar_height, color="#4A90D9",
        label="Male", edgecolor="white", linewidth=0.3)
ax.barh(y_pos, female_counts, height=bar_height, color="#E06B75",
        label="Female", edgecolor="white", linewidth=0.3)

# Y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(AGE_LABELS, fontsize=10)
ax.set_ylabel("Age group", fontsize=13)

# X-axis: show absolute counts, format nicely
max_val = max(male_counts.max(), female_counts.max())
ax.set_xlim(-max_val * 1.12, max_val * 1.12)

def thousands_formatter(x, pos):
    """Format tick as thousands with sign removed."""
    return f"{abs(x)/1000:.0f}k"

ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
ax.set_xlabel("Number of persons", fontsize=13)

# Center line
ax.axvline(0, color="black", linewidth=0.8)

# Legend
ax.legend(loc="lower right", fontsize=11, frameon=True)

# Subtle grid
ax.grid(axis="x", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_a = OUT_DIR / "fig_age_pyramid.pdf"
fig.savefig(out_a, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"[DONE] Saved: {out_a}")

# ===================================================================
# FIGURE B: Income Distribution
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 6))

x_pos = np.arange(len(income_counts))
bar_colors = ["#2C7BB6"] * (len(income_counts) - 1) + ["#D7191C"]  # last bin >$200k in red

ax.bar(x_pos, income_counts, width=0.85, color=bar_colors,
       edgecolor="white", linewidth=0.4)

# X-axis labels: show every other for readability
tick_labels_short = []
for i, e in enumerate(range(0, 200_000, 10_000)):
    tick_labels_short.append(f"${e//1000}k")
tick_labels_short.append(">$200k")

ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels_short, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Household income (HINCP)", fontsize=13)

# Y-axis
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, pos: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
ax.set_ylabel("Number of households", fontsize=13)

# Subtle grid
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_b = OUT_DIR / "fig_income_distribution.pdf"
fig.savefig(out_b, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"[DONE] Saved: {out_b}")
