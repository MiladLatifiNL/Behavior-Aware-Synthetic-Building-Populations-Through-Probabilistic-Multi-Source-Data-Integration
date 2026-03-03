"""
Script 03: Cross-Domain Decoupling -- Building-Type Metrics & State Correlation

Streams ALL 874 Phase 4 shards to produce two publication-quality figures:
  A) Building-Type Metrics  -- grouped bar chart of 5 normalised metrics per type
  B) State Correlation Matrix -- annotated Pearson-r heatmap across 6 metrics

Memory-efficient: accumulates sums / counts in dictionaries, then computes means.

Outputs (300 DPI PDF):
  revision/figures/fig_building_type_metrics.pdf
  revision/figures/fig_state_correlation.pdf
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import pickle
import gc
from pathlib import Path
from collections import defaultdict

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
# Metrics of interest
# ---------------------------------------------------------------------------
METRICS = ["energy_intensity", "energy_burden", "household_size",
           "rooms_per_person", "occupancy_intensity"]

METRIC_LABELS = {
    "energy_intensity": "Energy\nIntensity",
    "energy_burden": "Energy\nBurden",
    "household_size": "Household\nSize",
    "rooms_per_person": "Rooms per\nPerson",
    "occupancy_intensity": "Occupancy\nIntensity",
}

STATE_METRICS = ["energy_intensity", "energy_burden", "household_size",
                 "rooms_per_person", "building_age", "occupancy_intensity"]

STATE_METRIC_LABELS = {
    "energy_intensity": "Energy Intensity",
    "energy_burden": "Energy Burden",
    "household_size": "Household Size",
    "rooms_per_person": "Rooms / Person",
    "building_age": "Building Age",
    "occupancy_intensity": "Occupancy Intensity",
}

# Canonical building-type order
BTYPE_ORDER = ["Single Family", "Small Multi", "Large Multi",
               "Mobile Home", "Other"]

# Map raw values to canonical labels
BTYPE_MAP = {
    "single_family": "Single Family",
    "small_multi": "Small Multi",
    "large_multi": "Large Multi",
    "mobile": "Mobile Home",
    "other": "Other",
}

# ---------------------------------------------------------------------------
# Accumulators: sums & counts per (group, metric)
# ---------------------------------------------------------------------------
# Building-type accumulators
bt_sums = defaultdict(lambda: defaultdict(float))   # bt_sums[btype][metric]
bt_counts = defaultdict(lambda: defaultdict(int))

# State accumulators
st_sums = defaultdict(lambda: defaultdict(float))   # st_sums[state][metric]
st_counts = defaultdict(lambda: defaultdict(int))

# ---------------------------------------------------------------------------
# Stream over shards
# ---------------------------------------------------------------------------
print("Streaming 874 shards ...")
n_shards = 0
n_buildings = 0

for df in iterate_shards():
    n_shards += 1
    n_buildings += len(df)

    # --- Building-type accumulation ---
    if "building_type_simple" in df.columns:
        for metric in METRICS:
            if metric not in df.columns:
                continue
            valid = df[["building_type_simple", metric]].dropna(subset=[metric])
            for btype_raw, group in valid.groupby("building_type_simple"):
                btype = BTYPE_MAP.get(str(btype_raw), str(btype_raw))
                bt_sums[btype][metric] += group[metric].sum()
                bt_counts[btype][metric] += len(group)

    # --- State accumulation ---
    if "STATE" in df.columns:
        for metric in STATE_METRICS:
            if metric not in df.columns:
                continue
            valid = df[["STATE", metric]].dropna(subset=[metric])
            for state, group in valid.groupby("STATE"):
                st_sums[state][metric] += group[metric].sum()
                st_counts[state][metric] += len(group)

    if n_shards % 100 == 0:
        print(f"  processed {n_shards} shards, {n_buildings:,} buildings ...")
    del df
    gc.collect()

print(f"Finished streaming: {n_shards} shards, {n_buildings:,} buildings.")

# ---------------------------------------------------------------------------
# Compute means
# ---------------------------------------------------------------------------
# Building-type means
bt_means = {}  # {btype: {metric: mean}}
for btype in BTYPE_ORDER:
    bt_means[btype] = {}
    for metric in METRICS:
        c = bt_counts[btype][metric]
        bt_means[btype][metric] = bt_sums[btype][metric] / c if c > 0 else np.nan

# State means
states_with_data = sorted(st_sums.keys())
state_df = pd.DataFrame(index=states_with_data, columns=STATE_METRICS, dtype=float)
for state in states_with_data:
    for metric in STATE_METRICS:
        c = st_counts[state][metric]
        state_df.loc[state, metric] = (
            st_sums[state][metric] / c if c > 0 else np.nan
        )
state_df = state_df.dropna(how="any")

# ===================================================================
# FIGURE A: Building-Type Grouped Bar Chart
# ===================================================================
# Build data matrix and normalize to [0, 1]
raw = np.zeros((len(BTYPE_ORDER), len(METRICS)))
for i, btype in enumerate(BTYPE_ORDER):
    for j, metric in enumerate(METRICS):
        raw[i, j] = bt_means[btype].get(metric, np.nan)

# Min-max normalize each metric (column)
normed = np.zeros_like(raw)
for j in range(raw.shape[1]):
    col = raw[:, j]
    col_min = np.nanmin(col)
    col_max = np.nanmax(col)
    if col_max - col_min > 0:
        normed[:, j] = (col - col_min) / (col_max - col_min)
    else:
        normed[:, j] = 0.5

fig, ax = plt.subplots(figsize=(10, 6))

n_types = len(BTYPE_ORDER)
n_metrics = len(METRICS)
bar_width = 0.15
x = np.arange(n_types)

colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB", "#8E44AD"]

for j, metric in enumerate(METRICS):
    offset = (j - n_metrics / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, normed[:, j], width=bar_width,
                  color=colors[j], edgecolor="white", linewidth=0.4,
                  label=METRIC_LABELS[metric].replace("\n", " "))

ax.set_xticks(x)
ax.set_xticklabels(BTYPE_ORDER, fontsize=11)
ax.set_xlabel("Building type", fontsize=13)
ax.set_ylabel("Normalized metric value [0, 1]", fontsize=13)
ax.set_ylim(0, 1.15)

ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=True,
          fancybox=True, shadow=False)

ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_a = OUT_DIR / "fig_building_type_metrics.pdf"
fig.savefig(out_a, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"[DONE] Saved: {out_a}")

# Print raw means for reference
print("\n--- Building-Type Raw Means ---")
for btype in BTYPE_ORDER:
    vals = "  ".join(f"{m}: {bt_means[btype].get(m, float('nan')):.4f}" for m in METRICS)
    print(f"  {btype:15s}  {vals}")

# ===================================================================
# FIGURE B: State Correlation Matrix
# ===================================================================
corr = state_df.astype(float).corr(method="pearson")

fig, ax = plt.subplots(figsize=(8, 6))

# Diverging colormap: blue-white-red
cmap = plt.cm.RdBu_r
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

n = len(STATE_METRICS)
for i in range(n):
    for j in range(n):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold" if i == j else "normal",
                color=color)

# Labels
tick_labels = [STATE_METRIC_LABELS[m] for m in STATE_METRICS]
ax.set_xticks(range(n))
ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=10)
ax.set_yticks(range(n))
ax.set_yticklabels(tick_labels, fontsize=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=12)

plt.tight_layout()
out_b = OUT_DIR / "fig_state_correlation.pdf"
fig.savefig(out_b, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"[DONE] Saved: {out_b}")

# Print key correlation
ei_idx = STATE_METRICS.index("energy_intensity")
eb_idx = STATE_METRICS.index("energy_burden")
r_val = corr.values[ei_idx, eb_idx]
print(f"\n*** Key correlation (state-level): energy_intensity vs energy_burden  r = {r_val:.4f} ***")
print(f"    (n = {len(state_df)} states)")

# ===================================================================
# BUILDING-LEVEL CORRELATION: energy_intensity vs energy_burden
# ===================================================================
# Second pass over shards to compute building-level Pearson r using
# online accumulation (avoids loading all 1.3M rows into memory).
print("\nComputing building-level correlation (energy_intensity vs energy_burden) ...")

bl_n = 0
bl_sx = 0.0
bl_sy = 0.0
bl_sx2 = 0.0
bl_sy2 = 0.0
bl_sxy = 0.0

for df in iterate_shards():
    if "energy_intensity" in df.columns and "energy_burden" in df.columns:
        sub = df[["energy_intensity", "energy_burden"]].dropna()
        # Filter out extreme outliers and zero/negative income proxies
        sub = sub[(sub["energy_burden"] > 0) & (sub["energy_burden"] < 100)]
        x = sub["energy_intensity"].values.astype(np.float64)
        y = sub["energy_burden"].values.astype(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n > 0:
            bl_n += n
            bl_sx += float(x.sum())
            bl_sy += float(y.sum())
            bl_sx2 += float((x * x).sum())
            bl_sy2 += float((y * y).sum())
            bl_sxy += float((x * y).sum())
    del df
    gc.collect()

if bl_n > 2:
    num = bl_n * bl_sxy - bl_sx * bl_sy
    den = np.sqrt((bl_n * bl_sx2 - bl_sx**2) * (bl_n * bl_sy2 - bl_sy**2))
    bl_r = num / den if den > 0 else float("nan")
else:
    bl_r = float("nan")

print(f"*** Key correlation (building-level): energy_intensity vs energy_burden  r = {bl_r:.4f} ***")
print(f"    (n = {bl_n:,} buildings with valid data)")
