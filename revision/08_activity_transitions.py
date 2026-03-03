"""
Script 08: Activity Transition Analysis
========================================

Generates two publication-quality PDF figures showing activity transition
patterns across the entire synthetic population (3.23 million persons).
These replace the old low-resolution PNG figures (top_transitions_updated.png
and transition_matrix_heatmap_updated.png) that the reviewer flagged for
low DPI and small labels.

The analysis extracts consecutive activity transitions from every person's
minute-level activity sequence in the Phase 4 shards, maps ATUS codes to
seven major categories, and computes row-normalized transition probabilities.

Outputs:
  revision/figures/fig_activity_transitions_bar.pdf
  revision/figures/fig_activity_transition_matrix.pdf
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
import matplotlib.colors as mcolors

# Publication-quality defaults (matching other revision scripts)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST = Path("D:/PUMS_Enrichment/data/processed/phase4_shards/manifest.json")
FIG_DIR = Path("D:/PUMS_Enrichment/revision/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Eight major activity categories derived from ATUS 2-digit prefixes.
# Caregiving (03, 04) is merged into Household; Education (06) and rare
# codes go into Other for a cleaner visualization.
# Seven categories (ATUS tier-1 code 01 covers both sleeping and
# grooming, so we label it "Sleep/Pers. Care" rather than splitting
# into two categories with an empty one).
CATEGORY_ORDER = [
    "Sleep/Pers. Care",
    "Work",
    "Household",
    "Eating",
    "Leisure",
    "Travel",
    "Other",
]

# Map ATUS 2-digit major-category prefix to display category.
# ATUS tier-1 codes: 01 = Personal Care (incl. sleep, grooming),
# 02 = Household Activities (cooking, cleaning, yard work, etc.),
# 03/04 = Caring for household/non-household members,
# 05 = Work, 06 = Education, 07/08 = Consumer/Professional services,
# 09 = Household services, 10 = Civic, 11 = Eating, 12/13 = Leisure/Sports,
# 14 = Religious, 15 = Volunteer, 16 = Phone calls, 18 = Travel.
PREFIX_TO_CATEGORY = {
    "01": "Sleep/Pers. Care", # Personal Care (sleep dominates this tier)
    "02": "Household",       # Household activities (cooking, cleaning)
    "03": "Household",       # Caring for household members
    "04": "Household",       # Caring for non-household members
    "05": "Work",
    "06": "Other",           # Education
    "07": "Other",           # Consumer purchases
    "08": "Other",           # Professional services
    "09": "Household",       # Household services
    "10": "Other",           # Civic obligations (rare)
    "11": "Eating",
    "12": "Leisure",         # Socializing, relaxing
    "13": "Leisure",         # Sports, exercise, recreation
    "14": "Other",           # Religious
    "15": "Other",           # Volunteer
    "16": "Other",           # Telephone calls
    "17": "Other",           # (secondary codes)
    "18": "Travel",
    "19": "Other",           # Misc
    "50": "Other",           # Data codes (not actual activities)
}

# Colors for bar chart (one per category pair -- we use a gradient palette)
BAR_COLOR = "#2C6FAC"
BAR_EDGE = "#1A4A7A"

# Heatmap colormap
HEATMAP_CMAP = "YlOrRd"


# ---------------------------------------------------------------------------
# Shard iterator (same pattern as other revision scripts)
# ---------------------------------------------------------------------------
def iterate_shards():
    """Yield each shard DataFrame from the Phase 4 manifest."""
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
            with open(p, "rb") as fh:
                df = pickle.load(fh)
            if isinstance(df, pd.DataFrame):
                yield df
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Map an ATUS activity code to one of 8 categories
# ---------------------------------------------------------------------------
def code_to_category(code):
    """Map an ATUS activity code string to one of 7 major categories.

    ATUS codes are hierarchical (e.g., tier-2 code 0111 = sleeping).
    In the Phase 4 data, leading zeros are stripped, so code 0111 is
    stored as '111'.  We zero-pad to 4 digits before extracting the
    2-digit major-category prefix.
    """
    code_str = str(code).strip()
    if len(code_str) < 1 or not code_str.replace(".", "").isdigit():
        return "Other"
    # Remove any decimal part (e.g., '111.0' -> '111')
    if "." in code_str:
        code_str = code_str.split(".")[0]
    # Pad to at least 4 digits so that code '111' becomes '0111',
    # giving prefix '01' (Personal Care / Sleep) instead of '11' (Eating).
    code_str = code_str.zfill(4)
    prefix = code_str[:2]
    return PREFIX_TO_CATEGORY.get(prefix, "Other")


# ---------------------------------------------------------------------------
# Main accumulation and plotting
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 08: Activity Transition Analysis")
    print("=" * 60)

    n_cats = len(CATEGORY_ORDER)
    cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}

    # Transition count matrix: counts[from_cat_idx, to_cat_idx]
    counts = np.zeros((n_cats, n_cats), dtype=np.int64)

    n_buildings = 0
    n_persons = 0
    n_transitions = 0
    n_shards = 0

    for df in iterate_shards():
        n_shards += 1
        if n_shards % 50 == 0:
            print(f"  Processing shard {n_shards} ... "
                  f"({n_buildings:,} buildings, {n_persons:,} persons, "
                  f"{n_transitions:,} transitions so far)")

        for _, row in df.iterrows():
            persons = row.get("persons", [])
            if not isinstance(persons, list) or len(persons) == 0:
                continue

            n_buildings += 1

            for person in persons:
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list) or len(acts) < 2:
                    continue

                n_persons += 1

                # Sort by activity_num to ensure correct ordering
                try:
                    sorted_acts = sorted(acts, key=lambda a: int(a.get("activity_num", 0)))
                except (ValueError, TypeError):
                    sorted_acts = acts

                # Extract category sequence
                categories = []
                for act in sorted_acts:
                    code = act.get("activity_code", "")
                    cat = code_to_category(code)
                    categories.append(cat)

                # Count transitions (consecutive category changes)
                for i in range(len(categories) - 1):
                    from_cat = categories[i]
                    to_cat = categories[i + 1]
                    fi = cat_idx.get(from_cat)
                    ti = cat_idx.get(to_cat)
                    if fi is not None and ti is not None:
                        counts[fi, ti] += 1
                        n_transitions += 1

        del df
        gc.collect()

    print(f"\nDone streaming. {n_shards} shards, "
          f"{n_buildings:,} buildings, {n_persons:,} persons, "
          f"{n_transitions:,} transitions.")

    # ------------------------------------------------------------------
    # Compute row-normalized transition probabilities
    # ------------------------------------------------------------------
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    # Print summary
    print("\nTransition probability matrix:")
    header = "".join(f"{c[:6]:>8s}" for c in CATEGORY_ORDER)
    print(f"{'':>14s}{header}")
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = "".join(f"{probs[i, j]:8.3f}" for j in range(n_cats))
        print(f"{cat:>14s}{vals}")

    # ------------------------------------------------------------------
    # Figure 1: Top 15 transitions bar chart
    # ------------------------------------------------------------------
    print("\nGenerating bar chart ...")

    # Collect all transition pairs with their probabilities
    trans_list = []
    for i in range(n_cats):
        for j in range(n_cats):
            if probs[i, j] > 0:
                label = f"{CATEGORY_ORDER[i]} -> {CATEGORY_ORDER[j]}"
                trans_list.append((label, probs[i, j], counts[i, j]))

    # Sort by probability descending, take top 15
    trans_list.sort(key=lambda x: x[1], reverse=True)
    top_15 = trans_list[:15]

    # Reverse for horizontal bar chart (highest at top)
    labels = [t[0] for t in top_15][::-1]
    values = [t[1] for t in top_15][::-1]
    raw_counts = [t[2] for t in top_15][::-1]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, height=0.65, color=BAR_COLOR,
                   edgecolor=BAR_EDGE, linewidth=0.6)

    # Add probability labels to the right of each bar
    for bar_obj, val, cnt in zip(bars, values, raw_counts):
        ax.text(bar_obj.get_width() + 0.008, bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left", fontsize=10, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Transition Probability", fontsize=12)
    ax.set_title("Top 15 Activity Transitions by Probability", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=10)

    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    bar_path = FIG_DIR / "fig_activity_transitions_bar.pdf"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # ------------------------------------------------------------------
    # Figure 2: Transition probability heatmap
    # ------------------------------------------------------------------
    print("\nGenerating transition matrix heatmap ...")

    # Labels for axes
    short_labels = list(CATEGORY_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    im = ax.imshow(probs, cmap=HEATMAP_CMAP, aspect="auto",
                   vmin=0, vmax=min(probs.max() * 1.05, 1.0))

    # Add text annotations
    for i in range(n_cats):
        for j in range(n_cats):
            val = probs[i, j]
            if val < 0.005:
                continue
            # Use white text on dark cells, black on light cells
            text_color = "white" if val > 0.35 else "black"
            fontsize = 11 if val >= 0.1 else 9
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=text_color, fontweight="bold"
                    if val >= 0.1 else "normal")

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(short_labels, fontsize=11, rotation=35, ha="right")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(short_labels, fontsize=11)

    ax.set_xlabel("To Activity", fontsize=12, labelpad=8)
    ax.set_ylabel("From Activity", fontsize=12, labelpad=8)
    ax.set_title("Activity Transition Probability Matrix", fontsize=14,
                 fontweight="bold", pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    heatmap_path = FIG_DIR / "fig_activity_transition_matrix.pdf"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------
    prob_df = pd.DataFrame(probs, index=CATEGORY_ORDER, columns=CATEGORY_ORDER)
    csv_path = FIG_DIR / "activity_transition_probabilities.csv"
    prob_df.to_csv(csv_path, float_format="%.4f")
    print(f"Saved summary: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
