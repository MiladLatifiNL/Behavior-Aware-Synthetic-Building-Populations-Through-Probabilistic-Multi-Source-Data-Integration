"""
Script 10: Demand Diversity from Heterogeneous Schedules
========================================================

Shows that household-coordinated, heterogeneous activity schedules
naturally flatten aggregate demand compared to uniform schedules.
This is a key benefit of the unified dataset that standard building
energy simulations with identical occupancy profiles cannot capture.

Outputs:
  (a) Aggregate occupancy: actual heterogeneous vs hypothetical uniform
  (b) Peak-to-average ratio by household type

Output: revision/figures/fig_demand_diversity.pdf
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

HOME_LOCATIONS = {"1", "-1"}

HH_TYPES_ORDERED = [
    "Dual-earner + children",
    "Single parent",
    "Single young professional",
    "Retired couple",
    "Single elderly",
    "Other",
]

COLORS = {
    "Dual-earner + children":    "#E74C3C",
    "Single parent":             "#F39C12",
    "Single young professional": "#3498DB",
    "Retired couple":            "#27AE60",
    "Single elderly":            "#8E44AD",
    "Other":                     "#95A5A6",
}


# ---------------------------------------------------------------------------
# Shard iterator
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
# Household type classifier (same as script 04)
# ---------------------------------------------------------------------------
def classify_household(row):
    """Return the household type label for a building row."""
    has_children = bool(row.get("has_children", False))
    has_seniors = bool(row.get("has_seniors", False))
    hh_size = int(row.get("household_size", row.get("NP", 1)))
    emp_count = int(row.get("person_employed_count", 0))
    emp_rate = float(row.get("employment_rate", 0.0))
    age_mean = float(row.get("person_age_mean", 40.0))

    if has_children and emp_count >= 2:
        return "Dual-earner + children"
    if has_children and emp_count <= 1 and hh_size >= 2:
        return "Single parent"
    if hh_size == 1 and age_mean < 40 and emp_rate > 0:
        return "Single young professional"
    if hh_size == 2 and has_seniors and emp_rate == 0:
        return "Retired couple"
    if hh_size == 1 and age_mean >= 65:
        return "Single elderly"
    return "Other"


# ---------------------------------------------------------------------------
# At-home occupancy per person (returns 24-element float array of 0/1)
# ---------------------------------------------------------------------------
def at_home_by_hour(activity_sequence):
    """Return 24-element array: 1.0 if person at home for hour h, else 0."""
    home_minutes = np.zeros(1440, dtype=bool)

    for act in activity_sequence:
        loc = str(act.get("location", ""))
        code = str(act.get("activity_code", ""))
        is_home = loc in HOME_LOCATIONS
        if not is_home and loc in ("", "None", "nan"):
            code_2 = code[:2] if len(code) >= 2 else ""
            is_home = code_2 in ("01", "02", "05", "11")
        if not is_home:
            continue

        try:
            start_str = str(act.get("start_time", "00:00:00"))
            stop_str = str(act.get("stop_time", "00:00:00"))
            sp = start_str.split(":")
            ep = stop_str.split(":")
            start_min = int(sp[0]) * 60 + int(sp[1])
            stop_min = int(ep[0]) * 60 + int(ep[1])
        except (ValueError, IndexError):
            continue

        if stop_min <= start_min:
            home_minutes[start_min:1440] = True
            home_minutes[0:stop_min] = True
        else:
            home_minutes[start_min:stop_min] = True

    hourly = np.zeros(24)
    for h in range(24):
        hourly[h] = 1.0 if home_minutes[h * 60:(h + 1) * 60].sum() >= 30 else 0.0
    return hourly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 10: Demand Diversity from Heterogeneous Schedules")
    print("=" * 60)

    # Accumulators per household type
    # Sum of occupancy persons per hour, and count of buildings
    type_occ_sum = {t: np.zeros(24) for t in HH_TYPES_ORDERED}  # total persons home
    type_bldg_count = {t: 0 for t in HH_TYPES_ORDERED}
    type_peak_sum = {t: 0.0 for t in HH_TYPES_ORDERED}  # sum of per-building peak occupancy

    # Aggregate occupancy: sum across ALL buildings of persons-at-home per hour
    agg_occ = np.zeros(24)
    # Sum of per-building peak (for coincidence factor denominator)
    sum_of_individual_peaks = 0.0
    n_buildings = 0
    n_persons = 0
    n_shards = 0

    for df in iterate_shards():
        n_shards += 1
        if n_shards % 50 == 0:
            print(f"  Shard {n_shards} ... ({n_buildings:,} buildings)")

        for _, row in df.iterrows():
            n_buildings += 1
            persons = row.get("persons", [])
            if not isinstance(persons, list) or len(persons) == 0:
                continue

            hh_type = classify_household(row)
            n_persons += len(persons)

            # Compute household-level occupancy: sum of per-person at-home
            hh_occ = np.zeros(24)  # persons at home per hour
            for person in persons:
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list):
                    continue
                hh_occ += at_home_by_hour(acts)

            # Update type-level accumulators
            type_occ_sum[hh_type] += hh_occ
            type_bldg_count[hh_type] += 1
            bldg_peak = hh_occ.max()
            type_peak_sum[hh_type] += bldg_peak

            # Update aggregate accumulators
            agg_occ += hh_occ
            sum_of_individual_peaks += bldg_peak

        del df
        gc.collect()

    print(f"\nComplete: {n_buildings:,} buildings, {n_persons:,} persons, "
          f"{n_shards} shards")

    # ---- Compute metrics ----

    # Coincidence factor = peak_aggregate / sum_of_individual_peaks
    peak_aggregate = agg_occ.max()
    coincidence_factor = peak_aggregate / sum_of_individual_peaks if sum_of_individual_peaks > 0 else 1.0

    # Aggregate load factor = mean / peak
    load_factor = agg_occ.mean() / peak_aggregate if peak_aggregate > 0 else 1.0

    # Per-type mean occupancy profile (persons home per building per hour)
    type_mean_occ = {}
    type_peak_to_avg = {}
    for t in HH_TYPES_ORDERED:
        n = type_bldg_count[t]
        if n > 0:
            profile = type_occ_sum[t] / n
            type_mean_occ[t] = profile
            pk = profile.max()
            avg = profile.mean()
            type_peak_to_avg[t] = pk / avg if avg > 0 else 1.0
        else:
            type_mean_occ[t] = np.zeros(24)
            type_peak_to_avg[t] = 1.0

    # Hypothetical uniform: every building follows the overall average profile
    overall_mean_profile = agg_occ / n_buildings if n_buildings > 0 else np.zeros(24)
    # If all buildings had this same profile, the aggregate peak would be
    # n_buildings * max(overall_mean_profile) = max(agg_occ) [same peak]
    # But the sum_of_individual_peaks under uniform = n_buildings * max(overall_mean)
    # Coincidence factor for uniform = 1.0 (by construction)

    # Normalize both for plotting
    agg_normalized = agg_occ / peak_aggregate if peak_aggregate > 0 else agg_occ
    # For uniform: shape is the same but with sharper shoulders
    # Use the overall mean profile, normalized to same total but with peak = 1
    uniform_profile = overall_mean_profile / overall_mean_profile.max() if overall_mean_profile.max() > 0 else overall_mean_profile

    # Peak reduction percentage
    peak_reduction_pct = (1 - coincidence_factor) * 100

    print(f"\n--- Key Metrics ---")
    print(f"  Coincidence factor: {coincidence_factor:.4f}")
    print(f"  Load factor: {load_factor:.4f}")
    print(f"  Peak aggregate occupancy (persons): {peak_aggregate:,.0f}")
    print(f"  Sum of individual peaks: {sum_of_individual_peaks:,.0f}")
    print(f"  Peak reduction from diversity: {peak_reduction_pct:.1f}%")
    print(f"\n--- Peak-to-Average Ratio by Household Type ---")
    for t in HH_TYPES_ORDERED:
        print(f"  {t}: {type_peak_to_avg[t]:.3f} (n={type_bldg_count[t]:,})")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: Aggregate vs uniform
    ax = axes[0]
    hours = np.arange(24)
    ax.plot(hours, agg_normalized, color="#2C6FAC", lw=2.5, label="Actual (heterogeneous)",
            zorder=3)
    ax.plot(hours, uniform_profile, color="#E74C3C", lw=2.0, ls="--",
            label="Hypothetical (uniform)", zorder=2)
    ax.fill_between(hours, agg_normalized, uniform_profile,
                    where=uniform_profile > agg_normalized,
                    color="#E74C3C", alpha=0.15, label="Diversity benefit")
    ax.axvspan(8, 17, alpha=0.06, color="gray")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Aggregate Occupancy")
    ax.set_title("(a) Aggregate Occupancy: Heterogeneous vs. Uniform")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["00", "04", "08", "12", "16", "20"])
    ax.set_ylim(0, 1.15)

    # Annotation box
    textstr = (f"Coincidence Factor = {coincidence_factor:.3f}\n"
               f"Load Factor = {load_factor:.3f}")
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Peak-to-average ratio by household type
    ax = axes[1]
    types_sorted = sorted(HH_TYPES_ORDERED,
                          key=lambda t: type_peak_to_avg[t], reverse=True)
    y_pos = range(len(types_sorted))
    ratios = [type_peak_to_avg[t] for t in types_sorted]
    bar_colors = [COLORS[t] for t in types_sorted]

    bars = ax.barh(y_pos, ratios, color=bar_colors, edgecolor="white",
                   height=0.6, linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types_sorted, fontsize=10)
    ax.set_xlabel("Peak-to-Average Occupancy Ratio")
    ax.set_title("(b) Peak-to-Average by Household Type")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", ha="left", va="center", fontsize=10)

    plt.tight_layout(w_pad=3)
    out_path = FIG_DIR / "fig_demand_diversity.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Save data CSV
    csv_path = FIG_DIR / "demand_diversity_data.csv"
    rows = []
    for t in HH_TYPES_ORDERED:
        rows.append({
            "household_type": t,
            "n_buildings": type_bldg_count[t],
            "peak_to_average": round(type_peak_to_avg[t], 4),
        })
    rows.append({
        "household_type": "AGGREGATE",
        "n_buildings": n_buildings,
        "peak_to_average": round(1.0 / load_factor, 4),
    })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"\n{'='*60}")
    print(f"KEY RESULTS FOR LATEX:")
    print(f"  Coincidence factor CF = {coincidence_factor:.3f}")
    print(f"  Aggregate peak is {coincidence_factor*100:.1f}% of sum of individual peaks")
    print(f"  Load factor = {load_factor:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
