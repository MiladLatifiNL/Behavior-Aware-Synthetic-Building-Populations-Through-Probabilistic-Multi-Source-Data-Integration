"""
Script 09: Weather-Behavior Coupling Analysis
==============================================

Demonstrates that the unified dataset captures the interaction between
outdoor temperature and human activity patterns — an analysis impossible
with any single source survey. NSRDB has weather but no behavior; ATUS
has behavior but no weather. Only the unified dataset links them at the
individual level.

Outputs (3-panel figure):
  (a) Mean outdoor activity minutes per person vs temperature bins
  (b) Mean HVAC demand indicator by hour, stratified by household type
  (c) Share of person-day spent outdoors by daily temperature quintile

Output: revision/figures/fig_weather_behavior_coupling.pdf
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
# Publication-quality rc settings (matching other revision scripts)
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

# Temperature bins for Panel A (5-degree Celsius intervals)
TEMP_BIN_EDGES = list(range(-15, 45, 5))  # [-15, -10, ..., 40]
TEMP_BIN_CENTERS = [(TEMP_BIN_EDGES[i] + TEMP_BIN_EDGES[i + 1]) / 2
                    for i in range(len(TEMP_BIN_EDGES) - 1)]

# Household type definitions (same as script 04)
HH_TYPES_ORDERED = [
    "Dual-earner + children",
    "Single parent",
    "Single young professional",
    "Retired couple",
    "Single elderly",
    "Other",
]

STYLE_MAP = {
    "Dual-earner + children":    {"color": "#E74C3C", "ls": "-",  "lw": 2.2},
    "Single parent":             {"color": "#F39C12", "ls": "--", "lw": 2.0},
    "Single young professional": {"color": "#3498DB", "ls": "-.", "lw": 2.0},
    "Retired couple":            {"color": "#27AE60", "ls": "-",  "lw": 2.2},
    "Single elderly":            {"color": "#8E44AD", "ls": "--", "lw": 2.0},
    "Other":                     {"color": "#95A5A6", "ls": ":",  "lw": 1.6},
}

# Location codes that indicate "at home"
HOME_LOCATIONS = {"1", "-1"}


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
# Temperature bin helper
# ---------------------------------------------------------------------------
def temp_to_bin_index(temp_c):
    """Return the bin index for a temperature value, or -1 if out of range."""
    for i in range(len(TEMP_BIN_EDGES) - 1):
        if TEMP_BIN_EDGES[i] <= temp_c < TEMP_BIN_EDGES[i + 1]:
            return i
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 09: Weather-Behavior Coupling Analysis")
    print("=" * 60)

    n_bins = len(TEMP_BIN_CENTERS)

    # ----- Panel A accumulators: outdoor minutes by temperature bin -----
    outdoor_min_sum = np.zeros(n_bins)
    outdoor_min_sq_sum = np.zeros(n_bins)  # for SE calculation
    person_count_per_bin = np.zeros(n_bins, dtype=np.int64)

    # ----- Panel B accumulators: HVAC load by household type and hour -----
    hvac_load_sum = {t: np.zeros(24) for t in HH_TYPES_ORDERED}
    hvac_load_count = {t: np.zeros(24, dtype=np.int64) for t in HH_TYPES_ORDERED}

    # ----- Panel C: collect building-level temperature for quintile edges -----
    building_temps = []

    # ----- Panel C accumulators (second pass): outdoor/total minutes by quintile -----
    # Will be populated in second pass

    # ---- First pass: Panels A, B, and collect building temps for C ----
    n_buildings = 0
    n_persons = 0
    n_shards = 0

    print("\n--- Pass 1: Panels A & B + collecting building temperatures ---")
    for df in iterate_shards():
        n_shards += 1
        if n_shards % 50 == 0:
            print(f"  Shard {n_shards} ... ({n_buildings:,} buildings, "
                  f"{n_persons:,} persons)")

        for _, row in df.iterrows():
            n_buildings += 1
            persons = row.get("persons", [])
            if not isinstance(persons, list) or len(persons) == 0:
                continue

            hh_type = classify_household(row)

            # Collect building-level temp for Panel C quintiles
            ws = row.get("weather_summary", {})
            if isinstance(ws, dict):
                bldg_temp = ws.get("temp_mean", None)
                if bldg_temp is not None:
                    try:
                        building_temps.append(float(bldg_temp))
                    except (ValueError, TypeError):
                        pass

            for person in persons:
                n_persons += 1
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list):
                    continue

                # Per-person outdoor minutes accumulator for Panel A
                person_outdoor_by_bin = np.zeros(n_bins)

                for act in acts:
                    weather = act.get("weather", {})
                    if not isinstance(weather, dict):
                        continue

                    duration = 0
                    try:
                        duration = int(act.get("duration_minutes", 0))
                    except (ValueError, TypeError):
                        continue
                    if duration <= 0:
                        continue

                    temp = weather.get("temp_mean", None)
                    hvac = weather.get("hvac_load", None)
                    exposure = str(weather.get("exposure_type", ""))
                    location = str(act.get("location", ""))

                    is_outdoor = (exposure == "outdoor" or
                                  (location not in HOME_LOCATIONS and
                                   location not in ("", "None", "nan")))

                    # Panel A: outdoor minutes by temperature bin
                    if is_outdoor and temp is not None:
                        try:
                            bi = temp_to_bin_index(float(temp))
                            if bi >= 0:
                                person_outdoor_by_bin[bi] += duration
                        except (ValueError, TypeError):
                            pass

                    # Panel B: HVAC load by hour and household type
                    if hvac is not None:
                        try:
                            hvac_val = float(hvac)
                            start_str = str(act.get("start_time", "00:00:00"))
                            sp = start_str.split(":")
                            start_hour = int(sp[0])
                            if 0 <= start_hour < 24:
                                hvac_load_sum[hh_type][start_hour] += hvac_val * duration
                                hvac_load_count[hh_type][start_hour] += duration
                        except (ValueError, TypeError, IndexError):
                            pass

                # Accumulate per-person outdoor minutes into Panel A bins
                for bi in range(n_bins):
                    if person_outdoor_by_bin[bi] > 0:
                        outdoor_min_sum[bi] += person_outdoor_by_bin[bi]
                        outdoor_min_sq_sum[bi] += person_outdoor_by_bin[bi] ** 2
                        person_count_per_bin[bi] += 1

        del df
        gc.collect()

    print(f"\nPass 1 complete: {n_buildings:,} buildings, {n_persons:,} persons, "
          f"{n_shards} shards")

    # Compute Panel C quintile edges
    building_temps_arr = np.array(building_temps)
    if len(building_temps_arr) > 0:
        quintile_edges = np.percentile(building_temps_arr, [0, 20, 40, 60, 80, 100])
    else:
        quintile_edges = np.array([-10, 0, 10, 20, 30, 40])
    quintile_labels = [
        f"Q1\n({quintile_edges[0]:.0f}–{quintile_edges[1]:.0f}°C)",
        f"Q2\n({quintile_edges[1]:.0f}–{quintile_edges[2]:.0f}°C)",
        f"Q3\n({quintile_edges[2]:.0f}–{quintile_edges[3]:.0f}°C)",
        f"Q4\n({quintile_edges[3]:.0f}–{quintile_edges[4]:.0f}°C)",
        f"Q5\n({quintile_edges[4]:.0f}–{quintile_edges[5]:.0f}°C)",
    ]
    print(f"Temperature quintile edges: {quintile_edges}")

    # ---- Second pass: Panel C (outdoor vs indoor by temperature quintile) ----
    outdoor_by_quintile = np.zeros(5)
    total_by_quintile = np.zeros(5)
    person_count_by_quintile = np.zeros(5, dtype=np.int64)

    print("\n--- Pass 2: Panel C (outdoor/indoor by temperature quintile) ---")
    n_shards2 = 0
    for df in iterate_shards():
        n_shards2 += 1
        if n_shards2 % 50 == 0:
            print(f"  Shard {n_shards2} ...")

        for _, row in df.iterrows():
            ws = row.get("weather_summary", {})
            if not isinstance(ws, dict):
                continue
            bldg_temp = ws.get("temp_mean", None)
            if bldg_temp is None:
                continue
            try:
                bldg_temp = float(bldg_temp)
            except (ValueError, TypeError):
                continue

            # Determine quintile
            q_idx = -1
            for qi in range(5):
                if quintile_edges[qi] <= bldg_temp < quintile_edges[qi + 1]:
                    q_idx = qi
                    break
            if q_idx == -1:
                if bldg_temp >= quintile_edges[5]:
                    q_idx = 4
                else:
                    continue

            persons = row.get("persons", [])
            if not isinstance(persons, list):
                continue

            for person in persons:
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list):
                    continue

                person_outdoor = 0.0
                person_total = 0.0

                for act in acts:
                    try:
                        duration = int(act.get("duration_minutes", 0))
                    except (ValueError, TypeError):
                        continue
                    if duration <= 0:
                        continue
                    person_total += duration

                    weather = act.get("weather", {})
                    if not isinstance(weather, dict):
                        continue
                    exposure = str(weather.get("exposure_type", ""))
                    location = str(act.get("location", ""))
                    is_outdoor = (exposure == "outdoor" or
                                  (location not in HOME_LOCATIONS and
                                   location not in ("", "None", "nan")))
                    if is_outdoor:
                        person_outdoor += duration

                if person_total > 0:
                    outdoor_by_quintile[q_idx] += person_outdoor
                    total_by_quintile[q_idx] += person_total
                    person_count_by_quintile[q_idx] += 1

        del df
        gc.collect()

    print(f"Pass 2 complete.")

    # ---- Compute statistics ----

    # Panel A: mean outdoor minutes per person by temp bin
    mean_outdoor = np.zeros(n_bins)
    se_outdoor = np.zeros(n_bins)
    for i in range(n_bins):
        n = person_count_per_bin[i]
        if n > 0:
            mean_outdoor[i] = outdoor_min_sum[i] / n
            if n > 1:
                var = (outdoor_min_sq_sum[i] / n - mean_outdoor[i] ** 2)
                se_outdoor[i] = np.sqrt(max(0, var) / n)

    # Panel B: mean HVAC load per hour by household type
    hvac_mean = {}
    for t in HH_TYPES_ORDERED:
        hvac_mean[t] = np.zeros(24)
        for h in range(24):
            if hvac_load_count[t][h] > 0:
                hvac_mean[t][h] = hvac_load_sum[t][h] / hvac_load_count[t][h]

    # Panel C: outdoor fraction by quintile
    outdoor_fraction = np.zeros(5)
    for qi in range(5):
        if total_by_quintile[qi] > 0:
            outdoor_fraction[qi] = outdoor_by_quintile[qi] / total_by_quintile[qi] * 100

    # ---- Print summary ----
    print("\n--- Panel A: Outdoor minutes by temperature bin ---")
    for i, c in enumerate(TEMP_BIN_CENTERS):
        print(f"  {c:+6.1f}°C: {mean_outdoor[i]:6.1f} min/person "
              f"(n={person_count_per_bin[i]:,})")

    print("\n--- Panel C: Outdoor fraction by temperature quintile ---")
    for qi in range(5):
        print(f"  Q{qi + 1}: {outdoor_fraction[qi]:.1f}% outdoor "
              f"(n={person_count_by_quintile[qi]:,})")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Outdoor duration vs temperature
    ax = axes[0]
    valid = person_count_per_bin > 0
    ax.plot(np.array(TEMP_BIN_CENTERS)[valid], mean_outdoor[valid],
            color="#2C6FAC", lw=2.2, marker="o", markersize=5, zorder=3)
    ax.fill_between(np.array(TEMP_BIN_CENTERS)[valid],
                    (mean_outdoor - se_outdoor)[valid],
                    (mean_outdoor + se_outdoor)[valid],
                    color="#2C6FAC", alpha=0.2, zorder=2)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(30, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.text(0.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
            "0°C", ha="center", fontsize=9, color="gray")
    ax.text(30.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
            "30°C", ha="center", fontsize=9, color="gray")
    ax.set_xlabel("Outdoor Temperature (°C)")
    ax.set_ylabel("Mean Outdoor Activity (min/person)")
    ax.set_title("(a) Outdoor Duration vs. Temperature")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: HVAC demand by household type and hour
    ax = axes[1]
    hours = np.arange(24)
    for hh_type in HH_TYPES_ORDERED:
        style = STYLE_MAP[hh_type]
        ax.plot(hours, hvac_mean[hh_type],
                color=style["color"], ls=style["ls"], lw=style["lw"],
                label=hh_type)
    ax.axvspan(8, 17, alpha=0.08, color="gray", label="_nolegend_")
    ax.text(12.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
            "Work hours", ha="center", fontsize=9, color="gray", fontstyle="italic")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean HVAC Demand Indicator")
    ax.set_title("(b) HVAC Demand by Household Type")
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["00", "04", "08", "12", "16", "20"])
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Outdoor fraction by temperature quintile
    ax = axes[2]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 5))
    bars = ax.bar(range(5), outdoor_fraction, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(quintile_labels, fontsize=9)
    ax.set_ylabel("Time Spent Outdoors (%)")
    ax.set_title("(c) Outdoor Time by Temperature Quintile")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, outdoor_fraction):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(w_pad=3)
    out_path = FIG_DIR / "fig_weather_behavior_coupling.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Save summary CSV
    csv_path = FIG_DIR / "weather_behavior_coupling_data.csv"
    rows = []
    for i, c in enumerate(TEMP_BIN_CENTERS):
        rows.append({
            "temp_bin_center": c,
            "mean_outdoor_min": round(mean_outdoor[i], 2),
            "se_outdoor_min": round(se_outdoor[i], 4),
            "n_persons": int(person_count_per_bin[i]),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
