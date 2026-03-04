"""
Script 04: Hourly At-Home Occupancy Profiles by Household Type
==============================================================

Generates a publication-quality figure showing hourly at-home probability
curves stratified by household type. This analysis combines ATUS activity
patterns (from persons) with PUMS household composition -- an analysis
impossible with any single dataset alone.

Output: revision/figures/fig_occupancy_by_household_type.pdf
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
# Constants
# ---------------------------------------------------------------------------
MANIFEST = Path("D:/PUMS_Enrichment/data/processed/phase4_shards/manifest.json")
FIG_DIR = Path("D:/PUMS_Enrichment/revision/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Location codes that indicate "at home"
# Location "1" = respondent's home; Location "-1" = sleeping/personal care
# (activity codes 111=sleep, 112=sleeplessness, 121=grooming -- all at home)
HOME_LOCATIONS = {"1", "-1"}

# Fallback: activity codes that are definitively at-home
# (used only when location field is missing)
HOME_ACTIVITY_PREFIXES = ("01", "02", "05", "11")

HOURS = list(range(24))

# Household type definitions (checked in order; first match wins)
HH_TYPES_ORDERED = [
    "Dual-earner + children",
    "Single parent",
    "Single young professional",
    "Retired couple",
    "Single elderly",
    "Other",
]

# Plotting styles per household type
STYLE_MAP = {
    "Dual-earner + children":    {"color": "#E74C3C", "ls": "-",  "marker": "o", "lw": 2.2},
    "Single parent":             {"color": "#F39C12", "ls": "--", "marker": "s", "lw": 2.0},
    "Single young professional": {"color": "#3498DB", "ls": "-.", "marker": "^", "lw": 2.0},
    "Retired couple":            {"color": "#27AE60", "ls": "-",  "marker": "D", "lw": 2.2},
    "Single elderly":            {"color": "#8E44AD", "ls": "--", "marker": "v", "lw": 2.0},
    "Other":                     {"color": "#95A5A6", "ls": ":",  "marker": "x", "lw": 1.6},
}


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
            with open(p, "rb") as fh:
                df = pickle.load(fh)
            if isinstance(df, pd.DataFrame):
                yield df
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Household type classifier
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
# Determine at-home status for each hour from an activity sequence
# ---------------------------------------------------------------------------
def at_home_by_hour(activity_sequence):
    """
    Given a list of activity dicts, return a 24-element boolean array where
    True means the person is at home during that hour.

    Each activity has start_time (HH:MM:SS), stop_time (HH:MM:SS),
    duration_minutes, location, and activity_code.

    Strategy: for each hour h, check if the person has an at-home activity
    that overlaps with [h:00, h+1:00). We compute per-minute home status
    for efficiency using a 1440-minute array.
    """
    # 1440-minute array; default to False (away)
    home_minutes = np.zeros(1440, dtype=bool)

    for act in activity_sequence:
        # Determine if this activity is at-home
        loc = str(act.get("location", ""))
        code = str(act.get("activity_code", ""))

        is_home = loc in HOME_LOCATIONS
        if not is_home and loc in ("", "None", "nan"):
            # Fallback: check activity code prefix
            code_padded = code.zfill(4) if len(code) <= 4 else code.zfill(6)
            code_2 = code_padded[:2]
            is_home = code_2 in ("01", "02", "11")

        if not is_home:
            continue

        # Parse start/stop times
        try:
            start_str = str(act.get("start_time", "00:00:00"))
            stop_str = str(act.get("stop_time", "00:00:00"))
            sp = start_str.split(":")
            ep = stop_str.split(":")
            start_min = int(sp[0]) * 60 + int(sp[1])
            stop_min = int(ep[0]) * 60 + int(ep[1])
        except (ValueError, IndexError):
            continue

        # ATUS diaries start at 04:00 and wrap around.
        # Represent in a 1440-minute day (00:00 = minute 0).
        if stop_min <= start_min:
            # Activity crosses midnight: set [start, 1440) and [0, stop)
            home_minutes[start_min:1440] = True
            home_minutes[0:stop_min] = True
        else:
            home_minutes[start_min:stop_min] = True

    # Aggregate to hourly: person is "at home" for hour h if they are home
    # for at least 30 of the 60 minutes in that hour.
    hourly_home = np.zeros(24, dtype=bool)
    for h in range(24):
        minutes_home = home_minutes[h * 60 : (h + 1) * 60].sum()
        hourly_home[h] = minutes_home >= 30

    return hourly_home


# ---------------------------------------------------------------------------
# Main accumulation
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 04: Hourly At-Home Occupancy by Household Type")
    print("=" * 60)

    # Accumulators: home_counts[type][hour], total_counts[type][hour]
    home_counts = {t: np.zeros(24, dtype=np.int64) for t in HH_TYPES_ORDERED}
    total_counts = {t: np.zeros(24, dtype=np.int64) for t in HH_TYPES_ORDERED}

    n_buildings = 0
    n_persons = 0
    n_shards = 0

    for df in iterate_shards():
        n_shards += 1
        if n_shards % 50 == 0:
            print(f"  Processing shard {n_shards} ... "
                  f"({n_buildings:,} buildings, {n_persons:,} persons so far)")

        for _, row in df.iterrows():
            persons = row.get("persons", [])
            if not isinstance(persons, list) or len(persons) == 0:
                continue

            hh_type = classify_household(row)
            n_buildings += 1

            for person in persons:
                # ATUS only surveys persons aged 15+; younger children have
                # imputed schedules that do not reflect school attendance and
                # produce spurious midday at-home spikes.  Exclude them.
                age = 0
                try:
                    age = int(float(person.get("AGEP", 0)))
                except (ValueError, TypeError):
                    pass
                if age < 15:
                    continue

                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list) or len(acts) == 0:
                    continue

                n_persons += 1
                hourly_home = at_home_by_hour(acts)

                home_counts[hh_type] += hourly_home.astype(np.int64)
                total_counts[hh_type] += np.ones(24, dtype=np.int64)

        del df
        gc.collect()

    print(f"\nDone streaming. {n_shards} shards, "
          f"{n_buildings:,} buildings, {n_persons:,} persons.")

    # ------------------------------------------------------------------
    # Compute fractions
    # ------------------------------------------------------------------
    fractions = {}
    for t in HH_TYPES_ORDERED:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total_counts[t] > 0,
                            home_counts[t] / total_counts[t],
                            np.nan)
        fractions[t] = frac
        total_obs = total_counts[t].sum()
        if total_obs > 0:
            print(f"  {t:32s}: {total_obs // 24:>10,} persons")
        else:
            print(f"  {t:32s}: (no data)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nGenerating figure ...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded work-hours band
    ax.axvspan(8, 17, alpha=0.08, color="#2C3E50", label="_nolegend_")
    ax.text(12.5, 0.97, "Typical work hours", ha="center", va="top",
            fontsize=9, color="#555555", style="italic",
            transform=ax.get_xaxis_transform())

    for hh_type in HH_TYPES_ORDERED:
        frac = fractions[hh_type]
        if np.all(np.isnan(frac)):
            continue
        st = STYLE_MAP[hh_type]
        ax.plot(HOURS, frac,
                color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["marker"], markersize=5, markeredgewidth=0.8,
                markeredgecolor="white", label=hh_type)

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Fraction of Persons at Home", fontsize=12)
    ax.set_title("Hourly At-Home Occupancy by Household Type", fontsize=14,
                 fontweight="bold", pad=12)

    ax.set_xlim(-0.3, 23.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                       fontsize=10, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="y", labelsize=10)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", ncol=2)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_occupancy_by_household_type.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save a summary CSV for reference
    summary = pd.DataFrame(fractions, index=[f"{h:02d}:00" for h in HOURS])
    csv_path = FIG_DIR / "occupancy_data.csv"
    summary.to_csv(csv_path, float_format="%.4f")
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
