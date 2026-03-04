"""
Script 11: "A Day in the Life" — Household Activity Timelines
=============================================================

Creates a centerpiece visualization showing the unified dataset as a
living system. Selects four representative households from different
climate zones and renders 24-hour activity timelines for every person,
with concurrent outdoor temperature overlaid.

This figure makes the "alive creature" concept tangible: each building
has real people with individualized, minute-level, household-coordinated
schedules unfolding under actual weather conditions.

Output: revision/figures/fig_day_in_life.pdf
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
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Publication-quality rc settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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

# Activity code prefix → category
PREFIX_TO_CATEGORY = {
    "01": "Sleep/Pers. Care",
    "02": "Household",
    "03": "Household",       # Caring for household members
    "04": "Household",       # Caring for non-household members
    "05": "Work",
    "06": "Work",            # Education → grouped with Work
    "07": "Household",       # Consumer purchases / shopping
    "08": "Household",       # Professional services
    "09": "Household",       # Household services
    "10": "Leisure",         # Civic obligations
    "11": "Eating",
    "12": "Leisure",         # Socializing, relaxing
    "13": "Leisure",         # Sports, exercise
    "14": "Leisure",         # Religious activities
    "15": "Leisure",         # Volunteer activities
    "16": "Leisure",         # Telephone calls
    "17": "Leisure",         # Other socializing
    "18": "Travel",
    "19": "Other",
    "50": "Other",
}

ACTIVITY_COLORS = {
    "Sleep/Pers. Care": "#2C3E50",   # dark navy
    "Work":             "#E67E22",   # orange
    "Household":        "#27AE60",   # green
    "Eating":           "#F1C40F",   # yellow-gold
    "Leisure":          "#3498DB",   # blue
    "Travel":           "#7F8C8D",   # gray
    "Other":            "#BDC3C7",   # light gray
}

# FIPS code → state name (for labels)
FIPS_TO_STATE = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "DC", "12": "Florida", "13": "Georgia",
    "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana",
    "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
    "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
    "35": "New Mexico", "36": "New York", "37": "North Carolina",
    "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

# Target climate zones for geographic diversity
# We scan specific shard ranges to find diverse climate states
# Shard 0 = Alabama (hot-humid), Shard ~300 = Minnesota (very cold), etc.
DESIRED_HOUSEHOLDS = [
    {
        "label": "A",
        "name": "Dual-Earner Family",
        "criteria": lambda r: (
            bool(r.get("has_children", False)) and
            int(r.get("person_employed_count", 0)) >= 2 and
            int(r.get("household_size", r.get("NP", 1))) >= 4
        ),
        "preferred_climate": "hot_humid",
        "shard_hint": list(range(0, 50)),  # Southern states
    },
    {
        "label": "B",
        "name": "Retired Couple",
        "criteria": lambda r: (
            int(r.get("household_size", r.get("NP", 1))) == 2 and
            bool(r.get("has_seniors", False)) and
            float(r.get("employment_rate", 0.0)) == 0 and
            str(r.get("climate_zone", "")).lower().replace(" ", "_") in
            ("very_cold", "cold") and
            str(r.get("STATE", "")) != "02"  # Exclude Alaska (no weather data)
        ),
        "preferred_climate": "very_cold",
        "shard_hint": list(range(340, 420)),  # Maine(23)~shard350, Michigan(26)~shard400
    },
    {
        "label": "C",
        "name": "Single Parent",
        "criteria": lambda r: (
            bool(r.get("has_children", False)) and
            int(r.get("person_employed_count", 0)) <= 1 and
            int(r.get("household_size", r.get("NP", 1))) >= 2 and
            int(r.get("household_size", r.get("NP", 1))) <= 4
        ),
        "preferred_climate": "mixed_humid",
        "shard_hint": list(range(400, 500)),  # Mid-Atlantic
    },
    {
        "label": "D",
        "name": "Single Young Professional",
        "criteria": lambda r: (
            int(r.get("household_size", r.get("NP", 1))) == 1 and
            float(r.get("person_age_mean", 40)) < 35 and
            float(r.get("employment_rate", 0.0)) > 0
        ),
        "preferred_climate": "marine",
        "shard_hint": list(range(700, 800)),  # West Coast
    },
]


# ---------------------------------------------------------------------------
# Helper: code → category
# ---------------------------------------------------------------------------
def code_to_category(code):
    """Map ATUS activity code to display category."""
    code_str = str(code).strip()
    # ATUS codes can be 3-4 digits (XXYY) or 5-6 digits (XXYYYY)
    if len(code_str) <= 4:
        code_str = code_str.zfill(4)
    else:
        code_str = code_str.zfill(6)
    prefix = code_str[:2]
    return PREFIX_TO_CATEGORY.get(prefix, "Other")


# ---------------------------------------------------------------------------
# Helper: find representative households
# ---------------------------------------------------------------------------
def find_households():
    """Scan shards to find one representative household of each type."""
    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest_dir = MANIFEST.parent
    shard_files = manifest["files"]

    found = {}
    needed = set(range(len(DESIRED_HOUSEHOLDS)))

    for idx in needed.copy():
        spec = DESIRED_HOUSEHOLDS[idx]
        shard_indices = spec["shard_hint"]

        for si in shard_indices:
            if si >= len(shard_files):
                continue
            fp = shard_files[si]
            p = Path(fp)
            if not p.exists():
                p = manifest_dir / p.name
            if not p.exists():
                continue

            try:
                with open(p, "rb") as fh:
                    df = pickle.load(fh)
            except Exception:
                continue

            if not isinstance(df, pd.DataFrame):
                continue

            for _, row in df.iterrows():
                persons = row.get("persons", [])
                if not isinstance(persons, list) or len(persons) == 0:
                    continue

                # Check all persons have enough activities
                min_acts = min(len(p.get("activity_sequence", [])) for p in persons
                               if isinstance(p.get("activity_sequence", []), list))
                if min_acts < 10:
                    continue

                if spec["criteria"](row):
                    # Verify weather data exists for this household
                    first_acts = persons[0].get("activity_sequence", [])
                    if isinstance(first_acts, list) and len(first_acts) > 0:
                        w = first_acts[0].get("weather")
                        if not isinstance(w, dict) or w.get("temp_mean") is None:
                            continue  # Skip households without weather data
                    found[idx] = row.to_dict()
                    print(f"  Found HH {spec['label']} ({spec['name']}): "
                          f"STATE={row.get('STATE', '?')}, "
                          f"size={len(persons)}, "
                          f"shard={si}")
                    break

            del df
            gc.collect()

            if idx in found:
                break

    # Fallback: scan first 100 shards for any unfound types
    for idx in set(range(len(DESIRED_HOUSEHOLDS))) - set(found.keys()):
        spec = DESIRED_HOUSEHOLDS[idx]
        print(f"  Fallback search for HH {spec['label']} ({spec['name']})...")

        for si, fp in enumerate(shard_files[:100]):
            p = Path(fp)
            if not p.exists():
                p = manifest_dir / p.name
            if not p.exists():
                continue

            try:
                with open(p, "rb") as fh:
                    df = pickle.load(fh)
            except Exception:
                continue

            if not isinstance(df, pd.DataFrame):
                continue

            for _, row in df.iterrows():
                persons = row.get("persons", [])
                if not isinstance(persons, list) or len(persons) == 0:
                    continue
                min_acts = min(len(p.get("activity_sequence", [])) for p in persons
                               if isinstance(p.get("activity_sequence", []), list))
                if min_acts < 10:
                    continue
                if spec["criteria"](row):
                    found[idx] = row.to_dict()
                    print(f"  Fallback found HH {spec['label']}: "
                          f"STATE={row.get('STATE', '?')}, shard={si}")
                    break

            del df
            gc.collect()
            if idx in found:
                break

    return found


# ---------------------------------------------------------------------------
# Helper: extract hourly temperature from activities
# ---------------------------------------------------------------------------
def extract_hourly_temp(persons):
    """Extract an hourly temperature profile from activity-level weather."""
    temp_sum = np.zeros(24)
    temp_count = np.zeros(24, dtype=int)

    for person in persons:
        acts = person.get("activity_sequence", [])
        if not isinstance(acts, list):
            continue
        for act in acts:
            weather = act.get("weather", {})
            if not isinstance(weather, dict):
                continue
            temp = weather.get("temp_mean", None)
            if temp is None:
                continue
            try:
                temp = float(temp)
                start_str = str(act.get("start_time", "00:00:00"))
                sp = start_str.split(":")
                hour = int(sp[0])
                if 0 <= hour < 24:
                    temp_sum[hour] += temp
                    temp_count[hour] += 1
            except (ValueError, TypeError, IndexError):
                continue

    hourly_temp = np.full(24, np.nan)
    for h in range(24):
        if temp_count[h] > 0:
            hourly_temp[h] = temp_sum[h] / temp_count[h]

    # Interpolate NaNs
    valid = ~np.isnan(hourly_temp)
    if valid.sum() >= 2:
        xp = np.where(valid)[0]
        fp = hourly_temp[valid]
        hourly_temp = np.interp(np.arange(24), xp, fp)

    return hourly_temp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 11: 'A Day in the Life' Visualization")
    print("=" * 60)

    print("\n--- Searching for representative households ---")
    found = find_households()

    if len(found) < 4:
        print(f"\nWARNING: Found only {len(found)} of 4 household types.")
        if len(found) == 0:
            print("Cannot generate figure. Exiting.")
            return

    n_hh = len(found)

    # Debug: print activity code distribution for each found household
    for idx in sorted(found.keys()):
        spec = DESIRED_HOUSEHOLDS[idx]
        row_data = found[idx]
        persons = row_data.get("persons", [])
        if not isinstance(persons, list):
            continue
        cat_counts = {}
        for person in persons:
            acts = person.get("activity_sequence", [])
            if not isinstance(acts, list):
                continue
            for act in acts:
                code = str(act.get("activity_code", ""))
                cat = code_to_category(code)
                dur = 0
                try:
                    dur = int(act.get("duration_minutes", 0))
                except (ValueError, TypeError):
                    pass
                cat_counts[cat] = cat_counts.get(cat, 0) + dur
        print(f"  HH {spec['label']} activity minutes: {cat_counts}")

    # Determine figure height based on number of households and persons
    total_rows = 0
    hh_data = []
    for idx in sorted(found.keys()):
        spec = DESIRED_HOUSEHOLDS[idx]
        row_data = found[idx]
        persons = row_data.get("persons", [])
        if not isinstance(persons, list):
            persons = []

        state_code = str(row_data.get("STATE", "??")).zfill(2)
        state_name = FIPS_TO_STATE.get(state_code, f"State {state_code}")
        climate = str(row_data.get("climate_zone", "unknown")).replace("_", " ").title()
        income = row_data.get("HINCP", 0)
        try:
            income = int(float(income))
        except (ValueError, TypeError):
            income = 0

        hh_data.append({
            "idx": idx,
            "spec": spec,
            "persons": persons,
            "n_persons": len(persons),
            "state_name": state_name,
            "climate": climate,
            "income": income,
            "hourly_temp": extract_hourly_temp(persons),
        })
        total_rows += len(persons)

    fig_height = max(8, 2.2 * total_rows + 1.5)
    fig, axes = plt.subplots(n_hh, 2, figsize=(16, fig_height),
                             gridspec_kw={"width_ratios": [4, 1],
                                          "hspace": 0.35, "wspace": 0.08})
    if n_hh == 1:
        axes = axes.reshape(1, -1)

    for row_idx, hd in enumerate(hh_data):
        spec = hd["spec"]
        persons = hd["persons"]
        n_persons = hd["n_persons"]

        # --- Left panel: Activity timelines ---
        ax_act = axes[row_idx, 0]

        for p_idx, person in enumerate(persons):
            acts = person.get("activity_sequence", [])
            if not isinstance(acts, list):
                continue

            age = person.get("AGEP", "?")
            sex = "F" if str(person.get("SEX", "")) == "2" else "M"

            y_bottom = n_persons - p_idx - 1

            for act in acts:
                code = str(act.get("activity_code", ""))
                cat = code_to_category(code)
                color = ACTIVITY_COLORS.get(cat, "#BDC3C7")

                try:
                    start_str = str(act.get("start_time", "00:00:00"))
                    stop_str = str(act.get("stop_time", "00:00:00"))
                    sp = start_str.split(":")
                    ep = stop_str.split(":")
                    start_h = int(sp[0]) + int(sp[1]) / 60
                    stop_h = int(ep[0]) + int(ep[1]) / 60
                except (ValueError, IndexError):
                    continue

                if stop_h <= start_h:
                    # Crosses midnight
                    ax_act.barh(y_bottom, 24 - start_h, left=start_h,
                                height=0.8, color=color, edgecolor="white",
                                linewidth=0.3)
                    ax_act.barh(y_bottom, stop_h, left=0,
                                height=0.8, color=color, edgecolor="white",
                                linewidth=0.3)
                else:
                    ax_act.barh(y_bottom, stop_h - start_h, left=start_h,
                                height=0.8, color=color, edgecolor="white",
                                linewidth=0.3)

            ax_act.text(-0.5, y_bottom + 0.4,
                        f"{age}{sex}",
                        ha="right", va="center", fontsize=9, fontweight="bold")

        ax_act.set_xlim(0, 24)
        ax_act.set_ylim(-0.5, n_persons - 0.2)
        ax_act.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax_act.set_xticklabels(["00", "04", "08", "12", "16", "20", "24"])
        ax_act.set_yticks([])
        if row_idx == n_hh - 1:
            ax_act.set_xlabel("Hour of Day")
        else:
            ax_act.set_xticklabels([])

        # Title with household info
        income_k = hd["income"] / 1000
        person_word = "person" if n_persons == 1 else "persons"
        title = (f"Household {spec['label']}: {spec['name']} "
                 f"({n_persons} {person_word}, {hd['climate']}, "
                 f"{hd['state_name']}, ${income_k:.0f}k)")
        ax_act.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax_act.axvspan(8, 17, alpha=0.04, color="gray")

        # --- Right panel: Temperature profile ---
        ax_temp = axes[row_idx, 1]
        hours = np.arange(24)
        temp = hd["hourly_temp"]
        ax_temp.plot(temp, hours, color="#E74C3C", lw=1.8)
        ax_temp.fill_betweenx(hours, temp, alpha=0.15, color="#E74C3C")
        ax_temp.set_ylim(-0.5, 23.5)
        ax_temp.set_yticks([0, 6, 12, 18])
        ax_temp.set_yticklabels(["00", "06", "12", "18"], fontsize=8)
        ax_temp.invert_yaxis()
        if row_idx == 0:
            ax_temp.set_title("T (°C)", fontsize=10)
        if row_idx == n_hh - 1:
            ax_temp.set_xlabel("°C")
        else:
            ax_temp.set_xticklabels([])
        ax_temp.grid(axis="x", alpha=0.3)

        # Annotate min/max temp
        t_min = np.nanmin(temp) if not np.all(np.isnan(temp)) else 0
        t_max = np.nanmax(temp) if not np.all(np.isnan(temp)) else 0
        ax_temp.text(0.95, 0.95, f"{t_max:.0f}°",
                     transform=ax_temp.transAxes, ha="right", va="top",
                     fontsize=9, color="#E74C3C")
        ax_temp.text(0.95, 0.05, f"{t_min:.0f}°",
                     transform=ax_temp.transAxes, ha="right", va="bottom",
                     fontsize=9, color="#3498DB")

    # Legend at bottom
    legend_patches = [mpatches.Patch(color=ACTIVITY_COLORS[cat], label=cat)
                      for cat in ACTIVITY_COLORS]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(ACTIVITY_COLORS), fontsize=10,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.45, -0.02))

    out_path = FIG_DIR / "fig_day_in_life.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
