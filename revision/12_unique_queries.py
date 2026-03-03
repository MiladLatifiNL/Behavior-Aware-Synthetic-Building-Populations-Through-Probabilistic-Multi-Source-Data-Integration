"""
Script 12: Unique Analytical Queries — What Only Integration Can Answer
=======================================================================

Demonstrates six concrete research questions that are impossible to answer
with any single source survey but become answerable with the unified dataset.
Each query requires fields from at least two source datasets.

Outputs:
  revision/figures/fig_unique_queries.pdf      — 2×3 grid of mini-panels
  console output with exact numbers for LaTeX   tab:unique-queries

Output: revision/figures/fig_unique_queries.pdf
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
    "axes.titlesize": 12,
    "axes.labelsize": 11,
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

HOME_LOCATIONS = {"1", "-1"}

HH_TYPES_ORDERED = [
    "Dual-earner + children",
    "Single parent",
    "Single young professional",
    "Retired couple",
    "Single elderly",
    "Other",
]


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
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Script 12: Unique Analytical Queries")
    print("=" * 60)

    # ---- Query accumulators ----

    # Q1: Low-income elderly in hot-humid with energy burden >20%
    # Needs: PUMS (income_quintile, has_seniors) + RECS (energy_burden) + NSRDB (climate_zone)
    q1_total = 0
    q1_high_burden = 0

    # Q2: WFH vs non-WFH midday occupancy by building type
    # Needs: PUMS (wfh status) + ATUS (activity schedules) + RECS (building_type)
    q2_occ = defaultdict(lambda: {"wfh_home": 0, "wfh_total": 0,
                                   "nowfh_home": 0, "nowfh_total": 0})

    # Q3: Solar self-consumption hours retired vs dual-earner
    # Needs: ATUS (at-home during solar hours) + NSRDB (solar) + PUMS (demographics)
    q3_data = {"Retired couple": {"solar_home_hrs_sum": 0.0, "count": 0},
               "Dual-earner + children": {"solar_home_hrs_sum": 0.0, "count": 0}}

    # Q4: Thermal discomfort gap (high HVAC need + low income)
    # Needs: all four datasets
    q4_hvac_by_quintile = defaultdict(list)  # income_quintile -> list of mean hvac per building

    # Q5: Single parent vs dual-earner leisure minutes
    # Needs: PUMS (household composition) + ATUS (daily_time_use)
    q5_leisure = {"Single parent": [], "Dual-earner + children": []}

    # Q6: HVAC load when nobody is home
    # Needs: RECS (building) + NSRDB (weather/hvac) + ATUS (occupancy)
    q6_by_type = defaultdict(lambda: {"phantom_sum": 0.0, "total_sum": 0.0, "count": 0})

    n_buildings = 0
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
            income_q = str(row.get("income_quintile", ""))
            climate = str(row.get("climate_zone", ""))
            has_seniors_val = bool(row.get("has_seniors", False))
            btype = str(row.get("building_type_simple", ""))
            energy_burden = row.get("energy_burden", None)
            wfh = bool(row.get("likely_wfh", False)) or int(row.get("person_work_from_home_count", 0)) > 0

            # ---------- Q1: Low-income elderly in hot-humid ----------
            if income_q == "Q1" and has_seniors_val and "hot" in climate.lower():
                q1_total += 1
                if energy_burden is not None:
                    try:
                        if float(energy_burden) > 20:
                            q1_high_burden += 1
                    except (ValueError, TypeError):
                        pass

            # ---------- Q2: WFH midday occupancy ----------
            midday_home_count = 0
            midday_person_count = 0
            for person in persons:
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list):
                    continue
                # Check if home during 10-14
                home_minutes = np.zeros(1440, dtype=bool)
                for act in acts:
                    loc = str(act.get("location", ""))
                    code = str(act.get("activity_code", ""))
                    is_home = loc in HOME_LOCATIONS
                    if not is_home and loc in ("", "None", "nan"):
                        code_2 = code[:2] if len(code) >= 2 else ""
                        is_home = code_2 in ("01", "02", "05", "11")
                    if not is_home:
                        continue
                    try:
                        ss = str(act.get("start_time", "00:00:00")).split(":")
                        es = str(act.get("stop_time", "00:00:00")).split(":")
                        sm = int(ss[0]) * 60 + int(ss[1])
                        em = int(es[0]) * 60 + int(es[1])
                    except (ValueError, IndexError):
                        continue
                    if em <= sm:
                        home_minutes[sm:1440] = True
                        home_minutes[0:em] = True
                    else:
                        home_minutes[sm:em] = True

                midday_mins = home_minutes[600:840].sum()  # 10:00-14:00
                midday_person_count += 1
                if midday_mins >= 120:  # Home for at least 2 of 4 hours
                    midday_home_count += 1

            if midday_person_count > 0 and btype:
                key = btype
                if wfh:
                    q2_occ[key]["wfh_home"] += midday_home_count
                    q2_occ[key]["wfh_total"] += midday_person_count
                else:
                    q2_occ[key]["nowfh_home"] += midday_home_count
                    q2_occ[key]["nowfh_total"] += midday_person_count

            # ---------- Q3: Solar self-consumption ----------
            if hh_type in q3_data:
                solar_home_hours = 0
                for person in persons:
                    acts = person.get("activity_sequence", [])
                    if not isinstance(acts, list):
                        continue
                    for act in acts:
                        loc = str(act.get("location", ""))
                        code = str(act.get("activity_code", ""))
                        is_home = loc in HOME_LOCATIONS
                        if not is_home and loc in ("", "None", "nan"):
                            code_2 = code[:2] if len(code) >= 2 else ""
                            is_home = code_2 in ("01", "02", "05", "11")
                        if not is_home:
                            continue
                        try:
                            ss = str(act.get("start_time", "00:00:00")).split(":")
                            start_h = int(ss[0])
                            dur = int(act.get("duration_minutes", 0))
                        except (ValueError, TypeError, IndexError):
                            continue
                        # Solar hours: 9-16
                        if 9 <= start_h <= 16 and dur > 0:
                            solar_home_hours += dur / 60.0

                q3_data[hh_type]["solar_home_hrs_sum"] += solar_home_hours
                q3_data[hh_type]["count"] += 1

            # ---------- Q4: Thermal discomfort gap ----------
            if income_q:
                total_hvac = 0.0
                hvac_count = 0
                for person in persons:
                    acts = person.get("activity_sequence", [])
                    if not isinstance(acts, list):
                        continue
                    for act in acts:
                        weather = act.get("weather", {})
                        if not isinstance(weather, dict):
                            continue
                        hvac = weather.get("hvac_load", None)
                        if hvac is not None:
                            try:
                                total_hvac += float(hvac)
                                hvac_count += 1
                            except (ValueError, TypeError):
                                pass
                if hvac_count > 0:
                    q4_hvac_by_quintile[income_q].append(total_hvac / hvac_count)

            # ---------- Q5: Leisure minutes ----------
            if hh_type in q5_leisure:
                for person in persons:
                    dtu = person.get("daily_time_use", {})
                    if isinstance(dtu, dict):
                        leisure = dtu.get("leisure", 0)
                        try:
                            q5_leisure[hh_type].append(float(leisure))
                        except (ValueError, TypeError):
                            pass

            # ---------- Q6: Phantom HVAC load ----------
            total_hvac_load = 0.0
            phantom_hvac_load = 0.0
            for person in persons:
                acts = person.get("activity_sequence", [])
                if not isinstance(acts, list):
                    continue
                for act in acts:
                    weather = act.get("weather", {})
                    if not isinstance(weather, dict):
                        continue
                    hvac = weather.get("hvac_load", None)
                    dur = act.get("duration_minutes", 0)
                    if hvac is None:
                        continue
                    try:
                        hvac_val = float(hvac) * int(dur)
                    except (ValueError, TypeError):
                        continue
                    total_hvac_load += hvac_val

                    loc = str(act.get("location", ""))
                    code = str(act.get("activity_code", ""))
                    is_home = loc in HOME_LOCATIONS
                    if not is_home and loc in ("", "None", "nan"):
                        code_2 = code[:2] if len(code) >= 2 else ""
                        is_home = code_2 in ("01", "02", "05", "11")
                    if not is_home:
                        phantom_hvac_load += hvac_val

            if total_hvac_load > 0:
                q6_by_type[hh_type]["phantom_sum"] += phantom_hvac_load
                q6_by_type[hh_type]["total_sum"] += total_hvac_load
                q6_by_type[hh_type]["count"] += 1

        del df
        gc.collect()

    print(f"\nComplete: {n_buildings:,} buildings, {n_shards} shards")

    # ---- Compute results ----
    print("\n" + "=" * 60)
    print("QUERY RESULTS (for LaTeX table)")
    print("=" * 60)

    # Q1
    q1_pct = (q1_high_burden / q1_total * 100) if q1_total > 0 else 0
    print(f"\nQ1: Low-income elderly in hot-humid with burden >20%")
    print(f"    {q1_high_burden}/{q1_total} = {q1_pct:.1f}%")

    # Q2
    print(f"\nQ2: WFH vs non-WFH midday at-home rate by building type")
    q2_results = {}
    for btype in sorted(q2_occ.keys()):
        d = q2_occ[btype]
        wfh_rate = d["wfh_home"] / d["wfh_total"] * 100 if d["wfh_total"] > 0 else 0
        nowfh_rate = d["nowfh_home"] / d["nowfh_total"] * 100 if d["nowfh_total"] > 0 else 0
        diff = wfh_rate - nowfh_rate
        q2_results[btype] = {"wfh": wfh_rate, "no_wfh": nowfh_rate, "diff": diff}
        print(f"    {btype}: WFH={wfh_rate:.1f}% vs non-WFH={nowfh_rate:.1f}% "
              f"(+{diff:.1f}pp)")

    # Q3
    print(f"\nQ3: Solar self-consumption hours (home during 9-16)")
    q3_results = {}
    for hh_type in ["Retired couple", "Dual-earner + children"]:
        d = q3_data[hh_type]
        mean_hrs = d["solar_home_hrs_sum"] / d["count"] if d["count"] > 0 else 0
        q3_results[hh_type] = mean_hrs
        print(f"    {hh_type}: {mean_hrs:.1f} hrs/day (n={d['count']:,})")

    # Q4
    print(f"\nQ4: Mean HVAC demand by income quintile")
    q4_means = {}
    for q in sorted(q4_hvac_by_quintile.keys()):
        vals = q4_hvac_by_quintile[q]
        mean_val = np.mean(vals) if vals else 0
        q4_means[q] = mean_val
        print(f"    {q}: mean HVAC = {mean_val:.3f} (n={len(vals):,})")

    # Q5
    print(f"\nQ5: Leisure minutes per person per day")
    q5_means = {}
    for hh_type in ["Single parent", "Dual-earner + children"]:
        vals = q5_leisure[hh_type]
        mean_val = np.mean(vals) if vals else 0
        q5_means[hh_type] = mean_val
        print(f"    {hh_type}: {mean_val:.0f} min/day (n={len(vals):,})")
    if all(q5_means.values()):
        diff = q5_means.get("Dual-earner + children", 0) - q5_means.get("Single parent", 0)
        print(f"    Difference: {diff:+.0f} min/day")

    # Q6
    print(f"\nQ6: Phantom HVAC (running when nobody home) by household type")
    q6_results = {}
    for hh_type in HH_TYPES_ORDERED:
        d = q6_by_type.get(hh_type, {"phantom_sum": 0, "total_sum": 0, "count": 0})
        phantom_pct = d["phantom_sum"] / d["total_sum"] * 100 if d["total_sum"] > 0 else 0
        q6_results[hh_type] = phantom_pct
        print(f"    {hh_type}: {phantom_pct:.1f}% phantom (n={d['count']:,})")

    # Overall phantom
    total_phantom = sum(d["phantom_sum"] for d in q6_by_type.values())
    total_hvac_all = sum(d["total_sum"] for d in q6_by_type.values())
    overall_phantom = total_phantom / total_hvac_all * 100 if total_hvac_all > 0 else 0
    print(f"    OVERALL: {overall_phantom:.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: Q1 - Single percentage bar
    ax = axes[0, 0]
    ax.barh(["Low-income elderly\nin hot-humid"], [q1_pct],
            color="#E74C3C", edgecolor="white", height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Households with Burden > 20% (%)")
    ax.set_title("Q1: Energy Poverty in\nVulnerable Elderly (PUMS+RECS+NSRDB)", fontsize=10)
    ax.text(q1_pct + 2, 0, f"{q1_pct:.1f}%", va="center", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: Q2 - WFH vs non-WFH grouped bars
    ax = axes[0, 1]
    btypes = [k for k in sorted(q2_results.keys()) if k != "other"][:4]
    if not btypes:
        btypes = list(q2_results.keys())[:4]
    x = np.arange(len(btypes))
    w = 0.35
    wfh_vals = [q2_results[b]["wfh"] for b in btypes]
    nowfh_vals = [q2_results[b]["no_wfh"] for b in btypes]
    ax.bar(x - w/2, wfh_vals, w, label="WFH", color="#3498DB", edgecolor="white")
    ax.bar(x + w/2, nowfh_vals, w, label="Non-WFH", color="#E74C3C", edgecolor="white")
    ax.set_xticks(x)
    labels = [b.replace("_", "\n") for b in btypes]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Midday At-Home Rate (%)")
    ax.set_title("Q2: WFH Impact on\nMidday Occupancy (PUMS+ATUS+RECS)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Q3 - Solar self-consumption
    ax = axes[0, 2]
    types_q3 = list(q3_results.keys())
    vals_q3 = [q3_results[t] for t in types_q3]
    colors_q3 = ["#27AE60", "#E74C3C"]
    bars = ax.bar(types_q3, vals_q3, color=colors_q3, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals_q3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}h", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Solar-Concurrent At-Home Hours")
    ax.set_title("Q3: Solar Self-Consumption\nPotential (ATUS+NSRDB+PUMS)", fontsize=10)
    ax.set_xticklabels([t.replace(" + ", "\n+ ") for t in types_q3], fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Q4 - HVAC demand by income quintile
    ax = axes[1, 0]
    quintiles = sorted(q4_means.keys())
    hvac_vals = [q4_means[q] for q in quintiles]
    colors_q4 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(quintiles)))
    ax.bar(quintiles, hvac_vals, color=colors_q4, edgecolor="white", width=0.6)
    ax.set_xlabel("Income Quintile")
    ax.set_ylabel("Mean HVAC Demand Indicator")
    ax.set_title("Q4: Thermal Discomfort Gap\n(All Four Datasets)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel 5: Q5 - Leisure comparison
    ax = axes[1, 1]
    types_q5 = ["Single parent", "Dual-earner\n+ children"]
    vals_q5 = [q5_means.get("Single parent", 0),
               q5_means.get("Dual-earner + children", 0)]
    colors_q5 = ["#F39C12", "#E74C3C"]
    bars = ax.bar(types_q5, vals_q5, color=colors_q5, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals_q5):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f} min", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Daily Leisure (min/person)")
    ax.set_title("Q5: Leisure Time Gap\n(PUMS+ATUS)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel 6: Q6 - Phantom HVAC by household type
    ax = axes[1, 2]
    types_q6 = [t for t in HH_TYPES_ORDERED if q6_results.get(t, 0) > 0]
    if not types_q6:
        types_q6 = HH_TYPES_ORDERED
    vals_q6 = [q6_results.get(t, 0) for t in types_q6]
    sorted_pairs = sorted(zip(types_q6, vals_q6), key=lambda x: x[1], reverse=True)
    types_q6 = [p[0] for p in sorted_pairs]
    vals_q6 = [p[1] for p in sorted_pairs]
    colors_q6 = plt.cm.Oranges(np.linspace(0.3, 0.8, len(types_q6)))
    y_pos = range(len(types_q6))
    bars = ax.barh(y_pos, vals_q6, color=colors_q6, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace(" + ", "\n+ ") for t in types_q6], fontsize=9)
    ax.set_xlabel("HVAC Running While Away (%)")
    ax.set_title("Q6: Phantom HVAC Load\n(RECS+NSRDB+ATUS)", fontsize=10)
    for bar, val in zip(bars, vals_q6):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(h_pad=3, w_pad=2)
    out_path = FIG_DIR / "fig_unique_queries.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Print LaTeX-ready summary
    print("\n" + "=" * 60)
    print("LATEX TABLE CONTENT:")
    print("=" * 60)
    wfh_diffs = [q2_results[b]["diff"] for b in q2_results if q2_results[b].get("diff", 0) != 0]
    overall_wfh_diff = np.mean(wfh_diffs) if wfh_diffs else 0
    print(f"Q1: {q1_pct:.1f}\\%")
    print(f"Q2: +{overall_wfh_diff:.0f} pp higher")
    q3_retired = q3_results.get("Retired couple", 0)
    q3_dual = q3_results.get("Dual-earner + children", 0)
    print(f"Q3: {q3_retired:.1f} vs {q3_dual:.1f} hrs")
    q4_q1 = q4_means.get("Q1", 0)
    q4_q5 = q4_means.get("Q5", 0)
    print(f"Q4: Q1 HVAC={q4_q1:.3f}, Q5 HVAC={q4_q5:.3f}")
    sp_leisure = q5_means.get("Single parent", 0)
    de_leisure = q5_means.get("Dual-earner + children", 0)
    print(f"Q5: {abs(de_leisure - sp_leisure):.0f} min/day {'less' if sp_leisure < de_leisure else 'more'}")
    print(f"Q6: {overall_phantom:.1f}\\% overall")


if __name__ == "__main__":
    main()
