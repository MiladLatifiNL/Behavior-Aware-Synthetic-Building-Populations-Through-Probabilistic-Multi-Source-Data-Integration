"""
Script 06: Energy Poverty Drivers

Generates four publication-quality figures characterising energy poverty
across the full Phase 4 synthetic population:

  A. Energy poverty rate by building type   (fig_poverty_building_type.pdf)
  B. Energy poverty rate by climate zone    (fig_poverty_climate_zone.pdf)
  C. Mean energy burden by heating fuel     (fig_heating_tenure_poverty.pdf)
     and tenure type (two-panel)
  D. Price Exposure Index by state          (fig_price_exposure_ranking.pdf)

All outputs are saved at 300 DPI as PDF in  revision/figures/.
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
# Paths
# ---------------------------------------------------------------------------
MANIFEST = Path("D:/PUMS_Enrichment/data/processed/phase4_shards/manifest.json")
FIG_DIR = Path("D:/PUMS_Enrichment/revision/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FIPS -> two-letter state abbreviation (covers all 50 states + DC + PR)
# ---------------------------------------------------------------------------
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR",
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


# ===================================================================
# Pass 1: stream all shards and accumulate lightweight aggregates
# ===================================================================
print("Streaming shards to accumulate aggregates ...")

# -- Figure A: building-type poverty ---------------------------------
bt_den = defaultdict(int)       # denominator (valid energy_burden)
bt_num10 = defaultdict(int)     # count with eb > 10
bt_num20 = defaultdict(int)     # count with eb > 20

# -- Figure B: climate-zone poverty ----------------------------------
cz_den = defaultdict(int)
cz_num10 = defaultdict(int)
cz_num20 = defaultdict(int)

# -- Figure C: mean energy burden by heating fuel / tenure -----------
hf_sum = defaultdict(float)
hf_cnt = defaultdict(int)
tt_sum = defaultdict(float)
tt_cnt = defaultdict(int)

# -- Figure D: state-level price exposure ----------------------------
state_energy_costs = defaultdict(list)   # total_energy_cost per building
state_incomes = defaultdict(list)        # HINCP per building

n_shards = 0
for df in iterate_shards():
    n_shards += 1
    if n_shards % 100 == 0:
        print(f"  ... processed {n_shards} shards")

    # Ensure needed columns exist
    eb = df["energy_burden"] if "energy_burden" in df.columns else None
    bt = df["building_type_simple"] if "building_type_simple" in df.columns else None
    cz = df["climate_zone"] if "climate_zone" in df.columns else None
    hf = df["heating_fuel"] if "heating_fuel" in df.columns else None
    tt = df["tenure_type"] if "tenure_type" in df.columns else None
    st = df["STATE"] if "STATE" in df.columns else None
    ec = df["total_energy_cost"] if "total_energy_cost" in df.columns else None
    hi = df["HINCP"] if "HINCP" in df.columns else None

    if eb is not None:
        valid = eb.notna()

        # --- A: by building type ------------------------------------
        if bt is not None:
            for btype, grp in df.loc[valid].groupby(bt):
                vals = grp["energy_burden"]
                bt_den[btype] += len(vals)
                bt_num10[btype] += int((vals > 10).sum())
                bt_num20[btype] += int((vals > 20).sum())

        # --- B: by climate zone -------------------------------------
        if cz is not None:
            for zone, grp in df.loc[valid].groupby(cz):
                vals = grp["energy_burden"]
                cz_den[zone] += len(vals)
                cz_num10[zone] += int((vals > 10).sum())
                cz_num20[zone] += int((vals > 20).sum())

        # --- C: mean burden by heating fuel -------------------------
        if hf is not None:
            for fuel, grp in df.loc[valid].groupby(hf):
                vals = grp["energy_burden"]
                hf_sum[fuel] += float(vals.sum())
                hf_cnt[fuel] += len(vals)

        # --- C: mean burden by tenure type --------------------------
        if tt is not None:
            for ttype, grp in df.loc[valid].groupby(tt):
                vals = grp["energy_burden"]
                tt_sum[ttype] += float(vals.sum())
                tt_cnt[ttype] += len(vals)

    # --- D: state-level aggregates (keep running sums) ---------------
    if st is not None and ec is not None and hi is not None:
        for state_code, grp in df.groupby(st):
            ec_vals = grp["total_energy_cost"].dropna()
            hi_vals = grp["HINCP"].dropna()
            if len(ec_vals) > 0:
                state_energy_costs[state_code].append(
                    (float(ec_vals.sum()), len(ec_vals))
                )
            if len(hi_vals) > 0:
                # For median we collect a reservoir sample per state.
                # Since we cannot compute exact median across shards
                # without storing all values, we accumulate raw values
                # but cap at 50k per state to limit memory.
                existing = state_incomes[state_code]
                if len(existing) < 50000:
                    state_incomes[state_code].extend(hi_vals.tolist())

    del df
    gc.collect()

print(f"Done streaming {n_shards} shards.\n")

# ===================================================================
# Overall population energy poverty rates (for LaTeX verification)
# ===================================================================
total_den = sum(bt_den.values())
total_num6 = 0
total_num10 = sum(bt_num10.values())
total_num20 = sum(bt_num20.values())

# We need the 6% threshold too -- recount from building-type level is not
# possible since we only tracked 10% and 20%.  Run a quick pass to compute
# the 6% count.
print("Computing overall poverty rates (6%/10%/20% thresholds) ...")
overall_num6 = 0
overall_den_check = 0
for df in iterate_shards():
    if "energy_burden" in df.columns:
        eb = df["energy_burden"].dropna()
        overall_den_check += len(eb)
        overall_num6 += int((eb > 6).sum())
    del df
    gc.collect()

pct_6 = 100.0 * overall_num6 / overall_den_check if overall_den_check > 0 else 0
pct_10 = 100.0 * total_num10 / total_den if total_den > 0 else 0
pct_20 = 100.0 * total_num20 / total_den if total_den > 0 else 0

print(f"  Total buildings with valid energy_burden: {total_den:,}")
print(f"  Energy poverty rate (> 6% burden):  {pct_6:.1f}%  ({overall_num6:,} / {overall_den_check:,})")
print(f"  Energy poverty rate (>10% burden):  {pct_10:.1f}%  ({total_num10:,} / {total_den:,})")
print(f"  Energy poverty rate (>20% burden):  {pct_20:.1f}%  ({total_num20:,} / {total_den:,})")
print()


# ===================================================================
# Figure A: Energy Poverty by Building Type
# ===================================================================
print("Generating Figure A: Poverty by Building Type ...")

types = sorted(bt_den.keys())
share10 = [bt_num10[t] / bt_den[t] if bt_den[t] > 0 else 0 for t in types]
share20 = [bt_num20[t] / bt_den[t] if bt_den[t] > 0 else 0 for t in types]

# Pretty labels
label_map = {
    "single_family": "Single Family",
    "small_multi": "Small Multi",
    "large_multi": "Large Multi",
    "mobile": "Mobile Home",
    "other": "Other",
}
labels = [label_map.get(t, t.replace("_", " ").title()) for t in types]

x = np.arange(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width / 2, [s * 100 for s in share10], width,
               label="> 10 % burden", color="#E8960C", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width / 2, [s * 100 for s in share20], width,
               label="> 20 % burden", color="#C0392B", edgecolor="white", linewidth=0.5)

ax.set_xlabel("Building Type", fontsize=12)
ax.set_ylabel("Share of Households (%)", fontsize=12)
ax.set_title("Energy Poverty Rate by Building Type", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=11, frameon=False)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# Value labels on bars
for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_poverty_building_type.pdf", dpi=300)
plt.close(fig)
print("  -> Saved fig_poverty_building_type.pdf")


# ===================================================================
# Figure B: Energy Poverty by Climate Zone
# ===================================================================
print("Generating Figure B: Poverty by Climate Zone ...")

zones = sorted(cz_den.keys())
share10_cz = [cz_num10[z] / cz_den[z] if cz_den[z] > 0 else 0 for z in zones]
share20_cz = [cz_num20[z] / cz_den[z] if cz_den[z] > 0 else 0 for z in zones]

zone_labels = [z.replace("_", " ").title() for z in zones]

x = np.arange(len(zones))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width / 2, [s * 100 for s in share10_cz], width,
               label="> 10 % burden", color="#E8960C", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width / 2, [s * 100 for s in share20_cz], width,
               label="> 20 % burden", color="#C0392B", edgecolor="white", linewidth=0.5)

ax.set_xlabel("Climate Zone", fontsize=12)
ax.set_ylabel("Share of Households (%)", fontsize=12)
ax.set_title("Energy Poverty Rate by Climate Zone", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(zone_labels, fontsize=12)
ax.legend(fontsize=11, frameon=False)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_poverty_climate_zone.pdf", dpi=300)
plt.close(fig)
print("  -> Saved fig_poverty_climate_zone.pdf")


# ===================================================================
# Figure C: Mean Energy Burden by Heating Fuel & Tenure Type
# ===================================================================
print("Generating Figure C: Heating Fuel & Tenure Poverty ...")

# Heating fuel
fuels = sorted(hf_sum.keys())
fuel_means = [hf_sum[f] / hf_cnt[f] if hf_cnt[f] > 0 else 0 for f in fuels]
fuel_order = np.argsort(fuel_means)[::-1]
fuels_sorted = [fuels[i] for i in fuel_order]
fuel_means_sorted = [fuel_means[i] for i in fuel_order]
fuel_labels = [f.replace("_", " ").title() for f in fuels_sorted]

# Tenure type
tenures = sorted(tt_sum.keys())
tenure_means = [tt_sum[t] / tt_cnt[t] if tt_cnt[t] > 0 else 0 for t in tenures]
tenure_order = np.argsort(tenure_means)[::-1]
tenures_sorted = [tenures[i] for i in tenure_order]
tenure_means_sorted = [tenure_means[i] for i in tenure_order]
tenure_labels = [t.replace("_", " ").title() for t in tenures_sorted]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Heating fuel
colors_fuel = plt.cm.Oranges(np.linspace(0.4, 0.85, len(fuels_sorted)))
ax1.barh(range(len(fuels_sorted)), fuel_means_sorted, color=colors_fuel,
         edgecolor="white", linewidth=0.5)
ax1.set_yticks(range(len(fuels_sorted)))
ax1.set_yticklabels(fuel_labels, fontsize=11)
ax1.set_xlabel("Mean Energy Burden (%)", fontsize=12)
ax1.set_title("Energy Burden by Heating Fuel", fontsize=13, fontweight="bold")
ax1.invert_yaxis()
for i, v in enumerate(fuel_means_sorted):
    ax1.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right panel: Tenure type
colors_tenure = plt.cm.Reds(np.linspace(0.4, 0.85, len(tenures_sorted)))
ax2.barh(range(len(tenures_sorted)), tenure_means_sorted, color=colors_tenure,
         edgecolor="white", linewidth=0.5)
ax2.set_yticks(range(len(tenures_sorted)))
ax2.set_yticklabels(tenure_labels, fontsize=11)
ax2.set_xlabel("Mean Energy Burden (%)", fontsize=12)
ax2.set_title("Energy Burden by Tenure Type", fontsize=13, fontweight="bold")
ax2.invert_yaxis()
for i, v in enumerate(tenure_means_sorted):
    ax2.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout(w_pad=3)
fig.savefig(FIG_DIR / "fig_heating_tenure_poverty.pdf", dpi=300)
plt.close(fig)
print("  -> Saved fig_heating_tenure_poverty.pdf")


# ===================================================================
# Figure D: Price Exposure Index by State (Top 20)
# ===================================================================
print("Generating Figure D: Price Exposure Index ...")

price_exposure = {}
for state_code in state_energy_costs:
    # Weighted mean of energy cost
    total_cost = sum(s for s, _ in state_energy_costs[state_code])
    total_n = sum(n for _, n in state_energy_costs[state_code])
    mean_cost = total_cost / total_n if total_n > 0 else 0

    # Median income (from reservoir sample)
    incomes = state_incomes.get(state_code, [])
    if len(incomes) > 0:
        median_inc = float(np.median(incomes))
    else:
        median_inc = 0

    if median_inc > 0:
        price_exposure[state_code] = mean_cost / median_inc
    # else skip (can't compute ratio)

# Sort by exposure, take top 20
pe_sorted = sorted(price_exposure.items(), key=lambda x: x[1], reverse=True)[:20]
pe_states = [s for s, _ in pe_sorted]
pe_values = [v for _, v in pe_sorted]
pe_labels = [FIPS_TO_ABBR.get(str(s).zfill(2), str(s)) for s in pe_states]

fig, ax = plt.subplots(figsize=(9, 7))
colors_pe = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(pe_values)))
ax.barh(range(len(pe_values)), pe_values, color=colors_pe,
        edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(pe_values)))
ax.set_yticklabels(pe_labels, fontsize=12)
ax.set_xlabel("Price Exposure Index (Energy Cost / Median Income)", fontsize=12)
ax.set_title("Price Exposure Index by State (Energy Cost / Median Income)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()

# Annotate values
for i, v in enumerate(pe_values):
    ax.text(v + max(pe_values) * 0.01, i, f"{v:.4f}", va="center", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_price_exposure_ranking.pdf", dpi=300)
plt.close(fig)
print("  -> Saved fig_price_exposure_ranking.pdf")


print("\nAll Figure 06 outputs saved to:", FIG_DIR)
