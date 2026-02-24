"""
Reviewer Comment 7: Matching transparency visualization.

Generates a single compact 2x2 figure (PDF + PNG) with four contingency
heatmaps showing how PUMS socioeconomic categories (income quintiles,
tenure) flow into RECS physical templates (building vintage, efficiency
grade) through the probabilistic linkage algorithm.

Usage:
    python run_matching_flow_visualization.py
    python run_matching_flow_visualization.py --max-shards 10        # Quick test
    python run_matching_flow_visualization.py --output-dir paper/
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ordered category labels
# ---------------------------------------------------------------------------
INCOME_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
INCOME_LABELS = {
    "Q1": "Q1\n(Lowest)", "Q2": "Q2", "Q3": "Q3",
    "Q4": "Q4", "Q5": "Q5\n(Highest)",
}

ERA_ORDER = ["pre1950", "1950-1980", "1980-2000", "2000-2010", "post2010"]
ERA_LABELS = {
    "pre1950": "Pre-\n1950",
    "1950-1980": "1950\u2013\n1979",
    "1980-2000": "1980\u2013\n1999",
    "2000-2010": "2000\u2013\n2015",
    "post2010": "2016\u2013\n2020",
}

TENURE_ORDER = ["owned_clear", "owned_mortgage", "rented"]
TENURE_LABELS = {
    "owned_clear": "Owned\n(no mortgage)",
    "owned_mortgage": "Owned\n(mortgage)",
    "rented": "Rented",
}

EFFICIENCY_ORDER = ["very_low", "low", "medium", "high"]
EFFICIENCY_LABELS = {
    "very_low": "Very\nLow", "low": "Low",
    "medium": "Medium", "high": "High",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_from_shards(shards_dir: str, max_shards: int = 0) -> pd.DataFrame:
    """Stream Phase 2 shards and extract relevant columns."""
    sdir = Path(shards_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    cols_needed = [
        "income_quintile", "tenure_type", "building_era",
        "efficiency_proxy", "match_probability", "match_weight",
    ]

    files = manifest["files"]
    if max_shards > 0:
        files = files[:max_shards]

    logger.info("Loading %d shards from %s ...", len(files), sdir)
    frames = []
    for i, fname in enumerate(files):
        fpath = sdir / fname
        if not fpath.exists():
            continue
        df = pd.read_pickle(fpath)
        available = [c for c in cols_needed if c in df.columns]
        frames.append(df[available])
        if (i + 1) % 100 == 0:
            logger.info("  loaded %d / %d shards", i + 1, len(files))

    result = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d buildings total.", len(result))
    return result


# ============================================================
# HELPERS
# ============================================================

def cramers_v(ct_array: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(ct_array)
    n = ct_array.sum()
    r, k = ct_array.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def _draw_panel(
    ax,
    cross_tab: pd.DataFrame,
    row_order: list,
    col_order: list,
    row_labels: dict,
    col_labels: dict,
    title: str,
    ylabel: str,
    xlabel: str,
    cmap: str,
):
    """Draw one heatmap panel on the given axes. Returns (chi2, p, dof, V)."""
    rows = [r for r in row_order if r in cross_tab.index]
    cols = [c for c in col_order if c in cross_tab.columns]
    ct = cross_tab.loc[rows, cols].copy()

    row_sums = ct.sum(axis=1).replace(0, 1)
    ct_pct = ct.div(row_sums, axis=0) * 100

    # Build annotation: "XX.X%\n(n=NN,NNN)"
    annot = ct_pct.copy().astype(object)
    for r in rows:
        for c in cols:
            annot.loc[r, c] = f"{ct_pct.loc[r, c]:.1f}%\n({int(ct.loc[r, c]):,})"

    ct_pct.index = [row_labels.get(r, r) for r in rows]
    ct_pct.columns = [col_labels.get(c, c) for c in cols]
    annot.index = ct_pct.index
    annot.columns = ct_pct.columns

    sns.heatmap(
        ct_pct, annot=annot, fmt="", cmap=cmap,
        linewidths=0.8, linecolor="white", ax=ax,
        vmin=0, vmax=ct_pct.values.max() * 1.15,
        cbar_kws={"label": "Row %", "shrink": 0.85},
        annot_kws={"fontsize": 6.5},
    )
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    # Chi-square annotation inside the bottom of the heatmap
    ct_raw = cross_tab.loc[rows, cols].values
    chi2, p, dof, _ = chi2_contingency(ct_raw)
    v = cramers_v(ct_raw)
    ax.text(
        0.02, 0.02,
        rf"$\chi^2\!=\!{chi2:,.0f}$,  $V\!=\!{v:.3f}$",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, lw=0.4),
    )
    return chi2, p, dof, v


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reviewer 7: matching flow visualization"
    )
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Limit shards for quick testing (0 = all)")
    parser.add_argument("--output-dir", type=str, default="paper",
                        help="Directory for output figures")
    parser.add_argument("--shards-dir", type=str,
                        default="data/processed/phase2_shards",
                        help="Phase 2 shard directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_dir = Path("results/reviewer7")
    stats_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df = load_from_shards(args.shards_dir, max_shards=args.max_shards)

    df = df[df["income_quintile"].isin(INCOME_ORDER)].copy()
    df = df[df["building_era"].isin(ERA_ORDER)].copy()
    df["tenure_simple"] = df["tenure_type"].where(
        df["tenure_type"].isin(TENURE_ORDER)
    )
    df = df.dropna(subset=["tenure_simple"])
    logger.info("After filtering: %d buildings", len(df))

    df_eff = df[df["efficiency_proxy"].isin(EFFICIENCY_ORDER)].copy()

    # --- Cross-tabulations ---
    ct_income_era = pd.crosstab(df["income_quintile"], df["building_era"])
    ct_income_eff = pd.crosstab(df_eff["income_quintile"], df_eff["efficiency_proxy"])
    ct_tenure_era = pd.crosstab(df["tenure_simple"], df["building_era"])
    ct_tenure_eff = pd.crosstab(df_eff["tenure_simple"], df_eff["efficiency_proxy"])

    # ================================================================
    # SINGLE 2x2 FIGURE
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0), constrained_layout=True)

    panels = [
        # (ax,  cross_tab,     row_order,     col_order,        row_labels,     col_labels,       title,                                          ylabel,                xlabel,                 cmap)
        (axes[0, 0], ct_income_era, INCOME_ORDER, ERA_ORDER,        INCOME_LABELS, ERA_LABELS,
         "(a) Income quintile \u00d7 building vintage",   "Income Quintile", "Building Vintage",  "YlOrRd"),
        (axes[0, 1], ct_income_eff, INCOME_ORDER, EFFICIENCY_ORDER, INCOME_LABELS, EFFICIENCY_LABELS,
         "(b) Income quintile \u00d7 efficiency grade",   "Income Quintile", "Efficiency Grade",  "PuBuGn"),
        (axes[1, 0], ct_tenure_era, TENURE_ORDER, ERA_ORDER,        TENURE_LABELS, ERA_LABELS,
         "(c) Tenure type \u00d7 building vintage",       "Tenure Type",     "Building Vintage",  "YlGnBu"),
        (axes[1, 1], ct_tenure_eff, TENURE_ORDER, EFFICIENCY_ORDER, TENURE_LABELS, EFFICIENCY_LABELS,
         "(d) Tenure type \u00d7 efficiency grade",       "Tenure Type",     "Efficiency Grade",  "BuPu"),
    ]

    results = {}
    keys = ["income_x_vintage", "income_x_efficiency",
            "tenure_x_vintage", "tenure_x_efficiency"]

    for (ax, ct, ro, co, rl, cl, title, yl, xl, cmap), key in zip(panels, keys):
        chi2, p, dof, v = _draw_panel(ax, ct, ro, co, rl, cl, title, yl, xl, cmap)
        results[key] = dict(chi2=chi2, p=p, dof=dof, cramers_v=v)

    for ext in ["pdf", "png"]:
        path = out / f"matching_flow_heatmaps.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved matching_flow_heatmaps.pdf  and  .png")
    plt.close(fig)

    # ================================================================
    # STATS JSON
    # ================================================================
    cross_tabs = {
        "income_x_vintage": ct_income_era,
        "income_x_efficiency": ct_income_eff,
        "tenure_x_vintage": ct_tenure_era,
        "tenure_x_efficiency": ct_tenure_eff,
    }
    stats = {"n_buildings": int(len(df))}
    for key, res in results.items():
        stats[key] = {
            "chi2": float(res["chi2"]),
            "p_value": float(res["p"]) if res["p"] is not None else 0.0,
            "dof": int(res["dof"]),
            "cramers_v": float(res["cramers_v"]),
            "cross_tab": cross_tabs[key].to_dict(),
        }
    stats_path = stats_dir / "matching_flow_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved stats to %s", stats_path)

    # ================================================================
    # CONSOLE SUMMARY
    # ================================================================
    print("\n" + "=" * 65)
    print("MATCHING FLOW ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Buildings analysed: {len(df):,}\n")
    for key, res in results.items():
        print(f"{key}:  chi2={res['chi2']:,.0f}  V={res['cramers_v']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
