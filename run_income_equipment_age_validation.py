"""
Reviewer 6 validation: Income vs Equipment Age bivariate consistency.

Compares the joint distribution of income quintile and HVAC equipment age
between (a) raw RECS 2020 weighted survey data and (b) the Phase 2
matched synthetic population, to rule out spurious assignment of
high-efficiency equipment to low-income households.

Usage:
    python run_reviewer6_validation.py
    python run_reviewer6_validation.py --max-shards 10          # Quick test
    python run_reviewer6_validation.py --output-dir results/reviewer6
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ks_2samp, chi2_contingency, spearmanr

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RECS 2020 codebook mappings
# ---------------------------------------------------------------------------
EQUIPAGE_LABELS = {
    1: '<2 yr', 2: '2-4 yr', 3: '5-9 yr',
    4: '10-14 yr', 5: '15-19 yr', 6: '20+ yr',
}
EQUIPAGE_MIDPOINTS = {1: 1, 2: 3, 3: 7, 4: 12, 5: 17, 6: 22}

YEARMADERANGE_LABELS = {
    1: 'Pre-1950', 2: '1950s', 3: '1960s', 4: '1970s',
    5: '1980s', 6: '1990s', 7: '2000s', 8: '2010-15', 9: '2016-20',
}
YEARMADERANGE_MIDPOINTS = {
    1: 1940, 2: 1955, 3: 1965, 4: 1975,
    5: 1985, 6: 1995, 7: 2005, 8: 2013, 9: 2018,
}

MONEYPY_MIDPOINTS = {
    1: 2_500, 2: 6_250, 3: 8_750, 4: 12_500, 5: 17_500,
    6: 22_500, 7: 27_500, 8: 32_500, 9: 37_500, 10: 45_000,
    11: 55_000, 12: 70_000, 13: 90_000, 14: 110_000,
    15: 130_000, 16: 160_000,
}

QUINTILE_LABELS = ['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)']
QUINTILE_KEYS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw_recs(recs_path: str) -> pd.DataFrame:
    """Load raw RECS 2020 with equipment age, income, and survey weights."""
    logger.info(f"Loading raw RECS from {recs_path}")
    cols = ['DOEID', 'MONEYPY', 'EQUIPAGE', 'ACEQUIPAGE',
            'YEARMADERANGE', 'NWEIGHT']
    recs = pd.read_csv(recs_path, usecols=cols, low_memory=False)
    logger.info(f"  Raw RECS records: {len(recs):,}")

    # Map income to midpoint dollars
    recs['income_midpoint'] = recs['MONEYPY'].map(MONEYPY_MIDPOINTS)

    # Compute NWEIGHT-weighted income quintiles
    recs = recs.sort_values('income_midpoint').reset_index(drop=True)
    cum_wt = recs['NWEIGHT'].cumsum()
    total_wt = recs['NWEIGHT'].sum()
    breaks = [total_wt * q / 5 for q in range(1, 5)]
    labels = []
    bi = 0
    for _, row in recs.iterrows():
        while bi < len(breaks) and cum_wt.loc[row.name] > breaks[bi]:
            bi += 1
        labels.append(f'Q{bi + 1}')
    recs['income_quintile'] = pd.Categorical(labels, categories=QUINTILE_KEYS, ordered=True)

    logger.info(f"  RECS quintile distribution:\n{recs['income_quintile'].value_counts().sort_index()}")
    logger.info(f"  EQUIPAGE valid: {(recs['EQUIPAGE'] > 0).sum():,}  N/A: {(recs['EQUIPAGE'] <= 0).sum():,}")
    logger.info(f"  YEARMADERANGE valid: {(recs['YEARMADERANGE'] > 0).sum():,}")
    return recs


def load_matched_equipment_data(
    manifest_path: str,
    recs_path: str,
    max_shards: int = 0,
) -> pd.DataFrame:
    """Stream Phase 2 shards, join to raw RECS for equipment age."""
    logger.info("Building RECS equipment lookup ...")
    equip_lookup = pd.read_csv(
        recs_path,
        usecols=['DOEID', 'EQUIPAGE', 'ACEQUIPAGE', 'YEARMADERANGE'],
        low_memory=False,
    ).set_index('DOEID')

    logger.info(f"Loading Phase 2 shard manifest from {manifest_path}")
    manifest = json.load(open(manifest_path))
    files = manifest['files']
    if max_shards > 0:
        files = files[:max_shards]
    logger.info(f"  Processing {len(files)} / {len(manifest['files'])} shards")

    parts = []
    for i, fp in enumerate(files):
        shard = pd.read_pickle(fp)
        sub = shard[['recs_template_id', 'income_quintile']].copy()
        sub['DOEID'] = (
            sub['recs_template_id']
            .str.replace('RECS_', '', regex=False)
            .astype(int)
        )
        merged = sub.merge(equip_lookup, left_on='DOEID', right_index=True, how='left')
        parts.append(merged[['income_quintile', 'EQUIPAGE', 'ACEQUIPAGE', 'YEARMADERANGE']])
        del shard
        if (i + 1) % 100 == 0:
            logger.info(f"  ... processed {i + 1}/{len(files)} shards")

    matched = pd.concat(parts, ignore_index=True)
    matched['income_quintile'] = pd.Categorical(
        matched['income_quintile'].astype(str), categories=QUINTILE_KEYS, ordered=True,
    )
    logger.info(f"  Matched buildings: {len(matched):,}")
    logger.info(f"  Matched quintile distribution:\n{matched['income_quintile'].value_counts().sort_index()}")
    return matched


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def cramers_v(x, y):
    """Compute Cramer's V for two categorical arrays."""
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct + 0.5)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def compute_consistency_tests(recs: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """Per-quintile KS tests + global Cramer's V for EQUIPAGE and YEARMADERANGE."""
    results = {}
    for var in ['EQUIPAGE', 'YEARMADERANGE']:
        results[var] = {'per_quintile': {}, 'global': {}}
        recs_valid = recs[recs[var] > 0]
        matched_valid = matched[matched[var] > 0]

        # Per-quintile tests
        for q in QUINTILE_KEYS:
            r_vals = recs_valid.loc[recs_valid['income_quintile'] == q, var].values
            m_vals = matched_valid.loc[matched_valid['income_quintile'] == q, var].values

            if len(r_vals) < 5 or len(m_vals) < 5:
                results[var]['per_quintile'][q] = {'note': 'insufficient data'}
                continue

            ks_stat, ks_p = ks_2samp(r_vals, m_vals)

            results[var]['per_quintile'][q] = {
                'ks_statistic': round(float(ks_stat), 4),
                'ks_pvalue': round(float(ks_p), 6),
                'recs_mean': round(float(r_vals.mean()), 3),
                'matched_mean': round(float(m_vals.mean()), 3),
                'mean_diff': round(float(m_vals.mean() - r_vals.mean()), 3),
                'recs_n': int(len(r_vals)),
                'matched_n': int(len(m_vals)),
            }

        # Global: Cramer's V between dataset source and variable categories
        combined = pd.concat([
            recs_valid[['income_quintile', var]].assign(source='RECS'),
            matched_valid[['income_quintile', var]].assign(source='Matched'),
        ], ignore_index=True)
        cv = cramers_v(combined['source'], combined[var])
        results[var]['global']['cramers_v'] = round(float(cv), 4)

        # Gradient check: Spearman correlation of quintile rank vs mean age
        recs_means = [recs_valid.loc[recs_valid['income_quintile'] == q, var].mean()
                      for q in QUINTILE_KEYS]
        matched_means = [matched_valid.loc[matched_valid['income_quintile'] == q, var].mean()
                         for q in QUINTILE_KEYS]
        recs_rho, _ = spearmanr(range(5), recs_means)
        matched_rho, _ = spearmanr(range(5), matched_means)
        results[var]['global']['recs_gradient_rho'] = round(float(recs_rho), 4)
        results[var]['global']['matched_gradient_rho'] = round(float(matched_rho), 4)
        results[var]['global']['gradient_preserved'] = bool(
            np.sign(recs_rho) == np.sign(matched_rho)
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _weighted_conditional_distribution(
    df: pd.DataFrame, var: str, weight_col: str = None,
) -> pd.DataFrame:
    """Compute P(var_category | income_quintile), optionally weighted."""
    valid = df[df[var] > 0].copy()
    if weight_col and weight_col in valid.columns:
        # Weighted crosstab
        rows = []
        for q in QUINTILE_KEYS:
            qdf = valid[valid['income_quintile'] == q]
            cats = sorted(valid[var].unique())
            weighted_counts = {}
            for c in cats:
                weighted_counts[c] = qdf.loc[qdf[var] == c, weight_col].sum()
            total = sum(weighted_counts.values())
            if total > 0:
                rows.append({c: 100 * v / total for c, v in weighted_counts.items()})
            else:
                rows.append({c: 0.0 for c in cats})
        ct = pd.DataFrame(rows, index=QUINTILE_KEYS)
    else:
        ct = pd.crosstab(valid['income_quintile'], valid[var], normalize='index') * 100
    return ct


def create_figure(
    recs: pd.DataFrame,
    matched: pd.DataFrame,
    stats: dict,
    output_dir: Path,
):
    """Create 2x2 bivariate distribution comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), constrained_layout=True)

    variables = [
        ('EQUIPAGE', EQUIPAGE_LABELS, 'Heating Equipment Age'),
        ('YEARMADERANGE', YEARMADERANGE_LABELS, 'Building Vintage'),
    ]
    datasets = [
        (recs, 'Raw RECS 2020 (survey-weighted)', 'NWEIGHT'),
        (matched, 'Matched Synthetic Population', None),
    ]

    for col_idx, (var, label_map, var_title) in enumerate(variables):
        for row_idx, (data, ds_title, wt_col) in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            panel_letter = chr(97 + row_idx * 2 + col_idx)

            ct = _weighted_conditional_distribution(data, var, wt_col)
            cats = sorted(label_map.keys())
            for c in cats:
                if c not in ct.columns:
                    ct[c] = 0.0
            ct = ct[cats]

            x = np.arange(len(QUINTILE_KEYS))
            width = 0.8
            bottom = np.zeros(len(QUINTILE_KEYS))

            # Colour scheme: warm for older → cool for newer
            cmap = plt.cm.RdYlGn if col_idx == 0 else plt.cm.RdYlBu
            n_cats = len(cats)
            colors = [cmap(0.1 + 0.8 * i / (n_cats - 1)) for i in range(n_cats)]

            for ci, cat in enumerate(cats):
                vals = ct[cat].reindex(QUINTILE_KEYS, fill_value=0).values
                ax.bar(x, vals, width, bottom=bottom, color=colors[ci],
                       edgecolor='white', linewidth=0.4,
                       label=label_map[cat])
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels(QUINTILE_LABELS, fontsize=7)
            ax.set_ylabel('Share (%)', fontsize=8)
            ax.set_ylim(0, 108)
            ax.set_title(f'({panel_letter}) {ds_title}', fontsize=8.5, fontweight='bold')

            # Legend: show in top-row panels
            if row_idx == 0:
                ncol = 2 if col_idx == 0 else 3
                ax.legend(fontsize=5.2, title=var_title, title_fontsize=6,
                          loc='upper right', ncol=ncol, framealpha=0.88)

            # Annotate max KS statistic (effect size, not p-value)
            if var in stats:
                ks_vals = []
                for q in QUINTILE_KEYS:
                    entry = stats[var]['per_quintile'].get(q, {})
                    ks = entry.get('ks_statistic', None)
                    if ks is not None:
                        ks_vals.append(ks)
                if ks_vals:
                    max_ks = max(ks_vals)
                    txt = f'max KS = {max_ks:.3f}'
                    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                            fontsize=6.5, va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = output_dir / f'reviewer6_income_equipment_age.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(stats: dict):
    """Print a formatted summary of statistical tests."""
    for var in ['EQUIPAGE', 'YEARMADERANGE']:
        logger.info(f"\n{'='*70}")
        logger.info(f"  {var} -- Per-quintile consistency tests")
        logger.info(f"{'='*70}")
        header = (f"{'Quintile':<10} {'RECS mean':>10} {'Match mean':>11} "
                  f"{'Diff':>7} {'KS stat':>8} {'KS p':>10} "
                  f"{'n_RECS':>8} {'n_Match':>9}")
        logger.info(header)
        logger.info('-' * len(header))
        for q in QUINTILE_KEYS:
            e = stats[var]['per_quintile'].get(q, {})
            if 'note' in e:
                logger.info(f"{q:<10} {e['note']}")
                continue
            logger.info(
                f"{q:<10} {e['recs_mean']:>10.3f} {e['matched_mean']:>11.3f} "
                f"{e['mean_diff']:>+7.3f} {e['ks_statistic']:>8.4f} "
                f"{e['ks_pvalue']:>10.6f} {e['recs_n']:>8,} {e['matched_n']:>9,}"
            )
        g = stats[var]['global']
        logger.info(f"\n  Cramer's V (RECS vs Matched): {g['cramers_v']:.4f}")
        logger.info(f"  Gradient rho  -- RECS: {g['recs_gradient_rho']:+.4f}  "
                     f"Matched: {g['matched_gradient_rho']:+.4f}  "
                     f"Preserved: {g['gradient_preserved']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Reviewer 6: income vs equipment age validation')
    parser.add_argument('--output-dir', default='results/reviewer6',
                        help='Output directory')
    parser.add_argument('--max-shards', type=int, default=0,
                        help='Limit shards (0=all)')
    parser.add_argument('--recs-path',
                        default='data/raw/recs/2020/recs2020_public_v7.csv')
    parser.add_argument('--manifest-path',
                        default='data/processed/phase2_shards/manifest.json')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  Reviewer 6: Income vs Equipment Age Validation")
    logger.info("=" * 60)

    # Step 1: Load raw RECS
    recs = load_raw_recs(args.recs_path)

    # Step 2: Load matched data with equipment age join
    matched = load_matched_equipment_data(
        args.manifest_path, args.recs_path, args.max_shards)

    # Step 3: Statistical tests
    logger.info("\nComputing statistical consistency tests ...")
    stats = compute_consistency_tests(recs, matched)
    print_summary(stats)

    # Step 4: Create figure
    logger.info("\nCreating 2x2 validation figure ...")
    create_figure(recs, matched, stats, output_dir)

    # Step 5: Copy figure to paper directory
    paper_dir = Path('paper')
    if paper_dir.is_dir():
        import shutil
        for ext in ['png', 'pdf']:
            src = output_dir / f'reviewer6_income_equipment_age.{ext}'
            dst = paper_dir / f'reviewer6_income_equipment_age.{ext}'
            shutil.copy2(src, dst)
            logger.info(f"  Copied to {dst}")

    # Step 6: Save JSON results
    results_path = output_dir / 'reviewer6_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'max_shards': args.max_shards,
            'recs_records': len(recs),
            'matched_records': len(matched),
            'statistical_tests': stats,
        }, f, indent=2)
    logger.info(f"  Saved results to {results_path}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
