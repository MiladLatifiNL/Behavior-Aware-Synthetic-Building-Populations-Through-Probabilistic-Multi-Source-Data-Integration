"""
Aggregate benchmark validation for the synthetic building population.

Compares RECS-linked energy attributes in the synthetic population against
published RECS 2020 national benchmarks (weighted survey estimates) and
DOE LEAD energy burden data.

Produces a CSV table and formatted LaTeX table suitable for the manuscript.

Usage:
    python -m src.validation.benchmark_validation
    python -m src.validation.benchmark_validation --phase2-shards data/processed/phase2_shards
    python -m src.validation.benchmark_validation --output-dir results/benchmark_validation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RECS 2020 benchmark values (weighted survey estimates from microdata)
# Source: EIA RECS 2020, Table CE1.1 / weighted computation from microdata
# ---------------------------------------------------------------------------
RECS_BENCHMARKS = {
    'mean_electricity_kwh':     10_566,    # kWh/yr (weighted mean from RECS microdata)
    'mean_site_eui_kbtu_sqft':  51.0,      # kBtu/ft2 (weighted: total_kBtu / sqft)
    'mean_total_energy_cost':   1_884,     # $/yr (weighted sum of all fuel costs)
    'mean_floor_area_sqft':     1_619,     # sqft (weighted mean SQFTEST)
    'share_single_family_pct':  68.4,      # % (TYPEHUQ 2+3, weighted)
    'share_mobile_home_pct':    5.5,       # % (TYPEHUQ 1, weighted)
    'share_burden_above_6_pct': 25.0,      # % (DOE LEAD Tool estimate)
    'share_burden_above_10_pct': 13.0,     # % (DOE LEAD Tool estimate)
}


def compute_recs_weighted_benchmarks(recs_path: str) -> dict:
    """Compute weighted national benchmarks directly from RECS 2020 microdata.

    Args:
        recs_path: Path to recs2020_public_v7.csv

    Returns:
        Dict of benchmark metric name -> value
    """
    logger.info(f"Computing RECS weighted benchmarks from {recs_path}")
    recs = pd.read_csv(recs_path, low_memory=False)
    w = recs['NWEIGHT']
    total_w = w.sum()

    # Mean electricity (kWh/yr)
    mean_kwh = (recs['KWH'] * w).sum() / total_w

    # Total site energy in kBtu (BTUEL + BTUNG + BTULP + BTUFO are in thousand BTU)
    total_kbtu = (
        recs['BTUEL'].fillna(0) + recs['BTUNG'].fillna(0) +
        recs['BTULP'].fillna(0) + recs['BTUFO'].fillna(0)
    )
    # Site EUI (kBtu / sqft)
    eui = total_kbtu / recs['SQFTEST'].replace(0, np.nan)
    valid_eui = eui.notna()
    mean_eui = (eui[valid_eui] * w[valid_eui]).sum() / w[valid_eui].sum()

    # Total energy cost
    total_cost = (
        recs['DOLLAREL'].fillna(0) + recs['DOLLARNG'].fillna(0) +
        recs['DOLLARLP'].fillna(0) + recs['DOLLARFO'].fillna(0)
    )
    mean_cost = (total_cost * w).sum() / total_w

    # Mean floor area
    mean_sqft = (recs['SQFTEST'] * w).sum() / total_w

    # Building type shares (TYPEHUQ: 1=Mobile, 2=SF Det, 3=SF Att, 4=Apt2-4, 5=Apt5+)
    sf_share = recs.loc[recs['TYPEHUQ'].isin([2, 3]), 'NWEIGHT'].sum() / total_w * 100
    mobile_share = recs.loc[recs['TYPEHUQ'] == 1, 'NWEIGHT'].sum() / total_w * 100

    # Heating fuel shares (excluding not-applicable FUELHEAT=-2)
    heated = recs[recs['FUELHEAT'] > 0]
    hw_total = heated['NWEIGHT'].sum()
    gas_share = heated.loc[heated['FUELHEAT'].isin([1, 2]), 'NWEIGHT'].sum() / hw_total * 100
    elec_share = heated.loc[heated['FUELHEAT'] == 5, 'NWEIGHT'].sum() / hw_total * 100
    oil_share = heated.loc[heated['FUELHEAT'] == 3, 'NWEIGHT'].sum() / hw_total * 100

    benchmarks = {
        'mean_electricity_kwh': round(mean_kwh, 0),
        'mean_site_eui_kbtu_sqft': round(mean_eui, 1),
        'mean_total_energy_cost': round(mean_cost, 0),
        'mean_floor_area_sqft': round(mean_sqft, 0),
        'share_single_family_pct': round(sf_share, 1),
        'share_mobile_home_pct': round(mobile_share, 1),
        'share_gas_heating_pct': round(gas_share, 1),
        'share_electric_heating_pct': round(elec_share, 1),
        'share_fuel_oil_heating_pct': round(oil_share, 1),
        # Energy burden benchmarks from DOE LEAD (not derivable from RECS alone)
        'share_burden_above_6_pct': RECS_BENCHMARKS['share_burden_above_6_pct'],
        'share_burden_above_10_pct': RECS_BENCHMARKS['share_burden_above_10_pct'],
        'n_records': len(recs),
        'total_weighted_households': round(total_w, 0),
    }

    logger.info(f"RECS benchmarks computed from {len(recs)} records "
                f"({total_w:,.0f} weighted households)")
    return benchmarks


def compute_synthetic_statistics(shards_dir: str) -> dict:
    """Stream over Phase 2 shards and compute aggregate statistics.

    Args:
        shards_dir: Path to phase2_shards directory containing manifest.json

    Returns:
        Dict of synthetic population statistics
    """
    manifest_path = Path(shards_dir) / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    files = manifest['files']
    n_shards = len(files)
    logger.info(f"Streaming over {n_shards} shards from {shards_dir}")

    # Accumulators
    total_buildings = 0
    sum_kwh = 0.0
    sum_eui = 0.0
    sum_cost = 0.0
    sum_sqft = 0.0
    n_kwh = 0
    n_eui = 0
    n_cost = 0
    n_sqft = 0

    building_type_counts = {}
    heating_fuel_counts = {}
    tenure_counts = {}

    n_burden = 0
    burden_above_6 = 0
    burden_above_10 = 0
    burden_above_20 = 0

    for i, filepath in enumerate(files):
        if i % 100 == 0:
            logger.info(f"  Processing shard {i}/{n_shards}...")

        try:
            df = pd.read_pickle(filepath)
        except Exception as e:
            logger.warning(f"  Error reading {filepath}: {e}")
            continue

        total_buildings += len(df)

        # Electricity kWh (from RECS template)
        if 'recs_electricity_kwh' in df.columns:
            vals = pd.to_numeric(df['recs_electricity_kwh'], errors='coerce').dropna()
            sum_kwh += vals.sum()
            n_kwh += len(vals)

        # Site EUI (from RECS template)
        if 'recs_energy_use_intensity' in df.columns:
            vals = pd.to_numeric(df['recs_energy_use_intensity'], errors='coerce').dropna()
            sum_eui += vals.sum()
            n_eui += len(vals)

        # Total energy cost (from RECS template)
        if 'recs_total_energy_cost_annual' in df.columns:
            vals = pd.to_numeric(df['recs_total_energy_cost_annual'], errors='coerce').dropna()
            sum_cost += vals.sum()
            n_cost += len(vals)

        # Square footage (from RECS template)
        if 'recs_square_footage' in df.columns:
            vals = pd.to_numeric(df['recs_square_footage'], errors='coerce').dropna()
            sum_sqft += vals.sum()
            n_sqft += len(vals)

        # Building type (from PUMS)
        if 'building_type_simple' in df.columns:
            for val in df['building_type_simple'].dropna():
                k = str(val).lower()
                building_type_counts[k] = building_type_counts.get(k, 0) + 1

        # Heating fuel (from PUMS)
        if 'heating_fuel' in df.columns:
            for val in df['heating_fuel'].dropna():
                k = str(val).lower()
                heating_fuel_counts[k] = heating_fuel_counts.get(k, 0) + 1

        # Tenure (from PUMS)
        if 'tenure_type' in df.columns:
            for val in df['tenure_type'].dropna():
                k = str(val).lower()
                tenure_counts[k] = tenure_counts.get(k, 0) + 1

        # Energy burden
        if 'energy_burden' in df.columns:
            burden = pd.to_numeric(df['energy_burden'], errors='coerce').dropna()
            n_burden += len(burden)
            burden_above_6 += int((burden > 6).sum())
            burden_above_10 += int((burden > 10).sum())
            burden_above_20 += int((burden > 20).sum())

    # Compute means
    mean_kwh = sum_kwh / n_kwh if n_kwh > 0 else np.nan
    mean_eui = sum_eui / n_eui if n_eui > 0 else np.nan
    mean_cost = sum_cost / n_cost if n_cost > 0 else np.nan
    mean_sqft = sum_sqft / n_sqft if n_sqft > 0 else np.nan

    # Building type shares
    total_bt = sum(building_type_counts.values())
    sf_share = building_type_counts.get('single_family', 0) / total_bt * 100 if total_bt > 0 else 0
    mobile_share = building_type_counts.get('mobile', 0) / total_bt * 100 if total_bt > 0 else 0

    # Heating fuel shares
    total_fuel = sum(heating_fuel_counts.values())
    gas_share = heating_fuel_counts.get('gas', 0) / total_fuel * 100 if total_fuel > 0 else 0
    elec_share = heating_fuel_counts.get('electricity', 0) / total_fuel * 100 if total_fuel > 0 else 0
    oil_share = heating_fuel_counts.get('fuel_oil', 0) / total_fuel * 100 if total_fuel > 0 else 0

    # Burden shares
    burden_6_pct = burden_above_6 / n_burden * 100 if n_burden > 0 else 0
    burden_10_pct = burden_above_10 / n_burden * 100 if n_burden > 0 else 0
    burden_20_pct = burden_above_20 / n_burden * 100 if n_burden > 0 else 0

    stats = {
        'total_buildings': total_buildings,
        'mean_electricity_kwh': round(mean_kwh, 0),
        'mean_site_eui_kbtu_sqft': round(mean_eui, 1),
        'mean_total_energy_cost': round(mean_cost, 0),
        'mean_floor_area_sqft': round(mean_sqft, 0),
        'share_single_family_pct': round(sf_share, 1),
        'share_mobile_home_pct': round(mobile_share, 1),
        'share_gas_heating_pct': round(gas_share, 1),
        'share_electric_heating_pct': round(elec_share, 1),
        'share_fuel_oil_heating_pct': round(oil_share, 1),
        'share_burden_above_6_pct': round(burden_6_pct, 1),
        'share_burden_above_10_pct': round(burden_10_pct, 1),
        'share_burden_above_20_pct': round(burden_20_pct, 1),
        'n_burden_valid': n_burden,
        'building_type_counts': building_type_counts,
        'heating_fuel_counts': heating_fuel_counts,
        'tenure_counts': tenure_counts,
    }

    logger.info(f"Synthetic statistics computed from {total_buildings:,} buildings")
    return stats


def compute_deviations(synthetic: dict, benchmark: dict) -> list:
    """Compute deviations between synthetic and benchmark values.

    Returns:
        List of dicts with metric, synthetic, benchmark, deviation, unit info
    """
    rows = [
        {
            'metric': 'Mean electricity (kWh/yr)',
            'synthetic': f"{synthetic['mean_electricity_kwh']:,.0f}",
            'benchmark': f"{benchmark['mean_electricity_kwh']:,.0f}",
            'deviation': _pct_dev(synthetic['mean_electricity_kwh'],
                                  benchmark['mean_electricity_kwh']),
            'unit': '%',
        },
        {
            'metric': 'Mean site EUI (kBtu/sqft)',
            'synthetic': f"{synthetic['mean_site_eui_kbtu_sqft']:.1f}",
            'benchmark': f"{benchmark['mean_site_eui_kbtu_sqft']:.1f}",
            'deviation': _pct_dev(synthetic['mean_site_eui_kbtu_sqft'],
                                  benchmark['mean_site_eui_kbtu_sqft']),
            'unit': '%',
        },
        {
            'metric': 'Mean total energy cost ($/yr)',
            'synthetic': f"{synthetic['mean_total_energy_cost']:,.0f}",
            'benchmark': f"{benchmark['mean_total_energy_cost']:,.0f}",
            'deviation': _pct_dev(synthetic['mean_total_energy_cost'],
                                  benchmark['mean_total_energy_cost']),
            'unit': '%',
        },
        {
            'metric': 'Mean floor area (sqft)',
            'synthetic': f"{synthetic['mean_floor_area_sqft']:,.0f}",
            'benchmark': f"{benchmark['mean_floor_area_sqft']:,.0f}",
            'deviation': _pct_dev(synthetic['mean_floor_area_sqft'],
                                  benchmark['mean_floor_area_sqft']),
            'unit': '%',
        },
        {
            'metric': 'Single-family share (%)',
            'synthetic': f"{synthetic['share_single_family_pct']:.1f}",
            'benchmark': f"{benchmark['share_single_family_pct']:.1f}",
            'deviation': _pp_dev(synthetic['share_single_family_pct'],
                                 benchmark['share_single_family_pct']),
            'unit': 'pp',
        },
        {
            'metric': 'Mobile home share (%)',
            'synthetic': f"{synthetic['share_mobile_home_pct']:.1f}",
            'benchmark': f"{benchmark['share_mobile_home_pct']:.1f}",
            'deviation': _pp_dev(synthetic['share_mobile_home_pct'],
                                 benchmark['share_mobile_home_pct']),
            'unit': 'pp',
        },
        {
            'metric': 'Energy burden >6% (%)',
            'synthetic': f"{synthetic['share_burden_above_6_pct']:.1f}",
            'benchmark': f"~{benchmark['share_burden_above_6_pct']:.0f}",
            'deviation': _pp_dev(synthetic['share_burden_above_6_pct'],
                                 benchmark['share_burden_above_6_pct']),
            'unit': 'pp',
        },
        {
            'metric': 'Energy burden >10% (%)',
            'synthetic': f"{synthetic['share_burden_above_10_pct']:.1f}",
            'benchmark': f"~{benchmark['share_burden_above_10_pct']:.0f}",
            'deviation': _pp_dev(synthetic['share_burden_above_10_pct'],
                                 benchmark['share_burden_above_10_pct']),
            'unit': 'pp',
        },
    ]
    return rows


def _pct_dev(synthetic_val, benchmark_val) -> str:
    """Format percentage deviation."""
    if benchmark_val == 0:
        return 'N/A'
    dev = (synthetic_val - benchmark_val) / benchmark_val * 100
    return f"{dev:+.1f}%"


def _pp_dev(synthetic_val, benchmark_val) -> str:
    """Format percentage-point deviation."""
    dev = synthetic_val - benchmark_val
    return f"{dev:+.1f} pp"


def generate_csv(rows: list, output_path: str):
    """Save comparison table as CSV."""
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV table saved to {output_path}")


def generate_latex_table(rows: list, synthetic_stats: dict, output_path: str):
    """Generate a LaTeX table for the manuscript."""
    n_buildings = synthetic_stats['total_buildings']

    lines = []
    lines.append(r"% Auto-generated by src/validation/benchmark_validation.py")
    lines.append(r"% Paste into manuscript Section 5.3")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Aggregate energy attributes of the synthetic population")
    lines.append(f"  compared with RECS~2020 national benchmarks (weighted survey")
    lines.append(f"  estimates). Synthetic population values are computed from")
    lines.append(f"  RECS template-level attributes inherited through probabilistic")
    lines.append(f"  matching across {n_buildings:,} buildings.}}")
    lines.append(r"  \label{tab:recs-benchmark}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Metric}")
    lines.append(r"      & \textbf{Synthetic Pop.}")
    lines.append(r"      & \textbf{RECS~2020\textsuperscript{a}}")
    lines.append(r"      & \textbf{Deviation} \\")
    lines.append(r"    \midrule")

    for row in rows:
        # LaTeX-safe metric name
        metric_tex = row['metric'].replace('$', r'\$').replace('%', r'\%')
        metric_tex = metric_tex.replace('sqft', r'ft\textsuperscript{2}')
        metric_tex = metric_tex.replace('~', r'$\approx$')

        # Format synthetic/benchmark with thousand separators for LaTeX
        syn_tex = row['synthetic'].replace(',', '{,}')
        bench_tex = row['benchmark'].replace(',', '{,}').replace('~', r'$\approx$')

        # Format deviation
        dev_tex = row['deviation'].replace('%', r'\%')
        if row['unit'] == 'pp':
            dev_tex = f"${dev_tex.replace(' pp', '')}$~pp"
        else:
            dev_tex = f"${dev_tex}$"

        lines.append(f"    {metric_tex}")
        lines.append(f"      & {syn_tex} & {bench_tex} & {dev_tex} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"")
    lines.append(r"  \vspace{4pt}")
    lines.append(r"  \begin{flushleft}")
    lines.append(r"  \footnotesize")
    lines.append(r"  \textsuperscript{a}~Weighted estimates computed from RECS~2020")
    lines.append(r"  microdata ($n=18{,}496$; 123.5 million weighted")
    lines.append(r"  households)~\cite{RECS,EIA_CE11}. EUI computed as total site")
    lines.append(r"  energy (electricity + natural gas + propane + fuel oil, in kBtu)")
    lines.append(r"  divided by heated floor area.\\[2pt]")
    lines.append(r"  Energy burden benchmarks from U.S.\ Department of Energy,")
    lines.append(r"  Low-Income Energy Affordability Data (LEAD)")
    lines.append(r"  Tool~\cite{DOE_LEAD}. Burden defined as annual energy")
    lines.append(r"  expenditure divided by gross household income.\\[2pt]")
    lines.append(r"  \textit{Note:} ``pp'' denotes percentage points.")
    lines.append(r"  \end{flushleft}")
    lines.append(r"\end{table}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX table saved to {output_path}")


def generate_detail_report(synthetic_stats: dict, benchmark: dict, output_path: str):
    """Generate a detailed markdown report with all distributions."""
    lines = []
    lines.append("# Benchmark Validation Report")
    lines.append("")
    lines.append(f"**Synthetic population**: {synthetic_stats['total_buildings']:,} buildings")
    lines.append(f"**Burden-valid households**: {synthetic_stats['n_burden_valid']:,}")
    lines.append("")

    lines.append("## Building Type Distribution")
    lines.append("")
    lines.append("| Type | Count | Share |")
    lines.append("|------|-------|-------|")
    bt = synthetic_stats['building_type_counts']
    total_bt = sum(bt.values())
    for k, v in sorted(bt.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:,} | {v/total_bt*100:.1f}% |")

    lines.append("")
    lines.append("## Heating Fuel Distribution")
    lines.append("")
    lines.append("| Fuel | Count | Share |")
    lines.append("|------|-------|-------|")
    hf = synthetic_stats['heating_fuel_counts']
    total_hf = sum(hf.values())
    for k, v in sorted(hf.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:,} | {v/total_hf*100:.1f}% |")

    lines.append("")
    lines.append("## Tenure Distribution")
    lines.append("")
    lines.append("| Tenure | Count | Share |")
    lines.append("|--------|-------|-------|")
    ten = synthetic_stats['tenure_counts']
    total_ten = sum(ten.values())
    for k, v in sorted(ten.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:,} | {v/total_ten*100:.1f}% |")

    lines.append("")
    lines.append("## Energy Burden Distribution")
    lines.append("")
    lines.append(f"- Burden > 6%: {synthetic_stats['share_burden_above_6_pct']:.1f}% "
                 f"(benchmark: ~{benchmark['share_burden_above_6_pct']:.0f}%)")
    lines.append(f"- Burden > 10%: {synthetic_stats['share_burden_above_10_pct']:.1f}% "
                 f"(benchmark: ~{benchmark['share_burden_above_10_pct']:.0f}%)")
    lines.append(f"- Burden > 20%: {synthetic_stats['share_burden_above_20_pct']:.1f}%")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Detail report saved to {output_path}")


def run_benchmark_validation(
    phase2_shards_dir: str = 'data/processed/phase2_shards',
    recs_path: str = 'data/raw/recs/2020/recs2020_public_v7.csv',
    output_dir: str = 'results/benchmark_validation',
):
    """Run the full benchmark validation pipeline.

    Args:
        phase2_shards_dir: Path to phase2 shards directory
        recs_path: Path to RECS 2020 CSV
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute RECS benchmarks from source data (or use defaults)
    if Path(recs_path).exists():
        logger.info("Computing RECS benchmarks from source microdata...")
        benchmark = compute_recs_weighted_benchmarks(recs_path)
    else:
        logger.warning(f"RECS file not found at {recs_path}, using published defaults")
        benchmark = RECS_BENCHMARKS.copy()

    # Step 2: Stream synthetic population statistics
    logger.info("Computing synthetic population statistics from shards...")
    synthetic = compute_synthetic_statistics(phase2_shards_dir)

    # Step 3: Compute deviations
    rows = compute_deviations(synthetic, benchmark)

    # Step 4: Generate outputs
    generate_csv(rows, str(output_path / 'benchmark_comparison.csv'))
    generate_latex_table(rows, synthetic, str(output_path / 'benchmark_table.tex'))
    generate_detail_report(synthetic, benchmark, str(output_path / 'benchmark_report.md'))

    # Step 5: Save raw stats as JSON
    json_stats = {
        'synthetic': {k: v for k, v in synthetic.items()
                      if not isinstance(v, dict)},
        'benchmark': benchmark,
    }
    with open(output_path / 'benchmark_stats.json', 'w') as f:
        json.dump(json_stats, f, indent=2, default=str)
    logger.info(f"Raw stats saved to {output_path / 'benchmark_stats.json'}")

    # Step 6: Print summary
    print("\n" + "=" * 70)
    print(f"BENCHMARK VALIDATION ({synthetic['total_buildings']:,} buildings)")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'Synthetic':>12} {'RECS 2020':>12} {'Deviation':>12}")
    print("-" * 73)
    for row in rows:
        print(f"{row['metric']:<35} {row['synthetic']:>12} "
              f"{row['benchmark']:>12} {row['deviation']:>12}")
    print("-" * 73)
    print(f"\nOutputs saved to: {output_path}/")
    print(f"  - benchmark_comparison.csv")
    print(f"  - benchmark_table.tex  (copy into manuscript)")
    print(f"  - benchmark_report.md  (detailed distributions)")
    print(f"  - benchmark_stats.json (raw numbers)")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark validation: compare synthetic population against RECS 2020'
    )
    parser.add_argument(
        '--phase2-shards',
        default='data/processed/phase2_shards',
        help='Path to phase2 shards directory (default: data/processed/phase2_shards)'
    )
    parser.add_argument(
        '--recs-path',
        default='data/raw/recs/2020/recs2020_public_v7.csv',
        help='Path to RECS 2020 CSV (default: data/raw/recs/2020/recs2020_public_v7.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='results/benchmark_validation',
        help='Output directory (default: results/benchmark_validation)'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    run_benchmark_validation(
        phase2_shards_dir=args.phase2_shards,
        recs_path=args.recs_path,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
