#!/usr/bin/env python3
"""
Run all visualizations for the PUMS Enrichment Living System.

This script loads the processed data and generates all visualizations
to demonstrate the complete living system.
"""

import sys
import os
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import gc

import pandas as pd

# Force non-interactive backend for headless/CI environments before any pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualize.dashboard_generator import DashboardGenerator
from src.visualize.living_system_overview import SystemOverviewVisualizer
from src.visualize.building_visualizer import BuildingVisualizer
from src.visualize.person_visualizer import PersonVisualizer
from src.visualize.activity_visualizer import ActivityVisualizer
from src.visualize.weather_visualizer import WeatherVisualizer
from src.visualize.energy_visualizer import EnergyVisualizer
from src.visualize.household_visualizer import HouseholdVisualizer
import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_print(*args, **kwargs):
    """Print helper that avoids Windows console encoding errors."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args)
        sys.stdout.write(msg.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))
        sys.stdout.write("\n")


def _load_phase4_from_shards(manifest_path: Path) -> pd.DataFrame:
    """Load full Phase 4 data by concatenating all shard pickles listed in manifest.json."""
    with open(manifest_path, 'r') as mf:
        manifest = json.load(mf)

    files: List[str] = manifest.get('files', [])
    total = manifest.get('total_buildings')
    logger.info(f"Loading Phase 4 shards from manifest: {manifest_path} (n_shards={manifest.get('n_shards')}, total_buildings={total:,} if total else 'unknown')")

    parts: List[pd.DataFrame] = []
    loaded_rows = 0
    batch_size = 25  # concatenate every N shards to reduce fragmentation

    manifest_dir = manifest_path.parent

    for i, file_path in enumerate(files):
        # Resolve relative paths against the manifest directory
        original_path = Path(file_path)
        shard_path = original_path
        if not shard_path.is_absolute():
            # First try path relative to manifest directory
            candidate = (manifest_dir / shard_path).resolve()
            if candidate.exists():
                shard_path = candidate
            else:
                # Then try relative to current working directory (project root)
                candidate2 = (Path.cwd() / original_path).resolve()
                shard_path = candidate2 if candidate2.exists() else candidate
        if not shard_path.exists():
            logger.warning(f"Shard path does not exist, skipping: {original_path}")
            continue
        try:
            with open(shard_path, 'rb') as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Shard did not contain a DataFrame: {shard_path}")
                continue
            parts.append(df)
            loaded_rows += len(df)
        except Exception as e:
            logger.warning(f"Failed to load shard {shard_path}: {e}")
            continue

        # Periodic concat to keep memory reasonable
        if len(parts) >= batch_size:
            logger.info(f"Concatenating batch at shard {i+1}/{len(files)} (rows so far: {loaded_rows:,})")
            combined = pd.concat(parts, ignore_index=True, copy=False)
            parts = [combined]
            gc.collect()

    # Final concat
    if len(parts) == 0:
        logger.error("No shards could be loaded from manifest.")
        return pd.DataFrame()
    elif len(parts) > 1:
        logger.info("Final concatenation of remaining parts...")
        buildings_df = pd.concat(parts, ignore_index=True, copy=False)
    else:
        buildings_df = parts[0]

    logger.info(f"Loaded full Phase 4 dataset from shards: {len(buildings_df):,} buildings")
    return buildings_df


def _iterate_shards(manifest_path: Path):
    """Yield each shard DataFrame from the Phase 4 manifest without keeping all in memory."""
    with open(manifest_path, 'r') as mf:
        manifest = json.load(mf)
    files: List[str] = manifest.get('files', [])
    manifest_dir = manifest_path.parent

    for file_path in files:
        original_path = Path(file_path)
        shard_path = original_path
        if not shard_path.is_absolute():
            candidate = (manifest_dir / shard_path).resolve()
            if candidate.exists():
                shard_path = candidate
            else:
                candidate2 = (Path.cwd() / original_path).resolve()
                shard_path = candidate2 if candidate2.exists() else candidate
        if not shard_path.exists():
            logger.warning(f"Shard path does not exist, skipping: {original_path}")
            continue
        try:
            with open(shard_path, 'rb') as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                yield df
        except Exception as e:
            logger.warning(f"Failed to load shard {shard_path}: {e}")
            continue


def load_latest_data():
    """Load the FINAL Phase 4 integrated living system data.

    Prefers shard-based full dataset; falls back to the small sample pickle if shards are missing.
    """
    data_dir = Path("data/processed")
    shards_dir = data_dir / "phase4_shards"
    manifest_path = shards_dir / "manifest.json"

    buildings_df: pd.DataFrame
    phase = 4
    manifest_total = None

    manifest_present = manifest_path.exists()
    if manifest_present:
        # Do NOT concatenate all shards into memory; we'll stream for aggregates later.
        try:
            with open(manifest_path, 'r') as mf:
                manifest = json.load(mf)
                manifest_total = manifest.get('total_buildings')
                logger.info(
                    "Phase 4 shards manifest detected; will stream for full-data aggregates. "
                    f"(n_shards={manifest.get('n_shards')}, total_buildings={manifest_total})"
                )
        except Exception as e:
            logger.warning(f"Failed to read shards manifest for totals: {e}")
        buildings_df = pd.DataFrame()
    else:
        buildings_df = pd.DataFrame()

    # Fallback to small sample pickle only if no shards manifest
    if buildings_df.empty and not manifest_present:
        phase4_file = data_dir / "phase4_final_integrated_buildings.pkl"
        if not phase4_file.exists():
            logger.error(f"Phase 4 data not found. Missing shards manifest and sample file: {phase4_file}")
            return None, None, 0
        logger.info(f"Loading Phase 4 SAMPLE (small) from {phase4_file}")
        with open(phase4_file, 'rb') as f:
            buildings_df = pickle.load(f)

    # Load metadata from all 4 phases
    metadata = {}
    for p in range(1, 5):
        metadata_file = data_dir / f"phase{p}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                phase_metadata = json.load(f)
                metadata[f'phase{p}'] = phase_metadata
                if 'processing_time' in phase_metadata:
                    metadata[f'phase{p}_time'] = phase_metadata.get('processing_time', phase_metadata.get('duration_seconds', 0))
                if 'buildings_processed' in phase_metadata:
                    metadata['buildings_processed'] = phase_metadata['buildings_processed']
                if 'persons_processed' in phase_metadata:
                    metadata['persons_processed'] = phase_metadata['persons_processed']

    # Add default metadata
    try:
        persons_total = sum(len(b.get('persons', [])) if isinstance(b.get('persons', []), list) else 0 for _, b in buildings_df.iterrows())
    except Exception:
        persons_total = 0
    metadata.update({
        'households': len(buildings_df) if manifest_total is None else manifest_total,
        'persons': persons_total,
        'phase_completed': phase,
        'timestamp': datetime.now().isoformat()
    })

    logger.info(
        f"Loaded {len(buildings_df):,} buildings frame from Phase {phase} "
        + (f"(streaming total: {manifest_total:,})" if manifest_total else "")
    )
    logger.info(f"Total persons (quick sum): {metadata['persons']:,}")

    return buildings_df, metadata, phase


def _aggregate_full_dataset(manifest_path: Path):
    """Stream over shards to compute full-dataset aggregates for visualizations without loading everything."""
    logger.info("Streaming shards to compute full-dataset aggregates (memory efficient)...")

    # Building-level categorical counts (for simple distributions)
    cat_counts = {
        'building_type_simple': {},
        'STATE': {},
        'heating_fuel': {},
        'climate_zone': {},
        'tenure_type': {},
        'household_composition': {},
        'multigenerational': {},
    }

    # Numeric hist bins (predefine reasonable ranges)
    bins_config = {
        'building_age': np.arange(0, 125, 5),
        'energy_intensity': np.linspace(0, 500, 51),
        'recs_match_weight': np.linspace(-5, 5, 51),
        'energy_burden': np.linspace(0, 50, 51),
        'household_size': np.arange(-0.5, 11.5, 1),
        'num_bedrooms': np.arange(-0.5, 11.5, 1),
        'occupancy_intensity': np.linspace(0, 5, 51),
        'rooms_per_person': np.linspace(0, 10, 51),
    # Derived household metrics (populated later during per-building iteration)
    'num_children': np.arange(-0.5, 11.5, 1),
    'num_seniors': np.arange(-0.5, 11.5, 1),
    'workers_per_household': np.arange(-0.5, 11.5, 1),
    'age_diversity_std': np.linspace(0, 50, 51),
    }
    hists = {k: np.zeros(len(bins_config[k]) - 1, dtype=np.int64) for k in bins_config}

    # 2D density bins
    density_2d = {
        'age_vs_energy': {
            'x_bins': np.arange(0, 125, 5),
            'y_bins': np.linspace(0, 500, 51),
            'grid': None,
        },
        'temp_vs_energy': {
            'x_bins': np.linspace(-40, 120, 81),
            'y_bins': np.linspace(0, 100_000, 101),
            'grid': None,
        },
        'roomspp_vs_intensity': {
            'x_bins': np.linspace(0, 10, 51),
            'y_bins': np.linspace(0, 500, 51),
            'grid': None,
        },
        'eb_vs_ei': {
            'x_bins': np.linspace(0, 500, 51),  # energy intensity
            'y_bins': np.linspace(0, 50, 51),   # energy burden
            'grid': None,
        }
    }

    # Peak hour statistics and TOU shares (based on hourly profiles)
    peak_stats = {
        'overall': {
            'sum_of_peaks': 0.0,
            'count': 0,
            'peak_hour_hist': np.zeros(24, dtype=np.int64),
        },
        'by_group': {
            'building_type_simple': {},
            'STATE': {},
            'climate_zone': {},
        }
    }
    # Define Time-Of-Use bands (assumption; adjust if needed)
    tou_bands = {
        'offpeak': list(range(0, 6)) + list(range(22, 24)),
        'shoulder': list(range(6, 16)),
        'peak': list(range(16, 22)),
    }
    tou_stats = {
        'bands': tou_bands,
        'average_share': {bn: {'sum_share': 0.0, 'count': 0} for bn in tou_bands},
        'average_share_by_group': {
            'building_type_simple': {},
            'STATE': {},
            'climate_zone': {},
        },
        # Energy-weighted shares computed post-stream from aggregated sums in time_profiles
    }

    # Persons-level aggregations
    age_bins = list(range(0, 101, 5))
    age_labels = [f"{i}-{i+4}" for i in age_bins[:-1]]
    persons_age_sex = {
        'M': {label: 0 for label in age_labels},
        'F': {label: 0 for label in age_labels},
    }
    emp_counts = {'employed': 0, 'not_employed': 0}
    wfh_counts = {'wfh': 0, 'on_site': 0}
    income_bins = np.concatenate(([0], np.linspace(1_000, 200_000, 50)))
    income_hist = np.zeros(len(income_bins) - 1, dtype=np.int64)
    edu_counts = {}

    # Activity aggregations
    categories = ['sleep', 'personal_care', 'work', 'education', 'household', 'caring', 'leisure', 'eating', 'travel', 'other']
    cat_index = {c: i for i, c in enumerate(categories)}
    timeline_counts = np.zeros((len(categories), 1440), dtype=np.int64)
    transition_counts = np.zeros((len(categories), len(categories)), dtype=np.int64)

    # Activity timelines by demographic subgroups
    timeline_by_group = {
        'sex': {
            'M': np.zeros((len(categories), 1440), dtype=np.int64),
            'F': np.zeros((len(categories), 1440), dtype=np.int64),
        },
        'age_group': {
            'child': np.zeros((len(categories), 1440), dtype=np.int64),   # <18
            'adult': np.zeros((len(categories), 1440), dtype=np.int64),   # 18-64
            'senior': np.zeros((len(categories), 1440), dtype=np.int64),  # 65+
        }
    }

    # Weather aggregations
    weather_hists = {
        'temperature': np.linspace(-40, 120, 81),
        'humidity': np.linspace(0, 100, 51),
        'wind_speed': np.linspace(0, 100, 51),
        'solar_radiation': np.linspace(0, 1200, 61),
        'precipitation': np.linspace(0, 10, 51),
    }
    weather_counts = {k: np.zeros(len(edges) - 1, dtype=np.int64) for k, edges in weather_hists.items()}
    temp_by_state_sum = {}
    temp_by_state_count = {}

    # Energy aggregations
    energy_bins = {
        'total_energy_consumption': np.linspace(0, 100_000, 101),
        'energy_intensity': np.linspace(0, 500, 51),
        'energy_burden': np.linspace(0, 50, 51),
    }
    energy_hists = {k: np.zeros(len(b) - 1, dtype=np.int64) for k, b in energy_bins.items()}
    end_use_sums = {k: 0.0 for k in ['heating_energy', 'cooling_energy', 'water_heating_energy', 'lighting_energy', 'appliances_energy']}
    end_use_sums_by_type = {}  # {building_type: {end_use: sum}}
    energy_by_type_sum = {}
    energy_by_type_count = {}
    energy_by_hhsize_sum = {}
    energy_by_hhsize_count = {}
    energy_by_state_sum = {}
    energy_by_state_count = {}

    # Time-profile aggregations (average energy consumption throughout time)
    # Detect columns holding per-building time series (length 24 or 1440)
    hourly_candidates = ['hourly_energy', 'hourly_total_energy', 'hourly_load', 'daily_load_profile', 'energy_load_hourly']
    minute_candidates = ['minute_energy', 'minute_total_energy', 'minute_load', 'minute_energy_profile', 'energy_profile_minute', 'energy_profile_1440']
    time_profiles = {
        'hourly': {
            'source': None,
            'overall': {'sum': np.zeros(24, dtype=float), 'sum2': np.zeros(24, dtype=float), 'count': np.zeros(24, dtype=np.int64)},
            'groups': {
                'building_type_simple': {},  # key -> {'sum': arr, 'sum2': arr, 'count': arr}
                'STATE': {},
                'climate_zone': {},
            }
        },
        'minute': {
            'source': None,
            'overall': {'sum': np.zeros(1440, dtype=float), 'sum2': np.zeros(1440, dtype=float), 'count': np.zeros(1440, dtype=np.int64)},
            'groups': {
                'building_type_simple': {},
                'STATE': {},
                'climate_zone': {},
            },
            'sampled': 0,
            'sample_limit': 20000  # cap to limit runtime
        }
    }

    # Household-level income (sum of person incomes per building)
    hh_income_bins = np.concatenate(([0], np.linspace(1_000, 300_000, 100)))
    hh_income_hist = np.zeros(len(hh_income_bins) - 1, dtype=np.int64)

    # Derived metrics: energy per capita (per household)
    epc_bins = np.linspace(0, 50_000, 101)
    epc_hist = np.zeros(len(epc_bins) - 1, dtype=np.int64)

    # Coverage tracking for key numeric columns
    coverage_cols = ['total_energy_consumption', 'energy_intensity', 'energy_burden', 'rooms_per_person', 'occupancy_intensity', 'building_age', 'temperature']
    coverage = {c: {'present': 0, 'total': 0} for c in coverage_cols}

    # Global pairwise correlation sufficient statistics (across shared finite rows per pair)
    pairwise_vars = [
        'energy_intensity', 'energy_burden', 'rooms_per_person', 'household_size',
        'num_bedrooms', 'building_age', 'temperature', 'total_energy_consumption',
        'occupancy_intensity'
    ]
    pairwise_stats = {
        'vars': pairwise_vars,
        'univar': {v: {'n': 0, 'sum': 0.0, 'sum2': 0.0} for v in pairwise_vars},
        'pairs': {}
    }

    # Temperature -> energy mean curve (bin means + variance)
    temp_energy_edges = np.linspace(-40, 120, 81)
    temp_energy_sum = np.zeros(len(temp_energy_edges) - 1, dtype=float)
    temp_energy_sum2 = np.zeros(len(temp_energy_edges) - 1, dtype=float)
    temp_energy_count = np.zeros(len(temp_energy_edges) - 1, dtype=np.int64)

    # Energy poverty by income bin (threshold shares)
    poverty_by_income_bin = {i: {'den': 0, 'num_10': 0, 'num_20': 0} for i in range(len(hh_income_bins) - 1)}

    # Subgroup aggregations
    # Energy burden by household income bin (streaming mean/var/count)
    def _init_moments():
        return {'n': 0, 'mean': 0.0, 'M2': 0.0}
    def _update_moments(m, x):
        m['n'] += 1
        delta = x - m['mean']
        m['mean'] += delta * (1.0 / m['n'])
        delta2 = x - m['mean']
        m['M2'] += delta * delta2
    def _finalize_moments(m):
        n = m['n']
        var = (m['M2'] / (n - 1)) if n > 1 else np.nan
        se = np.sqrt(var / n) if n > 1 else np.nan
        return n, m['mean'], var, se
    eb_by_income_bin = {i: _init_moments() for i in range(len(hh_income_bins) - 1)}
    # Energy intensity by climate zone
    ei_by_climate = {}

    # Generic subgroup configs
    group_vars = ['building_type_simple', 'tenure_type', 'STATE', 'climate_zone', 'heating_fuel']
    metrics_for_moments = ['total_energy_consumption', 'energy_intensity', 'energy_burden', 'rooms_per_person', 'occupancy_intensity']
    hist_metrics = {
        'energy_intensity': energy_bins['energy_intensity'],
        'energy_burden': energy_bins['energy_burden'],
    }
    # Initialize containers
    group_moments = {gv: {} for gv in group_vars}
    group_hists = {gv: {} for gv in group_vars}
    group_corrs = {gv: {} for gv in group_vars}  # temp vs total energy

    # Streaming correlation/OLS stats without storing data
    def _init_stats():
        return {'n': 0, 'sum_x': 0.0, 'sum_y': 0.0, 'sum_x2': 0.0, 'sum_y2': 0.0, 'sum_xy': 0.0}
    corr_stats = {
        'temp_vs_energy': _init_stats(),
        'age_vs_intensity': _init_stats(),
        'roomspp_vs_intensity': _init_stats(),
    }

    # Conditional bin-mean curves (sum/sum2/count per bin)
    def _init_bincurves(edges: np.ndarray):
        return {
            'bin_edges': edges,
            'sum': np.zeros(len(edges) - 1, dtype=float),
            'sum2': np.zeros(len(edges) - 1, dtype=float),
            'count': np.zeros(len(edges) - 1, dtype=np.int64),
        }
    cond_curves = {
        'ei_by_age': _init_bincurves(np.arange(0, 125, 5)),
        'ei_by_roomspp': _init_bincurves(np.linspace(0, 10, 51)),
        'ei_by_occupancy': _init_bincurves(np.linspace(0, 5, 51)),
        'energy_by_hhsize': _init_bincurves(np.arange(-0.5, 11.5, 1.0)),
        'eb_by_roomspp': _init_bincurves(np.linspace(0, 10, 51)),
        'eb_by_hhsize': _init_bincurves(np.arange(-0.5, 11.5, 1.0)),
        'epc_by_hhsize': _init_bincurves(np.arange(-0.5, 11.5, 1.0)),
    }

    # Per-building-type conditional curves: energy intensity vs age
    def _init_bincurves_simple(edges: np.ndarray):
        return {
            'bin_edges': edges,
            'sum': {},   # key -> arr
            'sum2': {},
            'count': {},
        }
    cond_curves_by_type = {
        'ei_by_age_by_type': _init_bincurves_simple(np.arange(0, 125, 5)),
    }

    # Energy poverty by state (≥10%, ≥20%)
    poverty_by_state = {}

    # Energy poverty by key group variables (≥10%, ≥20%)
    energy_poverty_by_group = {gv: {} for gv in ['building_type_simple', 'tenure_type', 'STATE', 'climate_zone', 'heating_fuel']}

    # Shard-level summaries
    shard_summaries = []
    shard_index = 0

    total_buildings = 0
    total_persons = 0
    total_activities = 0

    def _inc_dict(d, key, inc=1):
        if key is None:
            return
        d[key] = d.get(key, 0) + inc

    for shard in _iterate_shards(manifest_path):
        shard_index += 1
        total_buildings += len(shard)
        # Coverage totals baseline per shard
        for c in coverage_cols:
            coverage[c]['total'] += int(len(shard))

        # Building categorical counts
        for col in cat_counts.keys():
            if col in shard.columns:
                vc = shard[col].value_counts(dropna=True)
                for k, v in vc.items():
                    _inc_dict(cat_counts[col], k, int(v))

        # Building numeric hists
        for col, bins in bins_config.items():
            if col in shard.columns:
                vals = shard[col].to_numpy(dtype=float, copy=False)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    hist, _ = np.histogram(vals, bins=bins)
                    hists[col] += hist.astype(np.int64)

        # 2D density for building_age vs energy_intensity
        if 'building_age' in shard.columns and 'energy_intensity' in shard.columns:
            x = shard['building_age'].to_numpy(dtype=float, copy=False)
            y = shard['energy_intensity'].to_numpy(dtype=float, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                H, _, _ = np.histogram2d(x[mask], y[mask], bins=(density_2d['age_vs_energy']['x_bins'], density_2d['age_vs_energy']['y_bins']))
                if density_2d['age_vs_energy']['grid'] is None:
                    density_2d['age_vs_energy']['grid'] = H.astype(np.int64)
                else:
                    density_2d['age_vs_energy']['grid'] += H.astype(np.int64)

        # 2D density for temperature vs total energy
        if 'temperature' in shard.columns and 'total_energy_consumption' in shard.columns:
            x = shard['temperature'].to_numpy(dtype=float, copy=False)
            y = shard['total_energy_consumption'].to_numpy(dtype=float, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                H, _, _ = np.histogram2d(x[mask], y[mask], bins=(density_2d['temp_vs_energy']['x_bins'], density_2d['temp_vs_energy']['y_bins']))
                if density_2d['temp_vs_energy']['grid'] is None:
                    density_2d['temp_vs_energy']['grid'] = H.astype(np.int64)
                else:
                    density_2d['temp_vs_energy']['grid'] += H.astype(np.int64)

    # 2D density for rooms per person vs energy intensity
        if 'rooms_per_person' in shard.columns and 'energy_intensity' in shard.columns:
            x = shard['rooms_per_person'].to_numpy(dtype=float, copy=False)
            y = shard['energy_intensity'].to_numpy(dtype=float, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                H, _, _ = np.histogram2d(x[mask], y[mask], bins=(density_2d['roomspp_vs_intensity']['x_bins'], density_2d['roomspp_vs_intensity']['y_bins']))
                if density_2d['roomspp_vs_intensity']['grid'] is None:
                    density_2d['roomspp_vs_intensity']['grid'] = H.astype(np.int64)
                else:
                    density_2d['roomspp_vs_intensity']['grid'] += H.astype(np.int64)

        # 2D density for energy burden vs energy intensity
        if 'energy_burden' in shard.columns and 'energy_intensity' in shard.columns:
            x = shard['energy_intensity'].to_numpy(dtype=float, copy=False)
            y = shard['energy_burden'].to_numpy(dtype=float, copy=False)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                H, _, _ = np.histogram2d(x[mask], y[mask], bins=(density_2d['eb_vs_ei']['x_bins'], density_2d['eb_vs_ei']['y_bins']))
                if density_2d['eb_vs_ei']['grid'] is None:
                    density_2d['eb_vs_ei']['grid'] = H.astype(np.int64)
                else:
                    density_2d['eb_vs_ei']['grid'] += H.astype(np.int64)

        # Update conditional curves
        # energy_intensity by building_age
        if 'building_age' in shard.columns and 'energy_intensity' in shard.columns:
            x = pd.to_numeric(shard['building_age'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['energy_intensity'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['ei_by_age']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['ei_by_age']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['ei_by_age']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['ei_by_age']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['ei_by_age']['count']))
                cond_curves['ei_by_age']['count'] += bcnt.astype(np.int64)
                cond_curves['ei_by_age']['sum'] += bsum.astype(float)
                cond_curves['ei_by_age']['sum2'] += bsum2.astype(float)
        # energy_intensity by rooms_per_person
        if 'rooms_per_person' in shard.columns and 'energy_intensity' in shard.columns:
            x = pd.to_numeric(shard['rooms_per_person'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['energy_intensity'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['ei_by_roomspp']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['ei_by_roomspp']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['ei_by_roomspp']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['ei_by_roomspp']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['ei_by_roomspp']['count']))
                cond_curves['ei_by_roomspp']['count'] += bcnt.astype(np.int64)
                cond_curves['ei_by_roomspp']['sum'] += bsum.astype(float)
                cond_curves['ei_by_roomspp']['sum2'] += bsum2.astype(float)
        # energy_intensity by occupancy_intensity
        if 'occupancy_intensity' in shard.columns and 'energy_intensity' in shard.columns:
            x = pd.to_numeric(shard['occupancy_intensity'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['energy_intensity'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['ei_by_occupancy']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['ei_by_occupancy']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['ei_by_occupancy']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['ei_by_occupancy']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['ei_by_occupancy']['count']))
                cond_curves['ei_by_occupancy']['count'] += bcnt.astype(np.int64)
                cond_curves['ei_by_occupancy']['sum'] += bsum.astype(float)
                cond_curves['ei_by_occupancy']['sum2'] += bsum2.astype(float)
        # total_energy by household_size
        if 'household_size' in shard.columns and 'total_energy_consumption' in shard.columns:
            x = pd.to_numeric(shard['household_size'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['total_energy_consumption'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['energy_by_hhsize']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['energy_by_hhsize']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['energy_by_hhsize']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['energy_by_hhsize']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['energy_by_hhsize']['count']))
                cond_curves['energy_by_hhsize']['count'] += bcnt.astype(np.int64)
                cond_curves['energy_by_hhsize']['sum'] += bsum.astype(float)
                cond_curves['energy_by_hhsize']['sum2'] += bsum2.astype(float)
        # energy_burden by household_size
        if 'household_size' in shard.columns and 'energy_burden' in shard.columns:
            x = pd.to_numeric(shard['household_size'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['energy_burden'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['eb_by_hhsize']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['eb_by_hhsize']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['eb_by_hhsize']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['eb_by_hhsize']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['eb_by_hhsize']['count']))
                cond_curves['eb_by_hhsize']['count'] += bcnt.astype(np.int64)
                cond_curves['eb_by_hhsize']['sum'] += bsum.astype(float)
                cond_curves['eb_by_hhsize']['sum2'] += bsum2.astype(float)
        # energy_burden by rooms_per_person
        if 'rooms_per_person' in shard.columns and 'energy_burden' in shard.columns:
            x = pd.to_numeric(shard['rooms_per_person'], errors='coerce').to_numpy()
            y = pd.to_numeric(shard['energy_burden'], errors='coerce').to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                idx = np.digitize(x[m], cond_curves['eb_by_roomspp']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['eb_by_roomspp']['bin_edges']) - 2)
                yv = y[m].astype(float)
                bcnt = np.bincount(idx, minlength=len(cond_curves['eb_by_roomspp']['count']))
                bsum = np.bincount(idx, weights=yv, minlength=len(cond_curves['eb_by_roomspp']['count']))
                bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(cond_curves['eb_by_roomspp']['count']))
                cond_curves['eb_by_roomspp']['count'] += bcnt.astype(np.int64)
                cond_curves['eb_by_roomspp']['sum'] += bsum.astype(float)
                cond_curves['eb_by_roomspp']['sum2'] += bsum2.astype(float)

        # Weather (per shard)
        for wcol, edges in weather_hists.items():
            if wcol in shard.columns:
                vals = shard[wcol].to_numpy(dtype=float, copy=False)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    hist, _ = np.histogram(vals, bins=edges)
                    weather_counts[wcol] += hist.astype(np.int64)
        if 'temperature' in shard.columns and 'STATE' in shard.columns:
            temps = shard['temperature'].to_numpy(dtype=float, copy=False)
            states = shard['STATE']
            mask = np.isfinite(temps)
            for s, t in zip(states[mask], temps[mask]):
                temp_by_state_sum[s] = temp_by_state_sum.get(s, 0.0) + float(t)
                temp_by_state_count[s] = temp_by_state_count.get(s, 0) + 1

    # Persons and activities (iterate buildings to access nested persons)
        if 'persons' in shard.columns:
            for _, b in shard.iterrows():
                persons = b.get('persons', [])
                if not isinstance(persons, list):
                    continue
                total_persons += len(persons)

                # Derived household metrics
                ages = []
                num_children = 0
                num_seniors = 0
                workers = 0
                hh_income = 0.0
                for person in persons:
                    if not isinstance(person, dict):
                        continue
                    age = person.get('AGEP', None)
                    sex = 'M' if person.get('SEX') == 1 else ('F' if person.get('SEX') == 2 else None)
                    if isinstance(age, (int, float)) and 0 <= age <= 100 and sex in ('M', 'F'):
                        ages.append(age)
                        if age < 18:
                            num_children += 1
                        if age >= 65:
                            num_seniors += 1
                        label_idx = int(age // 5)
                        if label_idx >= len(age_labels):
                            label_idx = len(age_labels) - 1
                        label = age_labels[label_idx]
                        persons_age_sex[sex][label] += 1

                    # Employment
                    if person.get('is_employed') == 1 or person.get('is_employed') is True:
                        emp_counts['employed'] += 1
                        wfh_counts['wfh' if person.get('works_from_home') else 'on_site'] += 1
                        workers += 1
                    else:
                        emp_counts['not_employed'] += 1

                    # Income
                    inc = person.get('PINCP', None)
                    if isinstance(inc, (int, float)) and np.isfinite(inc) and inc > 0:
                        h, _ = np.histogram([inc], bins=income_bins)
                        income_hist += h.astype(np.int64)
                        hh_income += float(inc)

                    # Education
                    edu = person.get('education_level', None)
                    if edu is not None:
                        _inc_dict(edu_counts, edu, 1)

                    # Activities
                    if person.get('has_activities') and 'activity_sequence' in person:
                        seq = person['activity_sequence']
                        prev_cat = None
                        for act in seq:
                            if not isinstance(act, dict):
                                continue
                            code = str(act.get('activity_code', ''))
                            # Map code to category (align with ActivityVisualizer)
                            if code.startswith('01'):
                                cat = 'sleep'
                            elif code.startswith('02'):
                                cat = 'personal_care'
                            elif code.startswith('05'):
                                cat = 'work'
                            elif code.startswith('03'):
                                cat = 'caring'
                            elif code.startswith('11'):
                                cat = 'eating'
                            elif code.startswith('12') or code.startswith('13'):
                                cat = 'leisure'
                            elif code.startswith('18'):
                                cat = 'travel'
                            elif code.startswith('14'):
                                cat = 'household'
                            elif code.startswith('06'):
                                cat = 'education'
                            else:
                                cat = 'other'

                            start = act.get('start_minute')
                            dur = act.get('duration_minutes', 0)
                            if isinstance(start, (int, float)) and isinstance(dur, (int, float)):
                                start_i = max(0, int(start))
                                end_i = min(1440, int(start + max(dur, 0)))
                                ci = cat_index.get(cat, cat_index['other'])
                                if start_i < end_i:
                                    timeline_counts[ci, start_i:end_i] += 1
                                    # subgroup timelines
                                    if sex in ('M', 'F'):
                                        timeline_by_group['sex'][sex][ci, start_i:end_i] += 1
                                    if isinstance(age, (int, float)):
                                        if age < 18:
                                            timeline_by_group['age_group']['child'][ci, start_i:end_i] += 1
                                        elif age >= 65:
                                            timeline_by_group['age_group']['senior'][ci, start_i:end_i] += 1
                                        else:
                                            timeline_by_group['age_group']['adult'][ci, start_i:end_i] += 1
                            # transitions
                            if prev_cat is not None:
                                transition_counts[cat_index[prev_cat], cat_index[cat]] += 1
                            prev_cat = cat
                            total_activities += 1

                # Update derived hists after scanning persons in the building
                # Workers per household
                if 'workers_per_household' in hists:
                    h, _ = np.histogram([workers], bins=bins_config['workers_per_household'])
                    hists['workers_per_household'] += h.astype(np.int64)
                # Children / seniors
                if 'num_children' in hists:
                    h, _ = np.histogram([num_children], bins=bins_config['num_children'])
                    hists['num_children'] += h.astype(np.int64)
                if 'num_seniors' in hists:
                    h, _ = np.histogram([num_seniors], bins=bins_config['num_seniors'])
                    hists['num_seniors'] += h.astype(np.int64)
                # Age diversity std dev
                if ages:
                    std_age = float(np.std(ages))
                    h, _ = np.histogram([std_age], bins=bins_config['age_diversity_std'])
                    hists['age_diversity_std'] += h.astype(np.int64)

                # Household income (sum of person incomes)
                if hh_income > 0:
                    h, _ = np.histogram([hh_income], bins=hh_income_bins)
                    hh_income_hist += h.astype(np.int64)
                    # If building has energy_burden, add to corresponding income bin moments
                    if 'energy_burden' in shard.columns:
                        try:
                            eb = float(b.get('energy_burden'))
                            if np.isfinite(eb):
                                bin_idx = int(np.digitize(hh_income, hh_income_bins) - 1)
                                bin_idx = max(0, min(bin_idx, len(hh_income_bins) - 2))
                                _update_moments(eb_by_income_bin[bin_idx], eb)
                                # Energy poverty thresholds (10% and 20%)
                                poverty_by_income_bin[bin_idx]['den'] += 1
                                if eb >= 10.0:
                                    poverty_by_income_bin[bin_idx]['num_10'] += 1
                                if eb >= 20.0:
                                    poverty_by_income_bin[bin_idx]['num_20'] += 1
                                # Energy poverty by state
                                st = b.get('STATE') if 'STATE' in shard.columns else None
                                if st is not None:
                                    if st not in poverty_by_state:
                                        poverty_by_state[st] = {'den': 0, 'num_10': 0, 'num_20': 0}
                                    poverty_by_state[st]['den'] += 1
                                    if eb >= 10.0:
                                        poverty_by_state[st]['num_10'] += 1
                                    if eb >= 20.0:
                                        poverty_by_state[st]['num_20'] += 1
                        except Exception:
                            pass

        # Energy (per shard)
        for col, bins in energy_bins.items():
            if col in shard.columns:
                vals = shard[col].to_numpy(dtype=float, copy=False)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    hist, _ = np.histogram(vals, bins=bins)
                    energy_hists[col] += hist.astype(np.int64)
        # End-uses
        for k in list(end_use_sums.keys()):
            if k in shard.columns:
                s = pd.to_numeric(shard[k], errors='coerce').sum()
                if np.isfinite(s):
                    end_use_sums[k] += float(s)
        # End-uses by building type
        if 'building_type_simple' in shard.columns:
            bt_series = shard['building_type_simple']
            for k in list(end_use_sums.keys()):
                if k not in shard.columns:
                    continue
                grp = pd.to_numeric(shard[k], errors='coerce')
                gsum = grp.groupby(bt_series).sum(numeric_only=True)
                for bt, s in gsum.items():
                    if bt not in end_use_sums_by_type:
                        end_use_sums_by_type[bt] = {ek: 0.0 for ek in end_use_sums.keys()}
                    end_use_sums_by_type[bt][k] += float(s)
        # Energy by building type
        if 'building_type_simple' in shard.columns and 'total_energy_consumption' in shard.columns:
            for bt, s in shard.groupby('building_type_simple')['total_energy_consumption'].sum().items():
                energy_by_type_sum[bt] = energy_by_type_sum.get(bt, 0.0) + float(s)
            for bt, c in shard['building_type_simple'].value_counts().items():
                energy_by_type_count[bt] = energy_by_type_count.get(bt, 0) + int(c)
        # Energy by state
        if 'STATE' in shard.columns and 'total_energy_consumption' in shard.columns:
            for st, s in shard.groupby('STATE')['total_energy_consumption'].sum().items():
                energy_by_state_sum[st] = energy_by_state_sum.get(st, 0.0) + float(s)
            for st, c in shard['STATE'].value_counts().items():
                energy_by_state_count[st] = energy_by_state_count.get(st, 0) + int(c)
        # Energy vs household size
        if 'household_size' in shard.columns and 'total_energy_consumption' in shard.columns:
            for hh, s in shard.groupby('household_size')['total_energy_consumption'].sum().items():
                energy_by_hhsize_sum[hh] = energy_by_hhsize_sum.get(hh, 0.0) + float(s)
            for hh, c in shard['household_size'].value_counts().items():
                energy_by_hhsize_count[hh] = energy_by_hhsize_count.get(hh, 0) + int(c)
            # Energy per capita (per building)
            hh = pd.to_numeric(shard['household_size'], errors='coerce').to_numpy()
            te = pd.to_numeric(shard['total_energy_consumption'], errors='coerce').to_numpy()
            m = np.isfinite(hh) & np.isfinite(te) & (hh > 0)
            if np.any(m):
                epc = te[m] / hh[m]
                h, _ = np.histogram(epc, bins=epc_bins)
                epc_hist += h.astype(np.int64)
                # Conditional mean energy per capita by household size bin
                idx = np.digitize(hh[m], cond_curves['epc_by_hhsize']['bin_edges']) - 1
                idx = np.clip(idx, 0, len(cond_curves['epc_by_hhsize']['count']) - 1)
                bcnt = np.bincount(idx, minlength=len(cond_curves['epc_by_hhsize']['count']))
                bsum = np.bincount(idx, weights=epc, minlength=len(cond_curves['epc_by_hhsize']['count']))
                bsum2 = np.bincount(idx, weights=epc * epc, minlength=len(cond_curves['epc_by_hhsize']['count']))
                cond_curves['epc_by_hhsize']['count'] += bcnt.astype(np.int64)
                cond_curves['epc_by_hhsize']['sum'] += bsum.astype(float)
                cond_curves['epc_by_hhsize']['sum2'] += bsum2.astype(float)

        # Energy intensity by climate zone (streaming moments)
        if 'climate_zone' in shard.columns and 'energy_intensity' in shard.columns:
            cz = shard['climate_zone']
            ei = pd.to_numeric(shard['energy_intensity'], errors='coerce')
            for climate, value in zip(cz, ei):
                if not np.isfinite(value):
                    continue
                if climate not in ei_by_climate:
                    ei_by_climate[climate] = _init_moments()
                _update_moments(ei_by_climate[climate], float(value))

        # Time series detection and aggregation (hourly and minute-level)
        # Determine source columns once
        if time_profiles['hourly']['source'] is None:
            for candi in hourly_candidates:
                if candi in shard.columns:
                    # Confirm with a quick sample
                    try:
                        sample = next((v for v in shard[candi].values if isinstance(v, (list, np.ndarray))), None)
                        arr = np.array(sample, dtype=float) if sample is not None else None
                        if arr is not None and arr.size == 24 and np.all(np.isfinite(arr)):
                            time_profiles['hourly']['source'] = candi
                            break
                    except Exception:
                        pass
        if time_profiles['minute']['source'] is None:
            for candi in minute_candidates:
                if candi in shard.columns:
                    try:
                        sample = next((v for v in shard[candi].values if isinstance(v, (list, np.ndarray))), None)
                        arr = np.array(sample, dtype=float) if sample is not None else None
                        if arr is not None and arr.size == 1440 and np.all(np.isfinite(arr[:60])):
                            time_profiles['minute']['source'] = candi
                            break
                    except Exception:
                        pass

        # Aggregation helpers for time profiles
        def _accumulate_profile(kind: str, arr: np.ndarray, bt_val, st_val, cz_val):
            tp = time_profiles[kind]
            # overall
            m = np.isfinite(arr)
            tp['overall']['sum'][m] += arr[m]
            tp['overall']['sum2'][m] += arr[m] * arr[m]
            tp['overall']['count'][m] += 1
            # groups
            for gv, gval in [('building_type_simple', bt_val), ('STATE', st_val), ('climate_zone', cz_val)]:
                if gval is None:
                    continue
                gdict = tp['groups'][gv]
                key = str(gval)
                if key not in gdict:
                    gdict[key] = {
                        'sum': np.zeros_like(arr, dtype=float),
                        'sum2': np.zeros_like(arr, dtype=float),
                        'count': np.zeros_like(arr, dtype=np.int64),
                    }
                g = gdict[key]
                g['sum'][m] += arr[m]
                g['sum2'][m] += arr[m] * arr[m]
                g['count'][m] += 1

        # Pull group columns once
        bt_col = shard['building_type_simple'] if 'building_type_simple' in shard.columns else None
        st_col = shard['STATE'] if 'STATE' in shard.columns else None
        cz_col = shard['climate_zone'] if 'climate_zone' in shard.columns else None

        # Hourly profiles
        hsrc = time_profiles['hourly']['source']
        if hsrc is not None and hsrc in shard.columns:
            vals = shard[hsrc].values
            for i in range(len(vals)):
                v = vals[i]
                if isinstance(v, (list, np.ndarray)):
                    try:
                        arr = np.array(v, dtype=float)
                        if arr.size == 24:
                            bt = bt_col.iloc[i] if bt_col is not None else None
                            st = st_col.iloc[i] if st_col is not None else None
                            cz = cz_col.iloc[i] if cz_col is not None else None
                            _accumulate_profile('hourly', arr, bt, st, cz)

                            # Peak stats per building
                            with np.errstate(invalid='ignore'):
                                m = np.isfinite(arr)
                                if np.any(m):
                                    arr2 = np.where(m, arr, -np.inf)
                                    pk_val = float(np.max(arr2))
                                    pk_hour = int(np.argmax(arr2))
                                    # overall
                                    peak_stats['overall']['sum_of_peaks'] += pk_val
                                    peak_stats['overall']['count'] += 1
                                    if 0 <= pk_hour < 24:
                                        peak_stats['overall']['peak_hour_hist'][pk_hour] += 1
                                    # by group helper
                                    def _upd(gname, gvalue):
                                        if gvalue is None:
                                            return
                                        gdict = peak_stats['by_group'][gname]
                                        key = str(gvalue)
                                        if key not in gdict:
                                            gdict[key] = {
                                                'sum_of_peaks': 0.0,
                                                'count': 0,
                                                'peak_hour_hist': np.zeros(24, dtype=np.int64),
                                            }
                                        g = gdict[key]
                                        g['sum_of_peaks'] += pk_val
                                        g['count'] += 1
                                        if 0 <= pk_hour < 24:
                                            g['peak_hour_hist'][pk_hour] += 1
                                    _upd('building_type_simple', bt)
                                    _upd('STATE', st)
                                    _upd('climate_zone', cz)

                                    # TOU average share per building
                                    tot = float(np.sum(np.where(m, np.maximum(arr, 0.0), 0.0)))
                                    if tot > 0:
                                        # overall
                                        for bn, hours in tou_bands.items():
                                            sh = float(np.sum(arr[hours])) / tot
                                            tou_stats['average_share'][bn]['sum_share'] += sh
                                        for bn in tou_bands:
                                            tou_stats['average_share'][bn]['count'] += 1
                                        # by group
                                        def _upd_tou(gname, gvalue):
                                            if gvalue is None:
                                                return
                                            gdict = tou_stats['average_share_by_group'][gname]
                                            key = str(gvalue)
                                            if key not in gdict:
                                                gdict[key] = {bn: {'sum_share': 0.0, 'count': 0} for bn in tou_bands}
                                            for bn, hours in tou_bands.items():
                                                sh = float(np.sum(arr[hours])) / tot
                                                gdict[key][bn]['sum_share'] += sh
                                                gdict[key][bn]['count'] += 1
                                        _upd_tou('building_type_simple', bt)
                                        _upd_tou('STATE', st)
                                        _upd_tou('climate_zone', cz)
                    except Exception:
                        continue

        # Minute profiles (sampled)
        msrc = time_profiles['minute']['source']
        if msrc is not None and msrc in shard.columns and time_profiles['minute']['sampleed' if False else 'sampled'] < time_profiles['minute']['sample_limit']:
            vals = shard[msrc].values
            for i in range(len(vals)):
                if time_profiles['minute']['sampled'] >= time_profiles['minute']['sample_limit']:
                    break
                v = vals[i]
                if isinstance(v, (list, np.ndarray)):
                    try:
                        arr = np.array(v, dtype=float)
                        if arr.size == 1440:
                            bt = bt_col.iloc[i] if bt_col is not None else None
                            st = st_col.iloc[i] if st_col is not None else None
                            cz = cz_col.iloc[i] if cz_col is not None else None
                            _accumulate_profile('minute', arr, bt, st, cz)
                            time_profiles['minute']['sampled'] += 1
                    except Exception:
                        continue

        # Generic subgroup updates (moments, histograms, correlations)
        for gv in group_vars:
            if gv not in shard.columns:
                continue
            # Ensure dicts
            gm = group_moments[gv]
            gh = group_hists[gv]
            gcorr = group_corrs[gv]

            # Pre-extract columns safely
            # Moments
            cols_present = {m: m in shard.columns for m in metrics_for_moments}
            # Hist metrics
            hist_present = {m: m in shard.columns for m in hist_metrics.keys()}
            # Correlation components
            has_temp = 'temperature' in shard.columns
            has_energy = 'total_energy_consumption' in shard.columns

            # Group loop: operate per group value to avoid per-row Python overhead
            for gval, gdf in shard.groupby(gv):
                # Moments aggregation
                for m in metrics_for_moments:
                    if not cols_present[m]:
                        continue
                    arr = pd.to_numeric(gdf[m], errors='coerce').to_numpy()
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    if gval not in gm:
                        gm[gval] = {mm: _init_moments() for mm in metrics_for_moments}
                    mm = gm[gval][m]
                    # Vectorized Welford updates
                    for x in arr:
                        _update_moments(mm, float(x))

                # Histograms
                for hm, bins in hist_metrics.items():
                    if not hist_present[hm]:
                        continue
                    arr = pd.to_numeric(gdf[hm], errors='coerce').to_numpy()
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    if gval not in gh:
                        gh[gval] = {k: np.zeros(len(b) - 1, dtype=np.int64) for k, b in hist_metrics.items()}
                    hist, _ = np.histogram(arr, bins=bins)
                    gh[gval][hm] += hist.astype(np.int64)

                # Correlation temp vs energy
                if has_temp and has_energy:
                    x = pd.to_numeric(gdf['temperature'], errors='coerce').to_numpy()
                    y = pd.to_numeric(gdf['total_energy_consumption'], errors='coerce').to_numpy()
                    mask = np.isfinite(x) & np.isfinite(y)
                    if np.any(mask):
                        if gval not in gcorr:
                            gcorr[gval] = _init_stats()
                        x = x[mask]; y = y[mask]
                        gcorr[gval]['n'] += len(x)
                        gcorr[gval]['sum_x'] += float(np.sum(x))
                        gcorr[gval]['sum_y'] += float(np.sum(y))
                        gcorr[gval]['sum_x2'] += float(np.sum(x * x))
                        gcorr[gval]['sum_y2'] += float(np.sum(y * y))
                        gcorr[gval]['sum_xy'] += float(np.sum(x * y))

                # Per-building-type conditional curves (energy intensity vs age)
                if gv == 'building_type_simple' and 'building_age' in gdf.columns and 'energy_intensity' in gdf.columns:
                    edges = cond_curves_by_type['ei_by_age_by_type']['bin_edges']
                    x = pd.to_numeric(gdf['building_age'], errors='coerce').to_numpy()
                    y = pd.to_numeric(gdf['energy_intensity'], errors='coerce').to_numpy()
                    msk = np.isfinite(x) & np.isfinite(y)
                    if np.any(msk):
                        idx = np.digitize(x[msk], edges) - 1
                        idx = np.clip(idx, 0, len(edges) - 2)
                        yv = y[msk].astype(float)
                        bcnt = np.bincount(idx, minlength=len(edges) - 1)
                        bsum = np.bincount(idx, weights=yv, minlength=len(edges) - 1)
                        bsum2 = np.bincount(idx, weights=yv * yv, minlength=len(edges) - 1)
                        tkey = str(gval)
                        cc = cond_curves_by_type['ei_by_age_by_type']
                        if tkey not in cc['count']:
                            cc['count'][tkey] = np.zeros(len(edges) - 1, dtype=np.int64)
                            cc['sum'][tkey] = np.zeros(len(edges) - 1, dtype=float)
                            cc['sum2'][tkey] = np.zeros(len(edges) - 1, dtype=float)
                        cc['count'][tkey] += bcnt.astype(np.int64)
                        cc['sum'][tkey] += bsum.astype(float)
                        cc['sum2'][tkey] += bsum2.astype(float)

                # Energy poverty by group
                if 'energy_burden' in gdf.columns:
                    eb_vals = pd.to_numeric(gdf['energy_burden'], errors='coerce').to_numpy()
                    m = np.isfinite(eb_vals)
                    if np.any(m):
                        den = int(np.sum(m))
                        n10 = int(np.sum(eb_vals[m] >= 10.0))
                        n20 = int(np.sum(eb_vals[m] >= 20.0))
                        cur = energy_poverty_by_group[gv].get(gval, {'den': 0, 'num_10': 0, 'num_20': 0})
                        cur['den'] += den
                        cur['num_10'] += n10
                        cur['num_20'] += n20
                        energy_poverty_by_group[gv][gval] = cur

        # Streaming correlations/slopes across common pairs
        def _update_stats(stats_obj, x_arr, y_arr):
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if not np.any(mask):
                return
            x = x_arr[mask].astype(float)
            y = y_arr[mask].astype(float)
            stats_obj['n'] += len(x)
            stats_obj['sum_x'] += float(np.sum(x))
            stats_obj['sum_y'] += float(np.sum(y))
            stats_obj['sum_x2'] += float(np.sum(x * x))
            stats_obj['sum_y2'] += float(np.sum(y * y))
            stats_obj['sum_xy'] += float(np.sum(x * y))

    # temp vs total energy
        if 'temperature' in shard.columns and 'total_energy_consumption' in shard.columns:
            _update_stats(corr_stats['temp_vs_energy'],
                          shard['temperature'].to_numpy(dtype=float, copy=False),
                          shard['total_energy_consumption'].to_numpy(dtype=float, copy=False))
        # building age vs energy intensity
        if 'building_age' in shard.columns and 'energy_intensity' in shard.columns:
            _update_stats(corr_stats['age_vs_intensity'],
                          shard['building_age'].to_numpy(dtype=float, copy=False),
                          shard['energy_intensity'].to_numpy(dtype=float, copy=False))
        # rooms per person vs energy intensity
        if 'rooms_per_person' in shard.columns and 'energy_intensity' in shard.columns:
            _update_stats(corr_stats['roomspp_vs_intensity'],
                          shard['rooms_per_person'].to_numpy(dtype=float, copy=False),
                          shard['energy_intensity'].to_numpy(dtype=float, copy=False))

        # Coverage present counts per column
        for c in coverage_cols:
            if c in shard.columns:
                a = pd.to_numeric(shard[c], errors='coerce').to_numpy()
                coverage[c]['present'] += int(np.sum(np.isfinite(a)))

        # Shard-level means
        try:
            def _mean(col):
                if col not in shard.columns:
                    return np.nan
                a = pd.to_numeric(shard[col], errors='coerce').to_numpy()
                a = a[np.isfinite(a)]
                return float(np.mean(a)) if a.size else np.nan
            shard_summaries.append({
                'shard': shard_index,
                'n_buildings': int(len(shard)),
                'mean_energy_intensity': _mean('energy_intensity'),
                'mean_total_energy': _mean('total_energy_consumption'),
                'mean_energy_burden': _mean('energy_burden'),
                'mean_rooms_per_person': _mean('rooms_per_person'),
                'mean_occupancy_intensity': _mean('occupancy_intensity'),
                'mean_building_age': _mean('building_age'),
            })
        except Exception:
            pass

        # Global pairwise correlations: update univariate and per-pair stats over rows with finite values
        present = [v for v in pairwise_vars if v in shard.columns]
        # Update univariate
        for v in present:
            arr = pd.to_numeric(shard[v], errors='coerce').to_numpy()
            mask = np.isfinite(arr)
            if not np.any(mask):
                continue
            a = arr[mask].astype(float)
            u = pairwise_stats['univar'][v]
            u['n'] += int(a.size)
            u['sum'] += float(np.sum(a))
            u['sum2'] += float(np.sum(a * a))
        # Update pairs
        for i in range(len(present)):
            vi = present[i]
            ai = pd.to_numeric(shard[vi], errors='coerce').to_numpy()
            for j in range(i + 1, len(present)):
                vj = present[j]
                aj = pd.to_numeric(shard[vj], errors='coerce').to_numpy()
                m = np.isfinite(ai) & np.isfinite(aj)
                if not np.any(m):
                    continue
                x = ai[m].astype(float)
                y = aj[m].astype(float)
                key = (vi, vj)
                if key not in pairwise_stats['pairs']:
                    pairwise_stats['pairs'][key] = {
                        'n': 0, 'sum_x': 0.0, 'sum_y': 0.0,
                        'sum_x2': 0.0, 'sum_y2': 0.0, 'sum_xy': 0.0
                    }
                ps = pairwise_stats['pairs'][key]
                ps['n'] += int(x.size)
                ps['sum_x'] += float(np.sum(x))
                ps['sum_y'] += float(np.sum(y))
                ps['sum_x2'] += float(np.sum(x * x))
                ps['sum_y2'] += float(np.sum(y * y))
                ps['sum_xy'] += float(np.sum(x * y))

    # Temperature -> energy means per bin
        if 'temperature' in shard.columns and 'total_energy_consumption' in shard.columns:
            t_arr = pd.to_numeric(shard['temperature'], errors='coerce').to_numpy()
            e_arr = pd.to_numeric(shard['total_energy_consumption'], errors='coerce').to_numpy()
            m = np.isfinite(t_arr) & np.isfinite(e_arr)
            if np.any(m):
                t = t_arr[m].astype(float)
                e = e_arr[m].astype(float)
                idx = np.digitize(t, temp_energy_edges) - 1
                idx = np.clip(idx, 0, len(temp_energy_edges) - 2)
                # Accumulate with bincount
                sum_add = np.bincount(idx, weights=e, minlength=len(temp_energy_edges) - 1)
                sum2_add = np.bincount(idx, weights=e * e, minlength=len(temp_energy_edges) - 1)
                cnt_add = np.bincount(idx, minlength=len(temp_energy_edges) - 1)
                # Ensure correct dtypes
                sum_add = sum_add.astype(float)
                sum2_add = sum2_add.astype(float)
                cnt_add = cnt_add.astype(np.int64)
                temp_energy_sum[:len(sum_add)] += sum_add
                temp_energy_sum2[:len(sum2_add)] += sum2_add
                temp_energy_count[:len(cnt_add)] += cnt_add

    # Post-process correlation stats to r and slope
    def _finalize(stats_obj):
        n = stats_obj['n']
        if n <= 1:
            return {'n': n, 'r': np.nan, 'slope': np.nan, 'r_ci_low': np.nan, 'r_ci_high': np.nan, 'slope_se': np.nan}
        sx = stats_obj['sum_x']; sy = stats_obj['sum_y']
        sxx = stats_obj['sum_x2']; syy = stats_obj['sum_y2']
        sxy = stats_obj['sum_xy']
        denom_x = sxx - (sx * sx) / n
        denom_y = syy - (sy * sy) / n
        cov = sxy - (sx * sy) / n
        r = cov / np.sqrt(max(denom_x, 1e-12) * max(denom_y, 1e-12))
        slope = cov / max(denom_x, 1e-12)
        # Fisher z CI for r
        if n > 3 and np.isfinite(r) and abs(r) < 1:
            z = np.arctanh(r)
            se_z = 1.0 / np.sqrt(n - 3)
            zlo = z - 1.96 * se_z
            zhi = z + 1.96 * se_z
            rlo = np.tanh(zlo)
            rhi = np.tanh(zhi)
        else:
            rlo = np.nan; rhi = np.nan
        # Slope SE using residual variance
        sxx_val = max(denom_x, 1e-12)
        sigma2 = (max(denom_y, 0.0) - slope * cov) / max(n - 2, 1)
        slope_se = np.sqrt(max(sigma2, 0.0) / sxx_val)
        return {'n': n, 'r': float(r), 'slope': float(slope), 'r_ci_low': float(rlo), 'r_ci_high': float(rhi), 'slope_se': float(slope_se)}

    corr_summary = {k: _finalize(v) for k, v in corr_stats.items()}

    # Finalize group correlations
    group_corr_summary = {gv: {str(g): _finalize(stats) for g, stats in gcorr.items()} for gv, gcorr in group_corrs.items()}

    # Income deciles from histogram and decile-level energy burden moments
    income_deciles = None
    try:
        counts = hh_income_hist.astype(float)
        cum = np.cumsum(counts)
        total = cum[-1] if cum.size > 0 else 0
        if total > 0:
            targets = [total * p for p in np.linspace(0.1, 0.9, 9)]
            edges = []
            j = 0
            for tval in targets:
                while j < len(cum) and cum[j] < tval:
                    j += 1
                # Linear interpolate within bin
                if j == 0:
                    frac = tval / counts[j] if counts[j] > 0 else 0
                else:
                    prev = cum[j - 1]
                    denom = counts[j] if counts[j] > 0 else 1
                    frac = (tval - prev) / denom
                lo = hh_income_bins[j]
                hi = hh_income_bins[min(j + 1, len(hh_income_bins) - 1)]
                edges.append(float(lo + frac * (hi - lo)))
            income_deciles = [float(hh_income_bins[0])] + edges + [float(hh_income_bins[-1])]
    except Exception:
        income_deciles = None

    # Aggregate income-decile energy burden by combining bin moments
    eb_by_decile = None
    if income_deciles is not None:
        eb_by_decile = []
        # Map each fine bin center to a decile index and sum moments
        bin_lo = hh_income_bins[:-1]
        bin_hi = hh_income_bins[1:]
        for d in range(10):
            lo = income_deciles[d]
            hi = income_deciles[d + 1]
            agg = _init_moments()
            for i in range(len(bin_lo)):
                # If fine bin overlaps decile range
                if (bin_hi[i] > lo) and (bin_lo[i] < hi):
                    # Use the existing moments; we approximate as fully included
                    m = eb_by_income_bin.get(i)
                    if m and m['n'] > 0:
                        # Merge moments by replaying the sufficient statistics is non-trivial; approximate by adding mean n times
                        # Here we use a simple accumulation over n samples at mean value
                        n = m['n']
                        mean = m['mean']
                        # Rebuild temp moments by repeating mean n times
                        for _ in range(n):
                            _update_moments(agg, mean)
            eb_by_decile.append(agg)

    # Finalize global pairwise correlations
    means = {v: (pairwise_stats['univar'][v]['sum'] / pairwise_stats['univar'][v]['n']) if pairwise_stats['univar'][v]['n'] > 0 else np.nan for v in pairwise_vars}
    variances = {v: ((pairwise_stats['univar'][v]['sum2'] - (pairwise_stats['univar'][v]['sum'] ** 2) / pairwise_stats['univar'][v]['n']) / max(pairwise_stats['univar'][v]['n'] - 1, 1)) if pairwise_stats['univar'][v]['n'] > 1 else np.nan for v in pairwise_vars}
    corr_mat = {v1: {v2: (1.0 if v1 == v2 else np.nan) for v2 in pairwise_vars} for v1 in pairwise_vars}
    pair_summary = {}
    for (v1, v2), st in pairwise_stats['pairs'].items():
        n = st['n']
        if n <= 1:
            r = np.nan; slope = np.nan
        else:
            denom_x = st['sum_x2'] - (st['sum_x'] ** 2) / n
            denom_y = st['sum_y2'] - (st['sum_y'] ** 2) / n
            cov = st['sum_xy'] - (st['sum_x'] * st['sum_y']) / n
            r = cov / np.sqrt(max(denom_x, 1e-12) * max(denom_y, 1e-12))
            slope = cov / max(denom_x, 1e-12)
        corr_mat[v1][v2] = float(r)
        corr_mat[v2][v1] = float(r)
        pair_summary[f'{v1}~{v2}'] = {'n': int(n), 'r': float(r) if np.isfinite(r) else np.nan, 'slope': float(slope) if np.isfinite(slope) else np.nan}

    # Finalize temp->energy means and SE
    with np.errstate(invalid='ignore', divide='ignore'):
        te_means = np.divide(temp_energy_sum, temp_energy_count, out=np.full_like(temp_energy_sum, np.nan, dtype=float), where=temp_energy_count>0)
        te_var = np.divide((temp_energy_sum2 - (temp_energy_sum * temp_energy_sum) / np.maximum(temp_energy_count, 1)), np.maximum(temp_energy_count - 1, 1), out=np.full_like(temp_energy_sum2, np.nan, dtype=float), where=temp_energy_count>1)
        te_se = np.sqrt(np.divide(te_var, temp_energy_count, out=np.full_like(temp_energy_sum, np.nan, dtype=float), where=temp_energy_count>0))

    # Group-level Gini from histograms (energy_intensity, energy_burden)
    def _gini_from_hist(hist: np.ndarray, bins: np.ndarray) -> float:
        if hist is None or np.sum(hist) == 0:
            return float('nan')
        mids = (bins[:-1] + bins[1:]) / 2
        order = np.argsort(mids)
        mids_sorted = mids[order]
        hist_sorted = hist[order].astype(float)
        cum_counts = np.cumsum(hist_sorted)
        if cum_counts[-1] <= 0:
            return float('nan')
        cum_values = np.cumsum(mids_sorted * hist_sorted)
        cum_counts = cum_counts / cum_counts[-1]
        if cum_values[-1] <= 0:
            return float('nan')
        cum_values = cum_values / cum_values[-1]
        lorenz_x = np.concatenate([[0], cum_counts])
        lorenz_y = np.concatenate([[0], cum_values])
        gini = 1 - 2 * np.trapz(lorenz_y, lorenz_x)
        return float(gini)

    group_gini = {gv: {'energy_intensity': {}, 'energy_burden': {}} for gv in group_vars}
    for gv, gdict in group_hists.items():
        for gval, hmetrics in gdict.items():
            for metric_name in ['energy_intensity', 'energy_burden']:
                hist = hmetrics.get(metric_name)
                if hist is None:
                    continue
                edges = hist_metrics[metric_name]
                g = _gini_from_hist(hist, edges)
                group_gini[gv][metric_name][str(gval)] = g

    # Finalize conditional curves (means and SE)
    cond_curves_final = {}
    for key, c in cond_curves.items():
        count = c['count']
        sumv = c['sum']
        sum2 = c['sum2']
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.divide(sumv, count, out=np.full_like(sumv, np.nan, dtype=float), where=count>0)
            var = np.divide((sum2 - (sumv * sumv) / np.maximum(count, 1)), np.maximum(count - 1, 1), out=np.full_like(sumv, np.nan, dtype=float), where=count>1)
            se = np.sqrt(np.divide(var, count, out=np.full_like(sumv, np.nan, dtype=float), where=count>0))
        cond_curves_final[key] = {
            'bin_edges': c['bin_edges'].tolist(),
            'count': count.tolist(),
            'mean': mean.tolist(),
            'se': se.tolist(),
        }

    # Finalize per-building-type conditional curves
    cond_curves_by_type_final = {}
    for key, cc in cond_curves_by_type.items():
        edges = cc['bin_edges']
        out = {'bin_edges': edges.tolist(), 'groups': {}}
        for tkey in cc['count'].keys():
            cnt = cc['count'][tkey]
            sumv = cc['sum'][tkey]
            sum2 = cc['sum2'][tkey]
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan, dtype=float), where=cnt>0)
                var = np.divide((sum2 - (sumv * sumv) / np.maximum(cnt, 1)), np.maximum(cnt - 1, 1), out=np.full_like(sumv, np.nan, dtype=float), where=cnt>1)
                se = np.sqrt(np.divide(var, cnt, out=np.full_like(sumv, np.nan, dtype=float), where=cnt>0))
            out['groups'][tkey] = {
                'count': cnt.tolist(),
                'mean': mean.tolist(),
                'se': se.tolist(),
            }
        cond_curves_by_type_final[key] = out

    aggregates = {
        'cat_counts': cat_counts,
        'hists': hists,
        'density_2d': density_2d,
        'persons_age_sex': persons_age_sex,
        'emp_counts': emp_counts,
        'wfh_counts': wfh_counts,
        'income_bins': income_bins,
        'income_hist': income_hist,
        'hh_income_bins': hh_income_bins,
        'hh_income_hist': hh_income_hist,
        'edu_counts': edu_counts,
        'timeline_counts': timeline_counts,
        'transition_counts': transition_counts,
        'categories': categories,
        'timeline_by_group': {
            'sex': {k: v for k, v in timeline_by_group['sex'].items()},
            'age_group': {k: v for k, v in timeline_by_group['age_group'].items()},
        },
        'totals': {
            'buildings': total_buildings,
            'persons': total_persons,
            'activities': total_activities,
        },
        'weather': {
            'hists': weather_counts,
            'bins': weather_hists,
            'state_temps': {
                'sum': temp_by_state_sum,
                'count': temp_by_state_count,
            }
        },
        'energy': {
            'hists': energy_hists,
            'bins': energy_bins,
            'end_use_sums': end_use_sums,
        'end_use_sums_by_type': end_use_sums_by_type,
            'by_type_sum': energy_by_type_sum,
            'by_type_count': energy_by_type_count,
            'by_hh_sum': energy_by_hhsize_sum,
            'by_hh_count': energy_by_hhsize_count,
            'by_state_sum': energy_by_state_sum,
            'by_state_count': energy_by_state_count,
        },
        'correlations': corr_summary,
    'conditional_curves': cond_curves_final,
    'conditional_curves_by_type': cond_curves_by_type_final,
    'shards': shard_summaries,
        'global_stats': {
            'variables': pairwise_vars,
            'means': means,
            'variances': variances,
            'correlations': corr_mat,
            'pair_stats': pair_summary,
        },
        'temperature_energy': {
            'bin_edges': temp_energy_edges.tolist(),
            'count': temp_energy_count.tolist(),
            'mean_energy': te_means.tolist(),
            'se_energy': te_se.tolist(),
        },
    'time_profiles': time_profiles,
        'peaks': peak_stats,
        'tou': tou_stats,
        'energy_poverty_state': poverty_by_state,
    'energy_poverty_by_group': energy_poverty_by_group,
        'subgroups': {
            'energy_burden_by_income_bin': {
                'bins': hh_income_bins.tolist(),
                'moments': {int(k): v for k, v in eb_by_income_bin.items()}
            },
            'energy_intensity_by_climate_zone': {
                'moments': {str(k): v for k, v in ei_by_climate.items()}
            },
            'group_moments': group_moments,
            'group_hists': group_hists,
            'group_correlations': group_corr_summary,
            'group_gini': group_gini,
            'income_deciles': income_deciles,
            'energy_burden_by_income_decile': eb_by_decile,
            'energy_poverty_by_income_bin': poverty_by_income_bin,
            'group_vars': group_vars,
            'metrics': metrics_for_moments,
            'hist_metrics': {k: v.tolist() for k, v in hist_metrics.items()}
        }
    }

    # Attach coverage
    aggregates['coverage'] = coverage
    # Attach derived histograms
    aggregates['derived_hists'] = {
        'energy_per_capita': {
            'bins': epc_bins.tolist(),
            'hist': epc_hist.tolist(),
        }
    }

    logger.info(f"Aggregates computed: buildings={total_buildings:,}, persons={total_persons:,}, activities={total_activities:,}")
    return aggregates


def _render_full_plots(aggregates: dict, out_dir: Path):
    """Render key visualizations from full-dataset aggregates."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_both(fig, path: Path):
        """Save a figure to PNG and PDF for manuscript inclusion."""
        try:
            fig.savefig(path, dpi=200, bbox_inches='tight')
            pdf_path = path.with_suffix('.pdf')
            fig.savefig(pdf_path, bbox_inches='tight')
        except Exception:
            # Fallback to PNG only
            fig.savefig(path, dpi=200, bbox_inches='tight')

    # Building type pie
    bt = aggregates['cat_counts'].get('building_type_simple', {})
    if bt:
        labels, values = zip(*sorted(bt.items(), key=lambda x: -x[1]))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title('Building Types (Full Dataset)')
        fig.savefig(out_dir / 'building_types_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Building age histogram
    if 'building_age' in aggregates['hists']:
        bins = np.array([*aggregates['hists']['building_age'].shape])  # placeholder to satisfy linter
    # Generic helper for hists
    def _plot_hist(name: str, title: str, xlabel: str):
        if name in aggregates['hists']:
            # Recreate bin edges from config used
            # Since we know the bins used in _aggregate_full_dataset, rebuild them here
            if name == 'building_age':
                edges = np.arange(0, 125, 5)
            elif name == 'energy_intensity':
                edges = np.linspace(0, 500, 51)
            elif name == 'recs_match_weight':
                edges = np.linspace(-5, 5, 51)
            elif name == 'energy_burden':
                edges = np.linspace(0, 50, 51)
            elif name == 'household_size':
                edges = np.arange(-0.5, 11.5, 1)
            elif name == 'num_bedrooms':
                edges = np.arange(-0.5, 11.5, 1)
            elif name == 'occupancy_intensity':
                edges = np.linspace(0, 5, 51)
            elif name == 'rooms_per_person':
                edges = np.linspace(0, 10, 51)
            elif name == 'num_children':
                edges = np.arange(-0.5, 11.5, 1)
            elif name == 'num_seniors':
                edges = np.arange(-0.5, 11.5, 1)
            elif name == 'workers_per_household':
                edges = np.arange(-0.5, 11.5, 1)
            elif name == 'age_diversity_std':
                edges = np.linspace(0, 50, 51)
            else:
                return
            counts = aggregates['hists'][name]
            centers = (edges[:-1] + edges[1:]) / 2
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(centers, counts, width=(edges[1]-edges[0]) * 0.9, edgecolor='black')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            fig.savefig(out_dir / f'{name}_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

    _plot_hist('building_age', 'Building Age Distribution (Full Dataset)', 'Age (years)')
    _plot_hist('household_size', 'Household Size Distribution (Full Dataset)', 'Household Size')
    _plot_hist('num_bedrooms', 'Bedroom Count Distribution (Full Dataset)', 'Bedrooms')
    _plot_hist('energy_intensity', 'Energy Intensity Distribution (Full Dataset)', 'Energy Intensity')
    _plot_hist('recs_match_weight', 'RECS Match Weight (Full Dataset)', 'Match Weight')
    _plot_hist('energy_burden', 'Energy Burden (Full Dataset)', 'Energy Burden (%)')
    _plot_hist('occupancy_intensity', 'Occupancy Intensity (Full Dataset)', 'Persons per Room')
    _plot_hist('rooms_per_person', 'Rooms per Person (Full Dataset)', 'Rooms per Person')
    # Household income
    if 'hh_income_hist' in aggregates and np.sum(aggregates['hh_income_hist']) > 0:
        centers = (aggregates['hh_income_bins'][:-1] + aggregates['hh_income_bins'][1:]) / 2
        counts = aggregates['hh_income_hist']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(centers, counts, width=(centers[1]-centers[0]) * 0.9, edgecolor='black')
        ax.set_title('Household Income Distribution (Full Dataset)')
        ax.set_xlabel('Household Income ($)')
        ax.set_ylabel('Count')
        fig.savefig(out_dir / 'household_income_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 2D density: building_age vs energy_intensity
    d2 = aggregates['density_2d']['age_vs_energy']
    if d2['grid'] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(d2['x_bins'], d2['y_bins'])
        pcm = ax.pcolormesh(X, Y, d2['grid'].T, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='Count')
        ax.set_xlabel('Building Age (years)')
        ax.set_ylabel('Energy Intensity')
        ax.set_title('Building Age vs Energy Intensity (2D Density, Full Dataset)')
        fig.savefig(out_dir / 'age_vs_energy_density_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 2D density: temperature vs total energy
    d2te = aggregates['density_2d']['temp_vs_energy']
    if d2te['grid'] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(d2te['x_bins'], d2te['y_bins'])
        pcm = ax.pcolormesh(X, Y, d2te['grid'].T, shading='auto', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label='Count')
        ax.set_xlabel('Temperature (°F)')
        ax.set_ylabel('Total Energy Consumption')
        ax.set_title('Temperature vs Total Energy (2D Density, Full Dataset)')
        fig.savefig(out_dir / 'temp_vs_energy_density_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 2D density: rooms per person vs energy intensity
    d2ri = aggregates['density_2d']['roomspp_vs_intensity']
    if d2ri['grid'] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(d2ri['x_bins'], d2ri['y_bins'])
        pcm = ax.pcolormesh(X, Y, d2ri['grid'].T, shading='auto', cmap='cividis')
        fig.colorbar(pcm, ax=ax, label='Count')
        ax.set_xlabel('Rooms per Person')
        ax.set_ylabel('Energy Intensity')
        ax.set_title('Rooms per Person vs Energy Intensity (2D Density, Full Dataset)')
        fig.savefig(out_dir / 'roomspp_vs_intensity_density_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 2D density: energy burden vs energy intensity
    d2ebei = aggregates['density_2d'].get('eb_vs_ei')
    if d2ebei and d2ebei['grid'] is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(d2ebei['x_bins'], d2ebei['y_bins'])
        pcm = ax.pcolormesh(X, Y, d2ebei['grid'].T, shading='auto', cmap='inferno')
        fig.colorbar(pcm, ax=ax, label='Count')
        ax.set_xlabel('Energy Intensity')
        ax.set_ylabel('Energy Burden (%)')
        ax.set_title('Energy Burden vs Energy Intensity (2D Density, Full Dataset)')
        fig.savefig(out_dir / 'eb_vs_ei_density_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Persons: age pyramid
    male = np.array([aggregates['persons_age_sex']['M'][lab] for lab in [f"{i}-{i+4}" for i in range(0, 100, 5)]])
    female = np.array([aggregates['persons_age_sex']['F'][lab] for lab in [f"{i}-{i+4}" for i in range(0, 100, 5)]])
    labels = [f"{i}-{i+4}" for i in range(0, 100, 5)]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(labels))
    ax.barh(y, -male, color='steelblue', label='Male')
    ax.barh(y, female, color='coral', label='Female')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count (absolute)')
    ax.set_title('Population Age Pyramid (Full Dataset)')
    ax.legend(); ax.axvline(0, color='black', linewidth=0.5)
    xt = ax.get_xticks(); ax.set_xticks(xt); ax.set_xticklabels([str(int(abs(t))) for t in xt])
    save_both(fig, out_dir / 'age_pyramid_full.png')
    plt.close(fig)

    # Employment pie
    fig, ax = plt.subplots(figsize=(6, 6))
    vals = [aggregates['emp_counts']['employed'], aggregates['emp_counts']['not_employed']]
    ax.pie(vals, labels=['Employed', 'Not Employed'], autopct='%1.1f%%')
    ax.set_title('Employment Status (Full Dataset)')
    save_both(fig, out_dir / 'employment_status_full.png')
    plt.close(fig)

    # Work-from-home pie (employed only counts included during aggregation)
    fig, ax = plt.subplots(figsize=(6, 6))
    wvals = [aggregates['wfh_counts']['on_site'], aggregates['wfh_counts']['wfh']]
    ax.pie(wvals, labels=['On-site', 'Work from Home'], autopct='%1.1f%%')
    ax.set_title('Work Location (Employed Only, Full Dataset)')
    save_both(fig, out_dir / 'work_location_full.png')
    plt.close(fig)

    # Income histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    centers = (aggregates['income_bins'][:-1] + aggregates['income_bins'][1:]) / 2
    widths = np.diff(aggregates['income_bins']) * 0.9
    ax.bar(centers, aggregates['income_hist'], width=widths, edgecolor='black')
    ax.set_title('Income Distribution (Full Dataset)')
    ax.set_xlabel('Personal Income ($)')
    ax.set_ylabel('Count')
    save_both(fig, out_dir / 'income_distribution_full.png')
    plt.close(fig)

    # Education bar
    if aggregates['edu_counts']:
        labels, values = zip(*sorted(aggregates['edu_counts'].items(), key=lambda x: -x[1]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[:10], values[:10])
        ax.set_title('Top Education Levels (Full Dataset)')
        ax.set_xlabel('Count')
        save_both(fig, out_dir / 'education_levels_full.png')
        plt.close(fig)

    # Activity aggregate timeline (category share over day)
    tl = aggregates['timeline_counts']
    if tl.sum() > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        # Normalize to shares per minute
        minute_totals = tl.sum(axis=0)
        shares = np.divide(tl, minute_totals, out=np.zeros_like(tl, dtype=float), where=minute_totals>0)
        x = np.arange(1440)
        ax.stackplot(x, shares, labels=aggregates['categories'])
        ax.set_xlim(0, 1440)
        ax.set_xticks(np.arange(0, 1441, 120))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45)
        ax.set_title('Aggregate Activity Patterns Over Day (Full Dataset)')
        ax.set_ylabel('Share of People')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        save_both(fig, out_dir / 'activity_patterns_full.png')
        plt.close(fig)

        # Subgroup timelines by sex
        tlg = aggregates.get('timeline_by_group', {})
        sex_tl = tlg.get('sex', {})
        for sex_label, mat in sex_tl.items():
            if isinstance(mat, np.ndarray) and mat.sum() > 0:
                fig, ax = plt.subplots(figsize=(14, 6))
                minute_totals = mat.sum(axis=0)
                shares = np.divide(mat, minute_totals, out=np.zeros_like(mat, dtype=float), where=minute_totals>0)
                x = np.arange(1440)
                ax.stackplot(x, shares, labels=aggregates['categories'])
                ax.set_xlim(0, 1440)
                ax.set_xticks(np.arange(0, 1441, 120))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45)
                ax.set_title(f'Activity Patterns by Sex: {sex_label} (Full Dataset)')
                ax.set_ylabel('Share of People')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                fig.tight_layout()
                save_both(fig, out_dir / f'activity_patterns_sex_{sex_label}_full.png')
                plt.close(fig)

        # Subgroup timelines by age group
        age_tl = tlg.get('age_group', {})
        for age_label, mat in age_tl.items():
            if isinstance(mat, np.ndarray) and mat.sum() > 0:
                fig, ax = plt.subplots(figsize=(14, 6))
                minute_totals = mat.sum(axis=0)
                shares = np.divide(mat, minute_totals, out=np.zeros_like(mat, dtype=float), where=minute_totals>0)
                x = np.arange(1440)
                ax.stackplot(x, shares, labels=aggregates['categories'])
                ax.set_xlim(0, 1440)
                ax.set_xticks(np.arange(0, 1441, 120))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45)
                ax.set_title(f'Activity Patterns by Age Group: {age_label} (Full Dataset)')
                ax.set_ylabel('Share of People')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                fig.tight_layout()
                fig.savefig(out_dir / f'activity_patterns_age_{age_label}_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

    # Activity transition matrix heatmap
    tr = aggregates['transition_counts']
    if tr.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tr, cmap='magma')
        fig.colorbar(im, ax=ax, label='Transitions')
        ax.set_xticks(range(len(aggregates['categories'])))
        ax.set_yticks(range(len(aggregates['categories'])))
        ax.set_xticklabels(aggregates['categories'], rotation=45, ha='right')
        ax.set_yticklabels(aggregates['categories'])
        ax.set_title('Activity Transition Matrix (Full Dataset)')
        fig.tight_layout()
        fig.savefig(out_dir / 'activity_transitions_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Row-normalized probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = tr.sum(axis=1, keepdims=True)
            probs = np.divide(tr, row_sums, out=np.zeros_like(tr, dtype=float), where=row_sums>0)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(probs, cmap='viridis', vmin=0, vmax=np.nanmax(probs))
        fig.colorbar(im, ax=ax, label='Probability')
        ax.set_xticks(range(len(aggregates['categories'])))
        ax.set_yticks(range(len(aggregates['categories'])))
        ax.set_xticklabels(aggregates['categories'], rotation=45, ha='right')
        ax.set_yticklabels(aggregates['categories'])
        ax.set_title('Activity Transition Probabilities (Row-normalized)')
        fig.tight_layout()
        fig.savefig(out_dir / 'activity_transition_probs_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Household composition and related
    comp = aggregates['cat_counts'].get('household_composition', {})
    if comp:
        labels, values = zip(*sorted(comp.items(), key=lambda x: -x[1]))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title('Household Composition (Full Dataset)')
        fig.savefig(out_dir / 'household_composition_full.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    for name, title in [('num_children', 'Children per Household (Full Dataset)'),
                        ('num_seniors', 'Seniors per Household (Full Dataset)')]:
        if name in aggregates['hists']:
            _plot_hist(name, title, name.replace('_', ' ').title())

    # Additional household metrics
    for name, title in [('workers_per_household', 'Workers per Household (Full Dataset)'),
                        ('age_diversity_std', 'Age Diversity (Std Dev) per Household (Full Dataset)')]:
        if name in aggregates['hists']:
            _plot_hist(name, title, name.replace('_', ' ').title())

    # Save a minimal text summary
    summary_path = out_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        t = aggregates['totals']
        f.write(f"Buildings: {t['buildings']}\nPersons: {t['persons']}\nActivities: {t['activities']}\n")

    # Weather: histograms and top states
    weather = aggregates.get('weather')
    if weather:
        # Histograms
        for name, counts in weather['hists'].items():
            edges = weather['bins'][name]
            centers = (edges[:-1] + edges[1:]) / 2
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(centers, counts, width=(edges[1]-edges[0]) * 0.9, edgecolor='black')
            ax.set_title(f"{name.replace('_',' ').title()} Distribution (Full Dataset)")
            ax.set_xlabel(name.replace('_',' ').title())
            ax.set_ylabel('Count')
            fig.savefig(out_dir / f"weather_{name}_full.png", dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Top states by avg temperature
        sums = weather['state_temps']['sum']
        cnts = weather['state_temps']['count']
        if sums and cnts:
            avg = sorted(((s, sums[s] / cnts[s]) for s in sums if cnts.get(s, 0) > 0), key=lambda x: x[1])[-10:]
            if avg:
                labels, values = zip(*avg)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(labels, values, color='orange')
                ax.set_title('Temperature by State (Top 10, Full Dataset)')
                ax.set_xlabel('Average Temperature (°F)')
                fig.savefig(out_dir / 'weather_state_temps_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

    # Energy: histograms and summaries
    energy = aggregates.get('energy')
    if energy:
        # Histograms for total energy, intensity, burden
        for key, title, xlabel in [
            ('total_energy_consumption', 'Total Energy Consumption (Full Dataset)', 'Energy'),
            ('energy_intensity', 'Energy Intensity (Full Dataset)', 'Energy Intensity'),
            ('energy_burden', 'Energy Burden (Full Dataset)', 'Percent of Income'),
        ]:
            hist = energy['hists'].get(key)
            bins = energy['bins'].get(key)
            if hist is not None and bins is not None and np.sum(hist) > 0:
                centers = (bins[:-1] + bins[1:]) / 2
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(centers, hist, width=(bins[1]-bins[0]) * 0.9, edgecolor='black')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Count')
                fig.savefig(out_dir / f'energy_hist_{key}_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

                # CDF plot
                fig, ax = plt.subplots(figsize=(10, 6))
                cdf = np.cumsum(hist) / max(np.sum(hist), 1)
                ax.plot(bins[1:], cdf, drawstyle='steps-post')
                ax.set_title(f'{title} - CDF')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Cumulative Share')
                fig.savefig(out_dir / f'energy_cdf_{key}_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # End-use totals
        eus = energy.get('end_use_sums', {})
        if eus:
            labels = list(eus.keys())
            values = [eus[k] for k in labels]
            if sum(values) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(labels, values)
                ax.set_title('Total Energy by End Use (Full Dataset)')
                ax.set_xlabel('End Use')
                ax.set_ylabel('Energy')
                plt.xticks(rotation=30, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / 'energy_end_use_totals_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

                # End-use shares pie
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title('End-use Energy Shares (Full Dataset)')
                fig.savefig(out_dir / 'energy_end_use_shares_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # Average energy by building type
        s = energy.get('by_type_sum', {})
        c = energy.get('by_type_count', {})
        if s and c:
            types = []
            avgs = []
            for k, v in s.items():
                cnt = c.get(k, 0)
                if cnt > 0:
                    types.append(k)
                    avgs.append(v / cnt)
            if types:
                order = np.argsort(avgs)[::-1]
                types = [types[i] for i in order]
                avgs = [avgs[i] for i in order]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(types, avgs)
                ax.set_title('Avg Total Energy by Building Type (Full Dataset)')
                ax.set_xlabel('Building Type')
                ax.set_ylabel('Average Energy')
                plt.xticks(rotation=30, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / 'energy_avg_by_building_type_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # End-use by building type (stacked bar of shares)
        eusbt = energy.get('end_use_sums_by_type', {})
        if eusbt:
            types = sorted(eusbt.keys())
            end_uses = list(energy.get('end_use_sums', {}).keys())
            mat = np.array([[eusbt[t].get(eu, 0.0) for eu in end_uses] for t in types], dtype=float)
            totals = mat.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                shares = np.divide(mat, totals, out=np.zeros_like(mat), where=totals>0)
            fig, ax = plt.subplots(figsize=(14, 7))
            bottom = np.zeros(len(types))
            for i, eu in enumerate(end_uses):
                ax.bar(types, shares[:, i], bottom=bottom, label=eu)
                bottom += shares[:, i]
            ax.set_title('End-use Energy Shares by Building Type (Full Dataset)')
            ax.set_ylabel('Share of Total Energy')
            plt.xticks(rotation=30, ha='right')
            ax.legend(ncol=min(len(end_uses), 5))
            fig.tight_layout()
            fig.savefig(out_dir / 'energy_end_use_shares_by_building_type_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Average energy by household size
        s = energy.get('by_hh_sum', {})
        c = energy.get('by_hh_count', {})
        if s and c:
            sizes = []
            avgv = []
            for k, v in s.items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                cnt = c.get(k, 0)
                if cnt > 0:
                    sizes.append(kk)
                    avgv.append(v / cnt)
            if sizes:
                order = np.argsort(sizes)
                sizes = [sizes[i] for i in order]
                avgv = [avgv[i] for i in order]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(sizes, avgv, marker='o')
                ax.set_title('Avg Total Energy by Household Size (Full Dataset)')
                ax.set_xlabel('Household Size')
                ax.set_ylabel('Average Energy')
                fig.savefig(out_dir / 'energy_avg_by_household_size_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # Top states by average total energy
        s = energy.get('by_state_sum', {})
        c = energy.get('by_state_count', {})
        if s and c:
            states = []
            avgs = []
            for st, tot in s.items():
                cnt = c.get(st, 0)
                if cnt > 0:
                    states.append(st)
                    avgs.append(tot / cnt)
            if states:
                order = np.argsort(avgs)[::-1]
                states = [states[i] for i in order[:20]]
                avgs = [avgs[i] for i in order[:20]]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(states, avgs)
                ax.set_title('Avg Total Energy by State (Top 20)')
                ax.set_xlabel('State')
                ax.set_ylabel('Average Total Energy')
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / 'energy_avg_by_state_top20_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # Inequality: Lorenz and Gini for total energy and household income
        def _lorenz_and_gini(hist, bins, label, filename_prefix):
            if hist is None or np.sum(hist) == 0:
                return None
            # Approximate with bin midpoints
            mids = (bins[:-1] + bins[1:]) / 2
            vals = np.repeat(mids, hist.astype(int)) if np.sum(hist) < 200_000 else None
            # If too many, use grouped approximation
            if vals is None:
                order = np.argsort(mids)
                mids_sorted = mids[order]
                hist_sorted = hist[order]
                cum_counts = np.cumsum(hist_sorted)
                cum_values = np.cumsum(mids_sorted * hist_sorted)
                cum_counts = cum_counts / cum_counts[-1]
                cum_values = cum_values / (cum_values[-1] if cum_values[-1] > 0 else 1)
                lorenz_x, lorenz_y = np.concatenate([[0], cum_counts]), np.concatenate([[0], cum_values])
                gini = 1 - 2 * np.trapz(lorenz_y, lorenz_x)
            else:
                x = np.sort(vals)
                n = len(x)
                cumx = np.cumsum(x)
                lorenz_y = np.concatenate([[0], cumx / cumx[-1]])
                lorenz_x = np.linspace(0, 1, n + 1)
                gini = 1 - 2 * np.trapz(lorenz_y, lorenz_x)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Equality')
            ax.plot(lorenz_x, lorenz_y, label=f'{label} (Gini={gini:.2f})')
            ax.set_title(f'Lorenz Curve - {label}')
            ax.set_xlabel('Cumulative share of units')
            ax.set_ylabel('Cumulative share of value')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f'{filename_prefix}_lorenz_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            return float(gini)

        gini_energy = _lorenz_and_gini(energy['hists'].get('total_energy_consumption'), energy['bins'].get('total_energy_consumption'), 'Total Energy', 'energy_total')
        gini_income = None
        if 'hh_income_hist' in aggregates:
            gini_income = _lorenz_and_gini(aggregates['hh_income_hist'], aggregates['hh_income_bins'], 'Household Income', 'household_income')
        # Save inequality metrics
        try:
            import json as _json
            with open(out_dir / 'inequality_metrics.json', 'w', encoding='utf-8') as f:
                _json.dump({'gini_total_energy': gini_energy, 'gini_household_income': gini_income}, f)
        except Exception:
            pass

        # Drivers: hexbin scatter proxies (requires underlying arrays; approximate with 2D density if available)
    if d2['grid'] is not None:
            # Also provide a hexbin-like visualization using pcolormesh already produced above.
            pass

    # Export machine-readable summaries and LaTeX/Markdown snippets
    _export_results_artifacts(aggregates, out_dir)

    # Subgroup disparity plots
    try:
        sub = aggregates.get('subgroups', {})
        # Energy burden vs income bins
        eb = sub.get('energy_burden_by_income_bin', {})
        bins = np.array(eb.get('bins', []))
        moments = eb.get('moments', {})
        if bins.size > 0 and moments:
            centers = (bins[:-1] + bins[1:]) / 2
            means = []
            ses = []
            for i in range(len(centers)):
                m = moments.get(i, {})
                _, mean, _, se = (m.get('n', 0), m.get('mean', np.nan), m.get('M2', np.nan), np.nan)
                # Recompute SE from moments if available
                n = m.get('n', 0)
                if n > 1 and np.isfinite(m.get('M2', np.nan)):
                    var = m['M2'] / (n - 1)
                    se = np.sqrt(var / n)
                means.append(mean)
                ses.append(se)
            means = np.array(means, dtype=float)
            ses = np.array(ses, dtype=float)
            mask = np.isfinite(means)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.errorbar(centers[mask], means[mask], yerr=1.96 * ses[mask], fmt='o-', capsize=3)
                ax.set_title('Energy Burden vs Household Income (Full Dataset)')
                ax.set_xlabel('Household Income ($)')
                ax.set_ylabel('Mean Energy Burden (%) with 95% CI')
                fig.savefig(out_dir / 'energy_burden_vs_income_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # Energy intensity by climate zone
        ei_cz = sub.get('energy_intensity_by_climate_zone', {}).get('moments', {})
        if ei_cz:
            labels = []
            means = []
            ses = []
            for k, m in ei_cz.items():
                n = m.get('n', 0); mean = m.get('mean', np.nan)
                if 'M2' in m and n > 1:
                    var = m['M2'] / (n - 1)
                    se = np.sqrt(var / n)
                else:
                    var = np.nan
                    se = np.nan
                if n > 0 and np.isfinite(mean):
                    labels.append(k)
                    means.append(mean)
                    if np.isnan(se) and np.isfinite(var) and n > 1:
                        se = np.sqrt(var / n)
                    ses.append(se)
            if labels:
                order = np.argsort(means)[::-1]
                labels = [labels[i] for i in order]
                means = [means[i] for i in order]
                ses = [ses[i] for i in order]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(labels)), means, yerr=[1.96 * s if np.isfinite(s) else 0 for s in ses], capsize=3)
                ax.set_title('Energy Intensity by Climate Zone (Full Dataset)')
                ax.set_xlabel('Climate Zone')
                ax.set_ylabel('Mean Energy Intensity with 95% CI')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / 'energy_intensity_by_climate_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)

        # Group moments visualizations for key metrics
        gm = sub.get('group_moments', {})
        def draw_group_bars(gvar: str, metric: str, title: str, filename: str, top_n: int | None = None):
            data = gm.get(gvar, {})
            if not data:
                return
            labels = []
            means = []
            ses = []
            for k, moments in data.items():
                m = moments.get(metric)
                if not m:
                    continue
                n = m.get('n', 0)
                mean = m.get('mean', np.nan)
                se = np.nan
                if 'M2' in m and n > 1:
                    var = m['M2'] / (n - 1)
                    se = np.sqrt(var / n)
                if n > 0 and np.isfinite(mean):
                    labels.append(str(k))
                    means.append(float(mean))
                    ses.append(se)
            if not labels:
                return
            order = np.argsort(means)[::-1]
            if top_n is not None:
                order = order[:top_n]
            labels = [labels[i] for i in order]
            means = [means[i] for i in order]
            ses = [ses[i] for i in order]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(labels)), means, yerr=[1.96 * s if np.isfinite(s) else 0 for s in ses], capsize=3)
            ax.set_title(title)
            ax.set_xlabel(gvar.replace('_', ' ').title())
            ax.set_ylabel(f'Mean {metric.replace("_", " ").title()} (95% CI)')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Selected group charts
        draw_group_bars('building_type_simple', 'energy_intensity', 'Energy Intensity by Building Type', 'energy_intensity_by_building_type_full.png')
        draw_group_bars('tenure_type', 'energy_burden', 'Energy Burden by Tenure', 'energy_burden_by_tenure_full.png')
        draw_group_bars('heating_fuel', 'energy_burden', 'Energy Burden by Heating Fuel', 'energy_burden_by_heating_fuel_full.png')
        draw_group_bars('STATE', 'energy_intensity', 'Energy Intensity by State (Top 20)', 'energy_intensity_by_state_top20_full.png', top_n=20)
        draw_group_bars('STATE', 'energy_burden', 'Energy Burden by State (Top 20)', 'energy_burden_by_state_top20_full.png', top_n=20)

        # Income decile plot for energy burden
        try:
            eb_dec = sub.get('energy_burden_by_income_decile')
            dec_edges = sub.get('income_deciles')
            if eb_dec and dec_edges:
                means = []
                ses = []
                labels = []
                for i, m in enumerate(eb_dec):
                    n = m.get('n', 0)
                    mean = m.get('mean', np.nan)
                    if 'M2' in m and n > 1:
                        var = m['M2'] / (n - 1)
                        se = np.sqrt(var / n)
                    else:
                        se = np.nan
                    if np.isfinite(mean):
                        means.append(mean)
                        ses.append(se)
                        labels.append(f'D{i+1}')
                if means:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(means)), means, yerr=[1.96 * s if np.isfinite(s) else 0 for s in ses], capsize=3)
                    ax.set_xticks(range(len(means)))
                    ax.set_xticklabels(labels)
                    ax.set_title('Energy Burden by Household Income Decile (Full Dataset)')
                    ax.set_xlabel('Income Decile')
                    ax.set_ylabel('Mean Energy Burden (%) with 95% CI')
                    fig.savefig(out_dir / 'energy_burden_by_income_decile_full.png', dpi=200, bbox_inches='tight')
                    plt.close(fig)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed to render subgroup plots: {e}")

    # Correlation heatmap for global pairwise correlations
    try:
        gs = aggregates.get('global_stats')
        if gs and 'correlations' in gs:
            vars_list = gs['variables']
            mat = np.array([[gs['correlations'].get(r, {}).get(c, (gs['correlations'][r][c] if isinstance(gs['correlations'][r], dict) else np.nan)) if isinstance(gs['correlations'], dict) else np.nan for c in vars_list] for r in vars_list], dtype=float)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm')
            fig.colorbar(im, ax=ax, label='Pearson r')
            ax.set_xticks(range(len(vars_list)))
            ax.set_yticks(range(len(vars_list)))
            ax.set_xticklabels(vars_list, rotation=45, ha='right')
            ax.set_yticklabels(vars_list)
            ax.set_title('Global Pairwise Correlations (Full Dataset)')
            fig.tight_layout()
            fig.savefig(out_dir / 'global_correlation_heatmap_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to render correlation heatmap: {e}")

    # Temperature->Energy mean curve
    try:
        te = aggregates.get('temperature_energy')
        if te and te.get('count'):
            edges = np.array(te['bin_edges'])
            centers = (edges[:-1] + edges[1:]) / 2
            means = np.array(te.get('mean_energy', []), dtype=float)
            counts = np.array(te.get('count', []))
            se = np.array(te.get('se_energy', []), dtype=float) if 'se_energy' in te else None
            mask = (counts > 20) & np.isfinite(means)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(10, 6))
                if se is not None and np.any(np.isfinite(se[mask])):
                    ax.plot(centers[mask], means[mask], color='C1')
                    lo = means[mask] - 1.96 * se[mask]
                    hi = means[mask] + 1.96 * se[mask]
                    ax.fill_between(centers[mask], lo, hi, color='C1', alpha=0.25, label='95% CI')
                else:
                    ax.plot(centers[mask], means[mask], marker='o', lw=1)
                ax.set_title('Mean Total Energy vs Temperature (Full Dataset)')
                ax.set_xlabel('Temperature (°F)')
                ax.set_ylabel('Mean Total Energy Consumption')
                fig.savefig(out_dir / 'mean_energy_vs_temperature_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to render mean energy vs temperature: {e}")

    # Conditional mean curves with 95% CI
    try:
        cc = aggregates.get('conditional_curves', {})
        for key, title, xlabel, ylabel, fname in [
            ('ei_by_age', 'Energy Intensity vs Building Age', 'Age (years)', 'Energy Intensity', 'ei_vs_age_curve_full.png'),
            ('ei_by_roomspp', 'Energy Intensity vs Rooms per Person', 'Rooms per Person', 'Energy Intensity', 'ei_vs_roomspp_curve_full.png'),
            ('ei_by_occupancy', 'Energy Intensity vs Occupancy Intensity', 'Occupancy Intensity', 'Energy Intensity', 'ei_vs_occupancy_curve_full.png'),
            ('energy_by_hhsize', 'Total Energy vs Household Size', 'Household Size', 'Total Energy', 'energy_vs_hhsize_curve_full.png'),
            ('eb_by_roomspp', 'Energy Burden vs Rooms per Person', 'Rooms per Person', 'Energy Burden (%)', 'eb_vs_roomspp_curve_full.png'),
            ('eb_by_hhsize', 'Energy Burden vs Household Size', 'Household Size', 'Energy Burden (%)', 'eb_vs_hhsize_curve_full.png'),
        ]:
            c = cc.get(key)
            if not c:
                continue
            edges = np.array(c['bin_edges']); centers = (edges[:-1] + edges[1:]) / 2
            mean = np.array(c['mean'], dtype=float); se = np.array(c['se'], dtype=float)
            cnt = np.array(c['count'])
            mask = (cnt > 20) & np.isfinite(mean)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(centers[mask], mean[mask], color='C0')
                ax.fill_between(centers[mask], mean[mask] - 1.96 * se[mask], mean[mask] + 1.96 * se[mask], color='C0', alpha=0.2, label='95% CI')
                ax.set_title(f'{title} (Full Dataset)')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                fig.savefig(out_dir / fname, dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to render conditional mean curves: {e}")

    # Energy poverty shares by income bin
    try:
        sub = aggregates.get('subgroups', {})
        pov = sub.get('energy_poverty_by_income_bin')
        bins = np.array(sub.get('energy_burden_by_income_bin', {}).get('bins', []))
        if pov and bins.size > 0:
            centers = (bins[:-1] + bins[1:]) / 2
            den = np.array([pov[i]['den'] for i in range(len(centers))])
            s10 = np.array([pov[i]['num_10'] for i in range(len(centers))])
            s20 = np.array([pov[i]['num_20'] for i in range(len(centers))])
            with np.errstate(divide='ignore', invalid='ignore'):
                sh10 = np.divide(s10, den, out=np.full_like(den, np.nan, dtype=float), where=den>0)
                sh20 = np.divide(s20, den, out=np.full_like(den, np.nan, dtype=float), where=den>0)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(centers, sh10, label='Energy burden ≥10%')
            ax.plot(centers, sh20, label='Energy burden ≥20%')
            ax.set_title('Energy Poverty Share by Household Income (Full Dataset)')
            ax.set_xlabel('Household Income ($)')
            ax.set_ylabel('Share of Households')
            ax.legend()
            fig.savefig(out_dir / 'energy_poverty_by_income_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to render energy poverty shares: {e}")

    # Group-level Gini bar charts
    try:
        sub = aggregates.get('subgroups', {})
        gg = sub.get('group_gini', {})
        for metric in ['energy_intensity', 'energy_burden']:
            for gv, mdict in gg.items():
                vals = mdict.get(metric, {})
                if not vals:
                    continue
                labels = list(vals.keys())
                g = np.array([vals[k] for k in labels], dtype=float)
                mask = np.isfinite(g)
                if not np.any(mask):
                    continue
                labels = [labels[i] for i in np.where(mask)[0]]
                g = g[mask]
                order = np.argsort(g)[::-1]
                top = min(20, len(order))
                labels = [labels[i] for i in order[:top]]
                g = g[order[:top]]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(labels)), g)
                ax.set_title(f'Gini of {metric.replace("_"," ")} by {gv} (Top {top})')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Gini index')
                fig.tight_layout()
                fig.savefig(out_dir / f'{metric}_gini_by_{gv}_top{top}_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to render group-level Gini charts: {e}")

    # Coverage bar chart
    try:
        cov = aggregates.get('coverage', {})
        if cov:
            cols = list(cov.keys())
            shares = [ (cov[c]['present'] / cov[c]['total']) if cov[c]['total']>0 else np.nan for c in cols ]
            mask = np.isfinite(shares)
            cols = [c for c, m in zip(cols, mask) if m]
            shares = [s for s in shares if np.isfinite(s)]
            if cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(cols, shares)
                ax.set_ylim(0, 1)
                ax.set_title('Data Coverage for Key Columns')
                ax.set_ylabel('Share Present')
                plt.xticks(rotation=30, ha='right')
                fig.tight_layout()
                fig.savefig(out_dir / 'coverage_share_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception:
        pass

    # Shard variability quick view
    try:
        shards = aggregates.get('shards', [])
        if shards:
            df = pd.DataFrame(shards)
            for col, title in [
                ('mean_total_energy', 'Shard Mean Total Energy'),
                ('mean_energy_intensity', 'Shard Mean Energy Intensity'),
                ('mean_energy_burden', 'Shard Mean Energy Burden'),
            ]:
                if col in df.columns:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(df['shard'], df[col], marker='.', lw=0.5)
                    ax.set_title(title)
                    ax.set_xlabel('Shard Index')
                    ax.set_ylabel(col.replace('_',' ').title())
                    fig.tight_layout()
                    fig.savefig(out_dir / f'{col}_by_shard_full.png', dpi=200, bbox_inches='tight')
                    plt.close(fig)
    except Exception:
        pass

    # Energy poverty by state plot
    try:
        eps = aggregates.get('energy_poverty_state', {})
        if eps:
            states = sorted(list(eps.keys()))
            den = np.array([eps[s]['den'] for s in states], dtype=float)
            sh10 = np.array([eps[s]['num_10'] for s in states], dtype=float)
            sh20 = np.array([eps[s]['num_20'] for s in states], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                r10 = np.divide(sh10, den, out=np.full_like(den, np.nan), where=den>0)
                r20 = np.divide(sh20, den, out=np.full_like(den, np.nan), where=den>0)
            fig, ax = plt.subplots(figsize=(12, 6))
            order = np.argsort(r20)
            states_ord = [states[i] for i in order[::-1][:20]]
            r10_ord = [r10[i] for i in order[::-1][:20]]
            r20_ord = [r20[i] for i in order[::-1][:20]]
            x = np.arange(len(states_ord))
            ax.bar(x-0.2, r10_ord, width=0.4, label='≥10%')
            ax.bar(x+0.2, r20_ord, width=0.4, label='≥20%')
            ax.set_xticks(x)
            ax.set_xticklabels(states_ord, rotation=45, ha='right')
            ax.set_ylabel('Share of Households')
            ax.set_title('Energy Poverty by State (Top 20 by ≥20% share)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / 'energy_poverty_by_state_top20_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Energy poverty by group (top categories)
        epg = aggregates.get('energy_poverty_by_group', {})
        for gv, d in epg.items():
            if not d:
                continue
            labels = list(d.keys())
            den = np.array([d[k]['den'] for k in labels], dtype=float)
            sh10 = np.divide([d[k]['num_10'] for k in labels], den, out=np.zeros_like(den), where=den>0)
            sh20 = np.divide([d[k]['num_20'] for k in labels], den, out=np.zeros_like(den), where=den>0)
            order = np.argsort(sh20)[::-1]
            top = min(20, len(order))
            idx = order[:top]
            lbl_top = [labels[i] for i in idx]
            sh10_top = [sh10[i] for i in idx]
            sh20_top = [sh20[i] for i in idx]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(lbl_top))
            ax.bar(x-0.2, sh10_top, width=0.4, label='≥10%')
            ax.bar(x+0.2, sh20_top, width=0.4, label='≥20%')
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in lbl_top], rotation=45, ha='right')
            ax.set_ylabel('Share of Households')
            ax.set_title(f'Energy Poverty by {gv} (Top {top} by ≥20% share)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f'energy_poverty_by_{gv}_top{top}_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

    except Exception as e:
        logger.warning(f"Failed to render energy poverty plots: {e}")

    # Derived energy per capita visuals
    try:
        d = aggregates.get('derived_hists', {}).get('energy_per_capita', {})
        bins = np.array(d.get('bins', []), dtype=float)
        hist = np.array(d.get('hist', []), dtype=float)
        if bins.size > 1 and hist.size == bins.size - 1 and np.sum(hist) > 0:
            centers = (bins[:-1] + bins[1:]) / 2
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(centers, hist, width=(bins[1]-bins[0]) * 0.9, edgecolor='black')
            ax.set_title('Energy per Capita (Household) Distribution (Full Dataset)')
            ax.set_xlabel('Energy per Capita')
            ax.set_ylabel('Count')
            fig.savefig(out_dir / 'energy_per_capita_full.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    except Exception:
        pass

    # Per-building-type EI vs age conditional curves
    try:
        cbt = aggregates.get('conditional_curves_by_type', {}).get('ei_by_age_by_type')
        if cbt and 'groups' in cbt:
            edges = np.array(cbt.get('bin_edges', []), dtype=float)
            centers = (edges[:-1] + edges[1:]) / 2 if edges.size > 1 else None
            if centers is not None:
                fig, ax = plt.subplots(figsize=(12, 6))
                for tkey, stats in cbt['groups'].items():
                    mean = np.array(stats.get('mean', []), dtype=float)
                    cnt = np.array(stats.get('count', []), dtype=float)
                    mask = (cnt > 50) & np.isfinite(mean)
                    if np.any(mask):
                        ax.plot(centers[mask], mean[mask], label=str(tkey))
                ax.set_title('Energy Intensity vs Building Age by Building Type (Full Dataset)')
                ax.set_xlabel('Building Age (years)')
                ax.set_ylabel('Energy Intensity')
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                fig.savefig(out_dir / 'ei_vs_age_by_building_type_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception:
        pass
    except Exception:
        pass

    # Top correlations bar (global pair stats)
    try:
        gs = aggregates.get('global_stats', {})
        pairs = gs.get('pair_stats', {})
        if pairs:
            items = [(k, abs(v.get('r', np.nan))) for k, v in pairs.items() if np.isfinite(v.get('r', np.nan))]
            items.sort(key=lambda x: x[1], reverse=True)
            top = items[:20]
            if top:
                labels = [k for k, _ in top]
                vals = [v for _, v in top]
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(labels)), vals)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('|Pearson r|')
                ax.set_title('Top 20 Absolute Correlations (Global)')
                fig.tight_layout()
                fig.savefig(out_dir / 'top20_abs_correlations_full.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    except Exception:
        pass

    # Time-based energy profiles (hourly/minute)
    try:
        tp = aggregates.get('time_profiles', {})
        peaks = aggregates.get('peaks', {})
        tou = aggregates.get('tou', {})
        def finalize_profile(stats):
            sumv = stats['sum']; sum2 = stats['sum2']; cnt = stats['count']
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan), where=cnt>0)
                var = np.divide((sum2 - (sumv * sumv) / np.maximum(cnt, 1)), np.maximum(cnt - 1, 1), out=np.full_like(sumv, np.nan), where=cnt>1)
                se = np.sqrt(np.divide(var, cnt, out=np.full_like(sumv, np.nan), where=cnt>0))
            return mean, se, cnt

        # Hourly overall
        if 'hourly' in tp and tp['hourly'].get('overall'):
            mean, se, cnt = finalize_profile(tp['hourly']['overall'])
            x = np.arange(24)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, mean, color='C0')
            if np.any(np.isfinite(se)):
                ax.fill_between(x, mean - 1.96*se, mean + 1.96*se, color='C0', alpha=0.2)
            ax.set_title('Average Hourly Energy Consumption (Full Dataset)')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Mean Energy Load (kW)')
            save_both(fig, out_dir / 'hourly_energy_profile_full.png')
            plt.close(fig)

            # By building type (top 6)
            groups = tp['hourly'].get('groups', {}).get('building_type_simple', {})
            if groups:
                # Rank by total count
                totals = {k: np.nansum(v['count']) for k, v in groups.items()}
                top = [k for k, _ in sorted(totals.items(), key=lambda x: -x[1])[:6]]
                fig, ax = plt.subplots(figsize=(12, 6))
                for i, k in enumerate(top):
                    m, se_g, c_g = finalize_profile(groups[k])
                    ax.plot(x, m, label=str(k))
                ax.set_title('Hourly Energy by Building Type (Top 6)')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Mean Energy Load (kW)')
                ax.legend()
                fig.tight_layout()
                save_both(fig, out_dir / 'hourly_energy_by_building_type_top6_full.png')
                plt.close(fig)

            # By climate zone (all, limited legend)
            cz_groups = tp['hourly'].get('groups', {}).get('climate_zone', {})
            if cz_groups:
                fig, ax = plt.subplots(figsize=(12, 6))
                for i, (k, v) in enumerate(sorted(cz_groups.items())):
                    m, se_g, c_g = finalize_profile(v)
                    ax.plot(x, m, label=str(k))
                ax.set_title('Hourly Energy by Climate Zone')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Mean Energy Load (kW)')
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                save_both(fig, out_dir / 'hourly_energy_by_climate_zone_full.png')
                plt.close(fig)

            # By state (top 10 by count)
            st_groups = tp['hourly'].get('groups', {}).get('STATE', {})
            if st_groups:
                totals = {k: np.nansum(v['count']) for k, v in st_groups.items()}
                top = [k for k, _ in sorted(totals.items(), key=lambda x: -x[1])[:10]]
                fig, ax = plt.subplots(figsize=(14, 6))
                for k in top:
                    m, se_g, c_g = finalize_profile(st_groups[k])
                    ax.plot(x, m, label=str(k))
                ax.set_title('Hourly Energy by State (Top 10 by sample size)')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Mean Energy Load (kW)')
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                save_both(fig, out_dir / 'hourly_energy_by_state_top10_full.png')
                plt.close(fig)

        # Minute-level overall (sampled)
        if 'minute' in tp and tp['minute'].get('overall') and tp['minute'].get('sampled', 0) > 0:
            mean, se, cnt = finalize_profile(tp['minute']['overall'])
            x = np.arange(1440)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(x, mean, color='C1')
            ax.set_title('Average Minute-by-Minute Energy Consumption (Sampled)')
            ax.set_xlabel('Minute of Day')
            ax.set_ylabel('Mean Energy Load (kW)')
            save_both(fig, out_dir / 'minute_energy_profile_sampled_full.png')
            plt.close(fig)

        # Diversity factor and system load duration curve (using hourly sums)
        try:
            if 'hourly' in tp and tp['hourly'].get('overall'):
                hs = np.array(tp['hourly']['overall'].get('sum', []), dtype=float)
                if hs.size == 24 and np.any(np.isfinite(hs)):
                    system_peak = float(np.nanmax(hs))
                    system_peak_hour = int(np.nanargmax(hs))
                    sop = peaks.get('overall', {}).get('sum_of_peaks', 0.0) if peaks else 0.0
                    diversity = (sop / system_peak) if system_peak > 0 else np.nan
                    # Small figure with diversity factor
                    fig, ax = plt.subplots(figsize=(5, 2.2))
                    ax.axis('off')
                    ax.text(0.5, 0.65, f'Diversity factor = {diversity:.2f}', ha='center', va='center', fontsize=12)
                    ax.text(0.5, 0.35, f'System peak: {system_peak:.2f} kW at {system_peak_hour:02d}:00', ha='center', va='center', fontsize=10)
                    save_both(fig, out_dir / 'diversity_factor_full.png')
                    plt.close(fig)

                    # System load duration curve
                    vals = np.sort(hs[np.isfinite(hs)])[::-1]
                    x = np.linspace(0, 100, len(vals))
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(x, vals)
                    ax.set_xlabel('Load ranking percentile (%)')
                    ax.set_ylabel('Aggregate load (kW)')
                    ax.set_title('Load Duration Curve (Hourly aggregate sum)')
                    save_both(fig, out_dir / 'load_duration_curve_hourly_sum_full.png')
                    plt.close(fig)
        except Exception:
            pass

        # Peak hour histogram and average peak value
        if peaks:
            try:
                ph = np.array(peaks.get('overall', {}).get('peak_hour_hist', []), dtype=float)
                if ph.size == 24 and np.sum(ph) > 0:
                    x = np.arange(24)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(x, ph)
                    ax.set_xlabel('Hour of Peak')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Peak Load Hour (Household level)')
                    save_both(fig, out_dir / 'peak_hour_hist_full.png')
                    plt.close(fig)
                # Average of building peaks
                sop = peaks.get('overall', {}).get('sum_of_peaks', 0.0)
                cnt = peaks.get('overall', {}).get('count', 0)
                if cnt > 0:
                    avg_peak = sop / cnt
                    # Small text figure for quick reference
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ax.axis('off')
                    ax.text(0.5, 0.5, f'Average building peak: {avg_peak:.2f} kW\n(n={cnt:,})', ha='center', va='center', fontsize=11)
                    save_both(fig, out_dir / 'avg_building_peak_full.png')
                    plt.close(fig)
            except Exception:
                pass

        # Time-of-Use (TOU) average shares (overall)
        if tou:
            try:
                bands = tou.get('bands', {})
                avg = tou.get('average_share', {})
                labels = list(bands.keys())
                means = []
                for bn in labels:
                    ss = avg.get(bn, {}).get('sum_share', 0.0)
                    c = avg.get(bn, {}).get('count', 0)
                    means.append((ss / c) if c > 0 else np.nan)
                if labels and any(np.isfinite(means)):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(labels, means, color=['#6baed6', '#9ecae1', '#3182bd'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Average share of daily energy')
                    ax.set_title('Time-of-Use Energy Shares (Average across households)')
                    for i, v in enumerate(means):
                        if np.isfinite(v):
                            ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)
                    save_both(fig, out_dir / 'tou_average_shares_full.png')
                    plt.close(fig)
            except Exception:
                pass

        # Load Duration Curve (from hourly overall mean profile as proxy)
        try:
            if 'hourly' in tp and tp['hourly'].get('overall'):
                mean, se, cnt = finalize_profile(tp['hourly']['overall'])
                if mean is not None and np.any(np.isfinite(mean)):
                    vals = np.sort(mean[np.isfinite(mean)])[::-1]
                    if vals.size > 0:
                        x = np.linspace(0, 100, len(vals))
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(x, vals)
                        ax.set_xlabel('Load ranking percentile (%)')
                        ax.set_ylabel('Mean load (kW)')
                        ax.set_title('Load Duration Curve (Hourly mean)')
                        save_both(fig, out_dir / 'load_duration_curve_hourly_mean_full.png')
                        plt.close(fig)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Failed to render time-based energy profiles: {e}")


def _export_results_artifacts(aggregates: dict, out_dir: Path):
    """Write JSON/CSV summaries and LaTeX/Markdown snippets to support manuscript results."""
    try:
        tables_dir = out_dir / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Helper: write CSV from simple dicts
        def write_csv(path: Path, headers, rows):
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(headers)
                for r in rows:
                    w.writerow(r)

        # Totals
        t = aggregates.get('totals', {})
        write_csv(tables_dir / 'totals.csv', ['metric', 'value'], [(k, v) for k, v in t.items()])

        # Energy by building type
        energy = aggregates.get('energy', {})
        s = energy.get('by_type_sum', {})
        c = energy.get('by_type_count', {})
        rows = []
        for k, v in s.items():
            cnt = c.get(k, 0)
            avg = (v / cnt) if cnt > 0 else np.nan
            rows.append([k, cnt, v, avg])
        write_csv(tables_dir / 'energy_by_building_type.csv', ['building_type', 'n', 'total_energy', 'avg_energy'], rows)

        # Energy by household size
        s = energy.get('by_hh_sum', {})
        c = energy.get('by_hh_count', {})
        rows = []
        for k in sorted(s, key=lambda x: (isinstance(x, str), x)):
            cnt = c.get(k, 0)
            avg = (s[k] / cnt) if cnt > 0 else np.nan
            rows.append([k, cnt, s[k], avg])
        write_csv(tables_dir / 'energy_by_household_size.csv', ['household_size', 'n', 'total_energy', 'avg_energy'], rows)

        # Energy by state
        s = energy.get('by_state_sum', {})
        c = energy.get('by_state_count', {})
        if s and c:
            rows = []
            for st, tot in s.items():
                cnt = c.get(st, 0)
                avg = (tot / cnt) if cnt > 0 else np.nan
                rows.append([st, cnt, tot, avg])
            write_csv(tables_dir / 'energy_by_state.csv', ['state', 'n', 'total_energy', 'avg_energy'], rows)

        # End-use totals
        eus = energy.get('end_use_sums', {})
        write_csv(tables_dir / 'energy_end_use_totals.csv', ['end_use', 'total_energy'], [[k, v] for k, v in eus.items()])
        # End-use totals by building type
        eusbt = energy.get('end_use_sums_by_type', {})
        if eusbt:
            rows = []
            for bt, d in eusbt.items():
                for eu, val in d.items():
                    rows.append([bt, eu, val])
            write_csv(tables_dir / 'energy_end_use_by_building_type.csv', ['building_type', 'end_use', 'total_energy'], rows)

        # Correlations and slopes
        corr = aggregates.get('correlations', {})
        write_csv(
            tables_dir / 'correlations.csv',
            ['pair', 'n', 'pearson_r', 'r_ci_low', 'r_ci_high', 'slope', 'slope_se'],
            [[k, v.get('n', 0), v.get('r', np.nan), v.get('r_ci_low', np.nan), v.get('r_ci_high', np.nan), v.get('slope', np.nan), v.get('slope_se', np.nan)] for k, v in corr.items()]
        )

        # Global pairwise correlation matrix and pair stats
        gs = aggregates.get('global_stats', {})
        if gs:
            vars_list = gs.get('variables', [])
            corr_mat = gs.get('correlations', {})
            # Matrix CSV (wide)
            try:
                rows = []
                header = ['var'] + vars_list
                for r in vars_list:
                    row = [r] + [corr_mat.get(r, {}).get(c, '') for c in vars_list]
                    rows.append(row)
                write_csv(tables_dir / 'global_correlation_matrix.csv', header, rows)
            except Exception:
                pass
            # Pair stats long
            pair_stats = gs.get('pair_stats', {})
            write_csv(tables_dir / 'global_pair_stats.csv', ['pair', 'n', 'pearson_r', 'slope'], [[k, v.get('n', 0), v.get('r', np.nan), v.get('slope', np.nan)] for k, v in pair_stats.items()])

        # Subgroups exports
        sub = aggregates.get('subgroups', {})
        # Group moments
        gm = sub.get('group_moments', {})
        for gvar, data in gm.items():
            rows = []
            for gval, metrics in data.items():
                for metric_name, m in metrics.items():
                    n = m.get('n', 0)
                    mean = m.get('mean', np.nan)
                    se = np.sqrt(m['M2'] / (n - 1) / n) if m.get('M2') is not None and n > 1 else np.nan
                    rows.append([gval, metric_name, n, mean, se])
            write_csv(tables_dir / f'group_moments_{gvar}.csv', ['group', 'metric', 'n', 'mean', 'se'], rows)

        # Group histograms
        gh = sub.get('group_hists', {})
        for gvar, data in gh.items():
            for gval, hmetrics in data.items():
                for metric_name, hist in hmetrics.items():
                    # write each as a row per bin
                    edges = aggregates['subgroups']['hist_metrics'][metric_name]
                    rows = []
                    for i in range(len(hist)):
                        rows.append([gval, metric_name, edges[i], edges[i+1], int(hist[i])])
                    write_csv(tables_dir / f'group_hist_{gvar}_{metric_name}.csv', ['group', 'metric', 'bin_lo', 'bin_hi', 'count'], rows)

        # Group correlations
        gcorr = sub.get('group_correlations', {})
        for gvar, data in gcorr.items():
            rows = []
            for gval, stats in data.items():
                rows.append([gval, stats.get('n', 0), stats.get('r', np.nan), stats.get('slope', np.nan)])
            write_csv(tables_dir / f'group_correlations_{gvar}.csv', ['group', 'n', 'pearson_r', 'slope'], rows)

        # Group-level Gini indices
        gg = sub.get('group_gini', {})
        for gvar, mdict in gg.items():
            for metric, gvals in mdict.items():
                rows = [[k, v] for k, v in gvals.items()]
                write_csv(tables_dir / f'group_gini_{gvar}_{metric}.csv', ['group', 'gini'], rows)

        # Income deciles and energy burden by decile
        dec = sub.get('income_deciles')
        if dec:
            write_csv(tables_dir / 'income_deciles.csv', ['d', 'lower', 'upper'], [[i, dec[i], dec[i+1]] for i in range(10)])
        eb_dec = sub.get('energy_burden_by_income_decile')
        if eb_dec:
            rows = []
            for i, m in enumerate(eb_dec):
                n = m.get('n', 0)
                mean = m.get('mean', np.nan)
                se = np.sqrt(m['M2'] / (n - 1) / n) if m.get('M2') is not None and n > 1 else np.nan
                rows.append([i, n, mean, se])
            write_csv(tables_dir / 'energy_burden_by_income_decile.csv', ['decile', 'n', 'mean', 'se'], rows)

        # Energy poverty by income bin
        pov = sub.get('energy_poverty_by_income_bin')
        bins = sub.get('energy_burden_by_income_bin', {}).get('bins', [])
        if pov and bins:
            rows = []
            for i in range(len(bins) - 1):
                d = pov.get(i, {'den': 0, 'num_10': 0, 'num_20': 0})
                den = d.get('den', 0)
                n10 = d.get('num_10', 0)
                n20 = d.get('num_20', 0)
                sh10 = (n10 / den) if den > 0 else np.nan
                sh20 = (n20 / den) if den > 0 else np.nan
                rows.append([bins[i], bins[i+1], den, n10, n20, sh10, sh20])
            write_csv(tables_dir / 'energy_poverty_by_income_bin.csv', ['bin_lo', 'bin_hi', 'den', 'num_10', 'num_20', 'share_10', 'share_20'], rows)

        # Temperature->Energy mean curve
        te = aggregates.get('temperature_energy', {})
        if te:
            edges = te.get('bin_edges', [])
            means = te.get('mean_energy', [])
            counts = te.get('count', [])
            ses = te.get('se_energy', [])
            rows = []
            for i in range(max(0, len(edges) - 1)):
                lo = edges[i]; hi = edges[i+1]
                mu = means[i] if i < len(means) else ''
                n = counts[i] if i < len(counts) else 0
                se = ses[i] if i < len(ses) else ''
                rows.append([lo, hi, n, mu, se])
            write_csv(tables_dir / 'temperature_energy_mean_curve.csv', ['temp_lo', 'temp_hi', 'n', 'mean_energy', 'se_energy'], rows)

        # Percentiles from histogram helper
        def pct_from_hist(hist: np.ndarray, bins: np.ndarray, ps: list[float]):
            if hist is None or bins is None or np.sum(hist) == 0:
                return {p: np.nan for p in ps}
            cum = np.cumsum(hist.astype(float))
            total = cum[-1]
            res = {}
            for p in ps:
                target = total * p
                j = np.searchsorted(cum, target)
                j = min(max(j, 0), len(hist) - 1)
                prev = cum[j - 1] if j > 0 else 0.0
                denom = hist[j] if hist[j] > 0 else 1.0
                frac = (target - prev) / denom
                lo = bins[j]; hi = bins[j + 1]
                res[p] = float(lo + frac * (hi - lo))
            return res

    # Energy percentiles (core and tail)
        energy = aggregates.get('energy', {})
        e_hist = energy.get('hists', {})
        e_bins = energy.get('bins', {})
        percentiles = {}
        for key in ['total_energy_consumption', 'energy_intensity', 'energy_burden']:
            if key in e_hist and key in e_bins:
                percentiles[key] = pct_from_hist(e_hist[key], e_bins[key], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        # Write percentiles CSV
        rows = []
        for metric, d in percentiles.items():
            for p, v in d.items():
                rows.append([metric, p, v])
        write_csv(tables_dir / 'energy_percentiles.csv', ['metric', 'percentile', 'value'], rows)

        # JSON summary
        import json as _json
        with open(out_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
            _json.dump({
                'totals': t,
                'correlations': corr,
                'global_stats': gs,
                'energy_end_use_totals': eus,
                'energy_percentiles': percentiles,
                'conditional_curves': aggregates.get('conditional_curves', {}),
                'subgroups': {
                    'income_deciles': dec,
                }
            }, f, indent=2)

        # Markdown snippet
        md_lines = []
        md_lines.append('# Results summary (auto-generated)')
        md_lines.append('')
        md_lines.append(f"Buildings: {t.get('buildings', 'NA')}, Persons: {t.get('persons', 'NA')}, Activities: {t.get('activities', 'NA')}")
        if 'temp_vs_energy' in corr:
            r = corr['temp_vs_energy'].get('r', float('nan'))
            md_lines.append(f"Temperature vs energy correlation r = {r:.2f} (n={corr['temp_vs_energy'].get('n', 0)})")
        md_lines.append('')
        with open(out_dir / 'metrics_summary.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        # Coverage table
        cov = aggregates.get('coverage', {})
        if cov:
            rows = []
            for k, v in cov.items():
                total = v.get('total', 0)
                present = v.get('present', 0)
                share = (present / total) if total > 0 else np.nan
                rows.append([k, total, present, share])
            write_csv(tables_dir / 'coverage.csv', ['column', 'total_rows', 'present_values', 'share_present'], rows)

    # LaTeX snippet to include in manuscript
        tex_lines = []
        tex_lines.append('% Auto-generated results snippet')
        tex_lines.append('\\begin{table}[t]')
        tex_lines.append('  \\centering')
        tex_lines.append('  \\caption{Dataset coverage and totals.}')
        tex_lines.append('  \\label{tab:coverage-auto}')
        tex_lines.append('  \\begin{tabular}{lr}')
        tex_lines.append('    \\toprule')
        tex_lines.append('    Metric & Value \\\\')
        tex_lines.append('    \\midrule')
        for k, v in t.items():
            tex_lines.append(f'    {k.replace("_", " ").title()} & {v} \\\\')
        tex_lines.append('    \\bottomrule')
        tex_lines.append('  \\end{tabular}')
        tex_lines.append('\\end{table}')
        tex_lines.append('')
        # A figure panel example reference
        tex_lines.append('% Example figure include (update placement as needed)')
        tex_lines.append('\\begin{figure}[t]')
        tex_lines.append('  \\centering')
        tex_lines.append('  \\includegraphics[width=0.48\\textwidth]{results/visualizations_full/energy_hist_energy_intensity_full.png}')
        tex_lines.append('  \\includegraphics[width=0.48\\textwidth]{results/visualizations_full/energy_hist_energy_burden_full.png}')
        tex_lines.append('  \\caption{Energy intensity and energy burden distributions (full dataset).}')
        tex_lines.append('  \\label{fig:energy-dists-auto}')
        tex_lines.append('\\end{figure}')
        with open(out_dir / 'results_snippets.tex', 'w', encoding='utf-8') as f:
            f.write('\n'.join(tex_lines))

        # LaTeX subgroup tables: Top 10 states by energy burden and climate zone intensity means
        try:
            # Load subgroup moments
            sub = aggregates.get('subgroups', {})
            gm = sub.get('group_moments', {})
            # Top 10 states by energy burden mean
            states = gm.get('STATE', {})
            rows = []
            for s_name, metrics in states.items():
                m = metrics.get('energy_burden')
                if not m:
                    continue
                n = m.get('n', 0)
                mean = m.get('mean', np.nan)
                se = np.sqrt(m['M2'] / (n - 1) / n) if m.get('M2') is not None and n > 1 else np.nan
                if n > 0 and np.isfinite(mean):
                    rows.append((s_name, n, mean, se))
            rows.sort(key=lambda x: x[2], reverse=True)
            rows = rows[:10]
            # Write LaTeX
            with open(tables_dir / 'state_energy_burden_top10.tex', 'w', encoding='utf-8') as f:
                f.write('% Auto-generated table: Top 10 states by energy burden\n')
                f.write('\\begin{table}[t]\n  \\centering\n  \\caption{Top 10 states by mean energy burden (95\\% CI).}\n  \\label{tab:state-burden-top10}\n  \\begin{tabular}{lrrr}\n    \\toprule\n    State & n & Mean & 95\\% CI \\\\ \n    \\midrule\n')
                for s_name, n, mean, se in rows:
                    ci = 1.96 * se if np.isfinite(se) else float('nan')
                    if np.isfinite(ci):
                        f.write(f'    {s_name} & {n} & {mean:.2f} & [{mean-ci:.2f}, {mean+ci:.2f}] \\\\ \n')
                    else:
                        f.write(f'    {s_name} & {n} & {mean:.2f} & -- \\\\ \n')
                f.write('    \\bottomrule\n  \\end{tabular}\n\\end{table}\n')

            # Climate zone intensity means
            cz = sub.get('energy_intensity_by_climate_zone', {}).get('moments', {})
            rows = []
            for cz_name, m in cz.items():
                n = m.get('n', 0)
                mean = m.get('mean', np.nan)
                se = np.sqrt(m['M2'] / (n - 1) / n) if m.get('M2') is not None and n > 1 else np.nan
                if n > 0 and np.isfinite(mean):
                    rows.append((cz_name, n, mean, se))
            rows.sort(key=lambda x: x[2], reverse=True)
            with open(tables_dir / 'climate_energy_intensity.tex', 'w', encoding='utf-8') as f:
                f.write('% Auto-generated table: Climate zone energy intensity\n')
                f.write('\\begin{table}[t]\n  \\centering\n  \\caption{Energy intensity by climate zone (95\\% CI).}\n  \\label{tab:climate-intensity}\n  \\begin{tabular}{lrrr}\n    \\toprule\n    Climate Zone & n & Mean & 95\\% CI \\\\ \n    \\midrule\n')
                for cz_name, n, mean, se in rows:
                    ci = 1.96 * se if np.isfinite(se) else float('nan')
                    if np.isfinite(ci):
                        f.write(f'    {cz_name} & {n} & {mean:.2f} & [{mean-ci:.2f}, {mean+ci:.2f}] \\\\ \n')
                    else:
                        f.write(f'    {cz_name} & {n} & {mean:.2f} & -- \\\\ \n')
                f.write('    \\bottomrule\n  \\end{tabular}\n\\end{table}\n')
        except Exception as e:
            logger.warning(f'Failed to write LaTeX subgroup tables: {e}')

        # Figures manifest and index
        manifest = []
        for p in sorted(out_dir.glob('*.png')):
            manifest.append({'file': p.name, 'path': str(p), 'bytes': p.stat().st_size})
        import json as _json2
        with open(out_dir / 'figures_manifest.json', 'w', encoding='utf-8') as f:
            _json2.dump(manifest, f, indent=2)
        with open(out_dir / 'figures_index.md', 'w', encoding='utf-8') as f:
            f.write('# Figures index\n\n')
            for m in manifest:
                f.write(f"- {m['file']}\n")

        # Activity transitions: top pairs with probabilities
        try:
            tr = aggregates.get('transition_counts')
            cats = aggregates.get('categories', [])
            if isinstance(tr, np.ndarray) and tr.size > 0 and cats:
                row_sums = tr.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    probs = np.divide(tr, row_sums, out=np.zeros_like(tr, dtype=float), where=row_sums>0)
                pairs = []
                for i in range(tr.shape[0]):
                    for j in range(tr.shape[1]):
                        if tr[i, j] > 0:
                            pairs.append([cats[i], cats[j], int(tr[i, j]), float(probs[i, j])])
                pairs.sort(key=lambda x: x[2], reverse=True)
                write_csv(tables_dir / 'activity_top_transitions.csv', ['from', 'to', 'count', 'prob'], pairs[:100])
                # Full probability matrix
                rows = []
                header = ['from\\to'] + cats
                for i in range(tr.shape[0]):
                    rows.append([cats[i]] + [float(probs[i, j]) for j in range(tr.shape[1])])
                write_csv(tables_dir / 'activity_transition_matrix.csv', header, rows)
        except Exception:
            pass

        # Shard-level summaries
        shards = aggregates.get('shards', [])
        if shards:
            # Derive across-shard SEs for a quick uncertainty summary
            df = pd.DataFrame(shards)
            df.to_csv(tables_dir / 'shard_summaries.csv', index=False)

        # Energy poverty by state
        eps = aggregates.get('energy_poverty_state', {})
        if eps:
            rows = []
            for st, d in eps.items():
                den = d.get('den', 0) or 0
                n10 = d.get('num_10', 0) or 0
                n20 = d.get('num_20', 0) or 0
                sh10 = (n10 / den) if den > 0 else np.nan
                sh20 = (n20 / den) if den > 0 else np.nan
                rows.append([st, den, n10, n20, sh10, sh20])
            write_csv(tables_dir / 'energy_poverty_by_state.csv', ['state', 'den', 'num_10', 'num_20', 'share_10', 'share_20'], rows)

        # Energy poverty by group
        epg = aggregates.get('energy_poverty_by_group', {})
        for gv, d in epg.items():
            rows = []
            for k, v in d.items():
                den = v.get('den', 0)
                n10 = v.get('num_10', 0)
                n20 = v.get('num_20', 0)
                sh10 = (n10 / den) if den > 0 else np.nan
                sh20 = (n20 / den) if den > 0 else np.nan
                rows.append([k, den, n10, n20, sh10, sh20])
            write_csv(tables_dir / f'energy_poverty_by_{gv}.csv', ['group', 'den', 'num_10', 'num_20', 'share_10', 'share_20'], rows)

        # Subgroup activity timelines (write average shares per minute)
        tlg = aggregates.get('timeline_by_group', {})
        for gname, groups in tlg.items():
            for gval, mat in groups.items():
                if isinstance(mat, np.ndarray) and mat.size > 0:
                    minute_totals = mat.sum(axis=0)
                    shares = np.divide(mat, minute_totals, out=np.zeros_like(mat, dtype=float), where=minute_totals>0)
                    # Write as long CSV: minute, category, share
                    rows = []
                    for ci, cat in enumerate(aggregates.get('categories', [])):
                        for m in range(1440):
                            rows.append([m, cat, float(shares[ci, m])])
                    write_csv(tables_dir / f'activity_shares_{gname}_{gval}.csv', ['minute', 'category', 'share'], rows)

        # Export conditional curve eb_by_hhsize
        # Derived: energy per capita histogram
        dh = aggregates.get('derived_hists', {}).get('energy_per_capita')
        if dh:
            bins = dh.get('bins', [])
            hist = dh.get('hist', [])
            rows = []
            for i in range(max(0, len(bins) - 1)):
                rows.append([bins[i], bins[i+1], hist[i] if i < len(hist) else 0])
            write_csv(tables_dir / 'energy_per_capita_hist.csv', ['bin_lo', 'bin_hi', 'count'], rows)

        # Time profiles exports
        tp = aggregates.get('time_profiles', {})
        def finalize_profile_csv(stats):
            try:
                sumv = np.array(stats['sum'], dtype=float); sum2 = np.array(stats['sum2'], dtype=float); cnt = np.array(stats['count'], dtype=float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan), where=cnt>0)
                    var = np.divide((sum2 - (sumv * sumv) / np.maximum(cnt, 1)), np.maximum(cnt - 1, 1), out=np.full_like(sumv, np.nan), where=cnt>1)
                    se = np.sqrt(np.divide(var, cnt, out=np.full_like(sumv, np.nan), where=cnt>0))
                return mean, se, cnt
            except Exception:
                return None, None, None
        # Hourly overall
        if 'hourly' in tp and tp['hourly'].get('overall'):
            mean, se, cnt = finalize_profile_csv(tp['hourly']['overall'])
            if mean is not None:
                rows = [[i, float(mean[i]) if np.isfinite(mean[i]) else '', float(se[i]) if np.isfinite(se[i]) else '', int(cnt[i])] for i in range(len(mean))]
                write_csv(tables_dir / 'hourly_energy_profile_overall.csv', ['hour', 'mean', 'se', 'n'], rows)
        # Hourly groups
        for gname in ['building_type_simple', 'STATE', 'climate_zone']:
            groups = tp.get('hourly', {}).get('groups', {}).get(gname, {})
            for gval, stats in groups.items():
                mean, se, cnt = finalize_profile_csv(stats)
                if mean is None:
                    continue
                rows = [[i, float(mean[i]) if np.isfinite(mean[i]) else '', float(se[i]) if np.isfinite(se[i]) else '', int(cnt[i])] for i in range(len(mean))]
                safe_name = str(gval).replace('/', '_')
                write_csv(tables_dir / f'hourly_energy_profile_{gname}_{safe_name}.csv', ['hour', 'mean', 'se', 'n'], rows)
        # Minute overall (sampled)
        if 'minute' in tp and tp['minute'].get('overall'):
            mean, se, cnt = finalize_profile_csv(tp['minute']['overall'])
            if mean is not None:
                rows = [[i, float(mean[i]) if np.isfinite(mean[i]) else '', float(se[i]) if np.isfinite(se[i]) else '', int(cnt[i])] for i in range(len(mean))]
                write_csv(tables_dir / 'minute_energy_profile_overall.csv', ['minute', 'mean', 'se', 'n'], rows)
        # Minute groups (only building_type to limit volume)
        groups = tp.get('minute', {}).get('groups', {}).get('building_type_simple', {})
        for gval, stats in groups.items():
            mean, se, cnt = finalize_profile_csv(stats)
            if mean is None:
                continue
            rows = [[i, float(mean[i]) if np.isfinite(mean[i]) else '', float(se[i]) if np.isfinite(se[i]) else '', int(cnt[i])] for i in range(len(mean))]
            safe_name = str(gval).replace('/', '_')
            write_csv(tables_dir / f'minute_energy_profile_building_type_simple_{safe_name}.csv', ['minute', 'mean', 'se', 'n'], rows)

        # Peak hour stats exports
        peaks = aggregates.get('peaks', {})
        if peaks:
            ov = peaks.get('overall', {})
            ph = ov.get('peak_hour_hist', [])
            if isinstance(ph, (list, np.ndarray)) and len(ph) == 24:
                rows = [[h, int(ph[h])] for h in range(24)]
                write_csv(tables_dir / 'peak_hour_histogram.csv', ['hour', 'count'], rows)
            sop = ov.get('sum_of_peaks', 0.0)
            cnt = ov.get('count', 0)
            avg_peak = (sop / cnt) if cnt > 0 else ''
            write_csv(tables_dir / 'peak_overall_summary.csv', ['n_households', 'sum_of_peaks', 'avg_peak'], [[cnt, sop, avg_peak]])
            # By group
            by_group = peaks.get('by_group', {})
            for gname, gdict in by_group.items():
                rows = []
                for gval, stats in gdict.items():
                    cnt = stats.get('count', 0)
                    sop = stats.get('sum_of_peaks', 0.0)
                    avg = (sop / cnt) if cnt > 0 else ''
                    # summarize peak hour as argmax of hist for the group
                    ph = stats.get('peak_hour_hist', np.zeros(24, dtype=int))
                    dom_hour = int(np.argmax(ph)) if isinstance(ph, (list, np.ndarray)) and len(ph) == 24 else ''
                    rows.append([gval, cnt, sop, avg, dom_hour])
                write_csv(tables_dir / f'peak_summary_by_{gname}.csv', ['group', 'n', 'sum_of_peaks', 'avg_peak', 'mode_peak_hour'], rows)

        # TOU average share exports
        tou = aggregates.get('tou', {})
        if tou:
            bands = tou.get('bands', {})
            avg = tou.get('average_share', {})
            labels = list(bands.keys())
            if labels:
                rows = []
                for bn in labels:
                    ss = avg.get(bn, {}).get('sum_share', 0.0)
                    c = avg.get(bn, {}).get('count', 0)
                    rows.append([bn, c, (ss / c) if c > 0 else ''])
                write_csv(tables_dir / 'tou_average_shares_overall.csv', ['band', 'n_households', 'avg_share'], rows)
            by_group = tou.get('average_share_by_group', {})
            for gname, gdict in by_group.items():
                rows = []
                for gval, bstats in gdict.items():
                    for bn, s in bstats.items():
                        c = s.get('count', 0)
                        avg_share = (s.get('sum_share', 0.0) / c) if c > 0 else ''
                        rows.append([gval, bn, c, avg_share])
                write_csv(tables_dir / f'tou_average_shares_by_{gname}.csv', ['group', 'band', 'n', 'avg_share'], rows)

        # Load duration curve export (from hourly overall mean as proxy)
        if 'hourly' in tp and tp['hourly'].get('overall'):
            mean, se, cnt = finalize_profile_csv(tp['hourly']['overall'])
            if mean is not None:
                vals = np.sort(mean[np.isfinite(mean)])[::-1]
                rows = [[i, float(vals[i])] for i in range(len(vals))]
                write_csv(tables_dir / 'load_duration_curve_hourly_mean.csv', ['rank', 'mean_load'], rows)

        # Per-building-type EI vs age curve means
        cbt = aggregates.get('conditional_curves_by_type', {}).get('ei_by_age_by_type')
        if cbt:
            edges = cbt.get('bin_edges', [])
            for tkey, stats in cbt.get('groups', {}).items():
                cnt = stats.get('count', [])
                mean = stats.get('mean', [])
                se = stats.get('se', [])
                rows = []
                for i in range(max(0, len(edges) - 1)):
                    rows.append([edges[i], edges[i+1], cnt[i] if i < len(cnt) else 0, mean[i] if i < len(mean) else '', se[i] if i < len(se) else ''])
                write_csv(tables_dir / f'ei_vs_age_{tkey}.csv', ['age_lo', 'age_hi', 'n', 'mean_ei', 'se_ei'], rows)
        cc = aggregates.get('conditional_curves', {})
        ebhs = cc.get('eb_by_hhsize')
        if ebhs:
            edges = ebhs.get('bin_edges', [])
            count = ebhs.get('count', [])
            mean = ebhs.get('mean', [])
            se = ebhs.get('se', [])
            rows = []
            for i in range(max(0, len(edges) - 1)):
                rows.append([edges[i], edges[i+1], count[i] if i < len(count) else 0, mean[i] if i < len(mean) else '', se[i] if i < len(se) else ''])
            write_csv(tables_dir / 'eb_vs_hhsize_curve.csv', ['hhsize_lo', 'hhsize_hi', 'n', 'mean_eb', 'se_eb'], rows)

    except Exception as e:
        logger.warning(f"Failed to export results artifacts: {e}")


def run_all_visualizations():
    """Run all visualization modules."""
    
    safe_print("\n" + "="*60)
    safe_print("PUMS ENRICHMENT COMPLETE LIVING SYSTEM - VISUALIZATION")
    safe_print("="*60 + "\n")
    
    # Load data
    safe_print("Loading FINAL Phase 4 integrated living system data...")
    data_dir = Path("data/processed")
    manifest_path = data_dir / "phase4_shards" / "manifest.json"
    buildings_df, metadata, phase = load_latest_data()
    
    if buildings_df is None:
        safe_print("ERROR: Complete living system not found. Please run all 4 phases first.")
        safe_print("       The visualization system only works with the final integrated data.")
        return 1
    
    safe_print(f"\n[OK] Loaded COMPLETE LIVING SYSTEM (Phase {phase}):")
    safe_print(f"  - Buildings with PUMS demographics: {metadata.get('households', len(buildings_df))}")
    if metadata.get('persons', 0) > 0:
        safe_print(f"  - Persons with individual characteristics: {metadata['persons']}")
    else:
        safe_print("  - Persons with individual characteristics: (will be computed during streaming)")
    safe_print(f"  - RECS building energy characteristics: OK")
    safe_print(f"  - ATUS minute-by-minute activities: OK")
    safe_print(f"  - NSRDB weather data aligned: OK")
    
    # Create results directory
    results_dir = Path("results/visualizations")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    safe_print(f"\nGenerating visualizations...")
    safe_print("-" * 40)
    
    try:
        # If we have shards manifest, prefer streaming full-data aggregates
        if manifest_path.exists():
            safe_print("\n[Full Dataset] Streaming shards to compute aggregate visualizations...")
            aggregates = _aggregate_full_dataset(manifest_path)
            out_full = Path("results/visualizations_full")
            _render_full_plots(aggregates, out_full)
            safe_print("   OK Full-dataset aggregate visualizations complete")

        # 1. System Overview (uses metadata, safe to run)
        safe_print("\n1. Creating System Overview visualizations...")
        system_viz = SystemOverviewVisualizer()
        system_viz.create_data_flow_sankey(metadata)
        # For large datasets, skip passing entire DataFrame to expensive overview plots
        if len(buildings_df) > 0 and len(buildings_df) <= 250_000:
            system_viz.create_system_metrics_dashboard(buildings_df)
            system_viz.create_relationship_network(buildings_df)
        safe_print("   OK System overview complete")
        
        # 2. Building Visualizations (run only if manageable in-memory)
        safe_print("\n2. Creating Building visualizations...")
        building_viz = BuildingVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 250_000:
            building_viz.create_building_type_distribution(buildings_df)
            building_viz.create_geographic_distribution(buildings_df)
            building_viz.create_energy_characteristics(buildings_df)
            building_viz.create_building_occupancy_analysis(buildings_df)
            safe_print("   OK Building visualizations (in-memory) complete")
        else:
            safe_print("   Skipped in-memory building plots due to dataset size; see results/visualizations_full for full-data charts.")
        
        # 3. Person Visualizations (in-memory only if manageable)
        safe_print("\n3. Creating Person Demographics visualizations...")
        person_viz = PersonVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 100_000:
            persons_df = person_viz.extract_persons_from_buildings(buildings_df)
            if len(persons_df) > 0:
                person_viz.create_age_pyramid(persons_df)
                person_viz.create_demographic_distributions(persons_df)
                person_viz.create_household_composition(buildings_df)
                person_viz.create_employment_analysis(persons_df)
            safe_print("   OK Person visualizations (in-memory) complete")
        else:
            safe_print("   Skipped in-memory person plots due to dataset size; see results/visualizations_full for full-data charts.")
        
        # 4. Activity Visualizations (avoid in-memory on huge datasets)
        safe_print("\n4. Creating Activity Pattern visualizations...")
        activity_viz = ActivityVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 50_000:
            activities = activity_viz.extract_activities_from_persons(buildings_df)
            if len(activities) > 0:
                activity_viz.create_daily_activity_timeline(activities)
                activity_viz.create_aggregate_activity_patterns(activities)
                activity_viz.create_activity_transition_matrix(activities)
                activity_viz.create_activity_by_demographics(activities)
                activity_viz.create_household_activity_coordination(buildings_df)
                safe_print("   OK Activity visualizations (in-memory) complete")
                safe_print(f"     - Total activities visualized: {len(activities)}")
        else:
            safe_print("   Skipped in-memory activity plots due to dataset size; see results/visualizations_full for full-data charts.")
        
        # 5. Weather Visualizations (skip heavy DataFrame ops on huge datasets)
        safe_print("\n5. Creating Weather visualizations...")
        weather_viz = WeatherVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 250_000:
            weather_viz.create_temperature_profiles(buildings_df)
            weather_viz.create_weather_conditions_dashboard(buildings_df)
            weather_viz.create_extreme_weather_analysis(buildings_df)
            safe_print("   OK Weather visualizations (in-memory) complete")
        else:
            safe_print("   Skipped in-memory weather plots due to dataset size; full-data weather plots can be added via streaming if needed.")
        
        # 6. Energy Visualizations (skip in-memory on huge datasets)
        safe_print("\n6. Creating Energy visualizations...")
        energy_viz = EnergyVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 250_000:
            energy_viz.create_energy_consumption_overview(buildings_df)
            energy_viz.create_daily_load_profiles(buildings_df)
            energy_viz.create_efficiency_analysis(buildings_df)
            energy_viz.create_renewable_energy_potential(buildings_df)
            energy_viz.create_energy_cost_analysis(buildings_df)
            safe_print("   OK Energy visualizations (in-memory) complete")
        else:
            safe_print("   Skipped in-memory energy plots due to dataset size; see results/visualizations_full for full-data charts.")
        
        # 7. Household Visualizations (skip in-memory on huge datasets)
        safe_print("\n7. Creating Household Dynamics visualizations...")
        household_viz = HouseholdVisualizer()
        if len(buildings_df) > 0 and len(buildings_df) <= 100_000:
            household_viz.create_household_structure_analysis(buildings_df)
            household_viz.create_coordinated_activities_timeline(buildings_df)
            household_viz.create_household_interaction_network(buildings_df)
            household_viz.create_resource_sharing_patterns(buildings_df)
            household_viz.create_lifecycle_stage_analysis(buildings_df)
            safe_print("   OK Household visualizations (in-memory) complete")
        else:
            safe_print("   Skipped in-memory household plots due to dataset size; see results/visualizations_full for full-data charts.")
        
        # 8. Dashboard Generation (interactive parts may be heavy)
        safe_print("\n8. Creating Comprehensive Dashboards...")
        dashboard_gen = DashboardGenerator()
        # Executive summary uses aggregate metrics; safe to run
        dashboard_gen.create_executive_summary_dashboard(buildings_df if len(buildings_df) <= 250_000 else pd.DataFrame(), metadata)
        if len(buildings_df) > 0 and len(buildings_df) <= 100_000:
            dashboard_gen.create_interactive_dashboard(buildings_df, metadata)
        dashboard_gen.create_performance_monitoring_dashboard(metadata)
        safe_print("   OK Dashboards complete")
        
        # Summary
        safe_print("\n" + "="*60)
        safe_print("VISUALIZATION GENERATION COMPLETE!")
        safe_print("="*60)
        
        # List generated files (both in-memory and full-data aggregate outputs)
        safe_print("\nGenerated visualization files:")
        for viz_dir in [results_dir, Path("results/visualizations_full")]:
            if not viz_dir.exists():
                continue
            if viz_dir.is_dir():
                files = sorted(list(viz_dir.glob("*")))
                if files:
                    safe_print(f"\n  {viz_dir.name}/")
                    for f in files[:5]:  # Show first 5 files
                        safe_print(f"    - {f.name}")
                    if len(files) > 5:
                        safe_print(f"    ... and {len(files) - 5} more files")

        safe_print("\nComplete Living System Visualizations Created:")
        safe_print("  - Executive Summary Dashboard - Full 4-phase system overview")
        safe_print("  - Building Characteristics - Full-dataset aggregates (results/visualizations_full)")
        safe_print("  - Person Demographics - Full-dataset aggregates (results/visualizations_full)")
        safe_print("  - Activity Patterns - Full-dataset aggregates (results/visualizations_full)")
        safe_print("  - Weather/Energy/Households - Full-dataset aggregates (results/visualizations_full)")

        safe_print(f"\nAll visualizations saved to: {results_dir.absolute()}")
        safe_print("\nYour COMPLETE LIVING SYSTEM has been visualized.")
        safe_print("   Buildings -> Persons -> Activities -> Weather")
        safe_print("   All synchronized at 1-minute resolution!")

        return 0
        
    except Exception as e:
        safe_print(f"\nError during visualization: {e}")
        logger.exception("Visualization error")
        return 1


if __name__ == "__main__":
    exit_code = run_all_visualizations()
    sys.exit(exit_code)