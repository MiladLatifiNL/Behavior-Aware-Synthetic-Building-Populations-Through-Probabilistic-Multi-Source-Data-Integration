# Pipeline Workflow

## Overview

The PUMS Enrichment Pipeline is a 4-phase sequential data integration system. Each phase consumes the prior phase's output, enriches building records with data from an additional source, and writes the result as shards for the next phase. The final output is a unified dataset of buildings with occupants, energy characteristics, activity schedules, and weather conditions.

---

## Data Flow

```text
                     PUMS Households + Persons
                              |
                     [ Phase 1: Deterministic Merge ]
                     (merge on SERIALNO, feature engineering)
                              |
                     phase1_shards/ + manifest.json
                              |
          RECS 2020 ------> [ Phase 2: Probabilistic RECS Matching ]
                             (Fellegi-Sunter + EM, multi-level blocking)
                              |
                     phase2_shards/ + manifest.json
                              |
          ATUS 2023 ------> [ Phase 3: Activity Pattern Assignment ]
                             (optimized k-NN, household coordination)
                              |
                     phase3_shards/ + manifest.json
                              |
     NSRDB Weather -------> [ Phase 4: Weather Integration ]
                             (state-level spatial join, interpolation)
                              |
                     phase4_shards/ + manifest.json
                              |
                     Final Integrated Buildings
```

---

## Phase Execution

### Phase 1: PUMS Household-Person Integration

- **Input**: Raw PUMS household CSV (`psam_hus*.csv`) and person CSV (`psam_pus*.csv`)
- **Process**: Deterministic merge on SERIALNO. Feature engineering generates 200+ derived features covering household composition, income, building type, energy profiles, and geographic blocking keys. Empty households are filtered.
- **Output**: `phase1_shards/` with `manifest.json`; sample pickle `phase1_pums_buildings.pkl`
- **Key modules**: `pums_loader.py`, `phase1_pums_integration.py`, `feature_engineering.py`

### Phase 2: RECS Building Characteristics Matching

- **Input**: Phase 1 shards + raw RECS data (`recs2020_public_v7.csv`)
- **Process**:
  1. Load and standardize RECS data with the same encoding as PUMS
  2. Generate 100+ comparison features (`enhanced_feature_engineering.py`)
  3. Apply multi-level blocking to generate candidate pairs (7 levels with fallback)
  4. Run EM algorithm to estimate Fellegi-Sunter m/u probabilities
  5. Calculate match weights and assign each PUMS building to a RECS template
- **Output**: `phase2_shards/` with `manifest.json`; sample pickle `phase2_pums_recs_buildings.pkl`
- **Parameters saved**: `data/matching_parameters/phase2_recs_weights.json`
- **Key modules**: `recs_loader.py`, `phase2_recs_matching.py`, `fellegi_sunter.py`, `em_algorithm.py`, `blocking.py`, `enhanced_feature_engineering.py`

### Phase 3: ATUS Activity Pattern Assignment

- **Input**: Phase 2 shards + ATUS 2023 data (10 files: respondent, activity, roster, etc.)
- **Process**:
  1. Load and merge ATUS respondent, activity, and roster records (8,548 respondents)
  2. Generate 50+ alignment features (`enhanced_feature_alignment.py`)
  3. Match each person to the nearest ATUS respondent using optimized k-NN (`scipy.cdist` on 8 key demographic features)
  4. Apply household coordination constraints (childcare coverage, meal times, schedule compatibility)
- **Output**: `phase3_shards/` with `manifest.json`; sample pickle `phase3_pums_recs_atus_buildings.pkl`
- **Parameters saved**: `data/matching_parameters/phase3_atus_weights.json`
- **Key modules**: `atus_loader.py`, `phase3_atus_matching_optimized.py`, `enhanced_feature_alignment.py`, `household_coordination.py`

### Phase 4: Weather Integration

- **Input**: Phase 3 shards + NSRDB weather data (state-level hourly observations)
- **Process**: State-level spatial join. Hourly weather observations are interpolated to align with minute-by-minute activity schedules. Adds temperature, humidity, solar radiation, wind speed, heating/cooling degree days, and weather condition classifications.
- **Output**: `phase4_shards/` with `manifest.json`; sample pickle `phase4_final_integrated_buildings.pkl`
- **Key modules**: `weather_loader.py`, `phase4_weather_integration.py`

---

## Streaming Architecture

All phases run in streaming mode by default to maintain a bounded memory footprint regardless of dataset size.

- Each phase reads the prior phase's shard files (listed in `manifest.json`) rather than loading a single monolithic pickle.
- Each phase writes its own shard directory (e.g., `phase2_shards/`) with a `manifest.json` containing shard file paths and total building count.
- Small sample pickle files (e.g., `phase1_pums_buildings.pkl`) are retained alongside shards for quick inspection but are not the canonical full-data output.
- At full scale, each phase produces several hundred shards. Downstream phases iterate over shards sequentially, processing one batch at a time.

---

## Performance Optimizations

The following optimizations are enabled by default and can be individually disabled:

| Feature | Default | Disable flag |
| ------- | ------- | ------------ |
| Parallel processing | Enabled (auto-detected CPU cores) | `--no-parallel` |
| Memory optimization | Enabled (dtype reduction, GC) | `--no-optimize-memory` |
| Checkpointing | Enabled (resume on failure) | `--no-checkpoint` |
| Streaming mode | Enabled (shard-based I/O) | `--no-streaming` |

Additional capabilities:

- Automatic chunk size and batch size calibration based on available RAM
- GPU detection and TF32 optimization for PyTorch operations where applicable
- Memory monitoring with configurable limits (`--memory-limit`)
- Progress tracking with tqdm for all long-running operations

---

## Configuration

All pipeline parameters are defined in `config.yaml`, organized into sections:

- `data_paths`: Input and output file locations
- `processing`: Sample size, random seed, chunk size, worker count, memory limits
- `phase1` through `phase4`: Phase-specific settings (column selections, matching thresholds)
- `matching`: Fellegi-Sunter parameters, blocking strategies, similarity thresholds, EM settings
- `validation`: Quality thresholds (minimum match rates, maximum missing percentages)
- `logging`: Log levels and formats

CLI flags override config values at runtime. See `README.md` for the complete CLI reference.

---

## Validation

Each phase can produce an HTML validation report in `data/validation/`, covering:

- Data completeness and missing value rates
- Internal consistency checks
- Match quality metrics (weight distributions, field agreement rates, EM convergence)
- Coverage percentages

Validation can run independently with `python main.py --validate-only` or be skipped with `--skip-validation`. The pipeline can optionally halt on validation failures based on configured thresholds.

---

## Visualization

The visualization system generates plots and dashboards from the Phase 4 final dataset:

- **Entry point**: `python run_visualizations.py`
- **Modules**: 8 visualizers in `src/visualize/` covering building characteristics, demographics, activity patterns, weather, energy analysis, household dynamics, and system overviews
- **Data source**: Reads Phase 4 shards (via manifest) or falls back to the Phase 4 sample pickle
- **Output**: 30+ plots saved to `results/visualizations/` with HTML dashboards

---

## Logging

- Phase-specific log files are written to `logs/` (e.g., `phase1.log`, `phase2.log`)
- `main.log` captures the overall pipeline execution
- Log level is configurable via `config.yaml` or `--verbose` for debug-level output
- Performance metrics (processing time, memory usage, throughput) are logged at each phase
- Matching diagnostics are written to `logs/matching/` when verbose mode is enabled
