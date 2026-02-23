# Behavior-Aware Synthetic Building Populations Through Probabilistic Multi-Source Data Integration

A data integration pipeline that constructs synthetic building populations for energy modeling by combining four U.S. government datasets using probabilistic record linkage. Each building in the output contains household demographics, detailed energy characteristics, occupant daily activity schedules, and local weather conditions.

## Data Sources

| Source | Provides | Scale |
| ------ | -------- | ----- |
| **PUMS** 2023 (American Community Survey) | Household demographics, housing, income, person-level attributes | 1,444,325 households; 3,405,809 persons |
| **RECS** 2020 (Residential Energy Consumption Survey) | Building characteristics, appliances, insulation, energy use | National sample used as matching templates |
| **ATUS** 2023 (American Time Use Survey) | 24-hour activity diaries with demographic profiles | 8,548 respondents; 153,120 activity records |
| **NSRDB** (National Solar Radiation Database) | Hourly weather: temperature, humidity, solar radiation, wind | State-level hourly time series |

## Pipeline Architecture

```text
PUMS Households + Persons
         |
  Phase 1: Deterministic merge on SERIALNO, 200+ engineered features
         |
  Phase 2: Probabilistic RECS matching (Fellegi-Sunter + EM algorithm)
         |
  Phase 3: ATUS activity assignment (optimized k-NN, household coordination)
         |
  Phase 4: State-level weather integration (hourly interpolation)
         |
  Final integrated buildings (shards + manifest)
```

## Prerequisites

- Python 3.9+
- CUDA 12.1 (optional, for PyTorch GPU acceleration)
- Raw data files placed in `data/raw/` (see [project_description.md](project_description.md) for details)

## Installation

```bash
git clone https://github.com/MiladLatifiNL/Behavior-Aware-Synthetic-Building-Populations-Through-Probabilistic-Multi-Source-Data-Integration.git
cd PUMS_Enrichment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Usage

### Basic Commands

```bash
# Run all phases with a small sample (development/testing)
python main.py --phase all --sample-size 100

# Run all phases with full data
python main.py --phase all --full-data

# Run a single phase
python main.py --phase 2 --sample-size 1000

# Resume from the last completed phase
python main.py --phase all --resume

# Resume starting at a specific phase
python main.py --phase all --resume --from-phase 3

# Validate existing output without reprocessing
python main.py --validate-only

# Generate visualizations from Phase 4 data
python run_visualizations.py
```

### CLI Reference

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--phase {1,2,3,4,all}` | Phase(s) to run | `1` |
| `--sample-size N` | Number of buildings to process | Full data |
| `--full-data` | Process the complete dataset | Off |
| `--validate-only` | Run validation on existing output only | Off |
| `--skip-validation` | Skip validation after processing | Off |
| `--resume` | Resume from last successful phase | Off |
| `--from-phase {1,2,3,4}` | Force resume starting at this phase | Auto |
| `--config PATH` | Path to configuration file | `config.yaml` |
| `--verbose`, `-v` | Enable debug-level logging | Off |
| `--streaming` / `--no-streaming` | Streaming shard-based I/O | Enabled |
| `--no-parallel` | Disable parallel processing | Parallel on |
| `--workers N` | Number of parallel workers | All CPUs |
| `--batch-size N` | Batch size for streaming mode | Auto |
| `--chunk-size N` | Chunk size for batch processing | Auto |
| `--no-optimize-memory` | Disable memory optimization | Memory opt on |
| `--no-checkpoint` | Disable checkpoint/resume | Checkpoint on |
| `--memory-limit N` | Maximum memory usage in GB | 90% of RAM |
| `--phase2-sub-batch N` | Phase 2 sub-batch size per shard | Auto (500-2000) |
| `--phase2-max-candidates N` | Phase 2 max candidates per record | Config default |

## Output

| Location | Contents |
| -------- | -------- |
| `data/processed/phase{1-4}_shards/` | Full data as shard files with `manifest.json` |
| `data/processed/phase{1-4}_*.pkl` | Small sample pickle files for quick inspection |
| `data/processed/phase{1-4}_metadata.json` | Processing metadata per phase |
| `data/matching_parameters/` | Learned Fellegi-Sunter weights (JSON) |
| `data/validation/` | HTML validation reports per phase |
| `results/visualizations/` | Plots and HTML dashboards from Phase 4 data |
| `logs/` | Phase-specific runtime logs |

The canonical full dataset resides in `data/processed/phase4_shards/`. The sample pickle files are subsets for quick inspection and are not the complete output.

## Documentation

- [project_description.md](project_description.md) -- Methodology, theoretical foundation, and data sources
- [workflow.md](workflow.md) -- Pipeline architecture, streaming design, and phase execution details
- [project_structure.md](project_structure.md) -- Complete file inventory of the repository

## Key Dependencies

Core: pandas, numpy, scikit-learn |
Record linkage: recordlinkage, jellyfish, python-Levenshtein, faiss-cpu |
Deep learning: torch (CUDA 12.1), torchvision, torchaudio |
Visualization: matplotlib, seaborn, plotly, networkx, jinja2 |
Utilities: pyyaml, tqdm, psutil, hmmlearn, mmh3

See [requirements.txt](requirements.txt) for the full list with version constraints.
