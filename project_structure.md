# Project Structure

```text
PUMS_Enrichment/
│
├── README.md                              # Project overview and quick start guide
├── requirements.txt                       # Python package dependencies
├── config.yaml                            # Pipeline configuration (paths, parameters, thresholds)
├── main.py                                # CLI entry point and pipeline orchestrator
├── run_visualizations.py                  # Generate all Phase 4 visualizations
├── project_description.md                 # Methodology and technical approach
├── project_structure.md                   # Repository file inventory (this file)
├── workflow.md                            # Pipeline architecture and data flow
│
├── data/                                  # All data files (not tracked in git)
│   ├── raw/                               # Original downloaded datasets
│   │   ├── pums/2023/                     # PUMS ACS 2023 microdata
│   │   │   ├── psam_hus*.csv              # Household files
│   │   │   └── psam_pus*.csv              # Person files
│   │   ├── recs/2020/                     # RECS 2020 building characteristics
│   │   │   └── recs2020_public_v7.csv     # Survey microdata
│   │   ├── atus/2023/                     # ATUS 2023 time use data
│   │   │   ├── atusact_2023.dat           # Activity records
│   │   │   ├── atusresp_2023.dat          # Respondent demographics
│   │   │   └── (8 additional files)       # Roster, who, summary, etc.
│   │   └── weather/2023/                  # NSRDB weather station data by state
│   │
│   ├── processed/                         # Pipeline output
│   │   ├── phase1_pums_buildings.pkl      # Phase 1 sample output
│   │   ├── phase1_sample.csv              # Phase 1 CSV excerpt (first 10 buildings)
│   │   ├── phase1_metadata.json           # Phase 1 processing metadata
│   │   ├── phase1_shards/                 # Phase 1 full data shards + manifest.json
│   │   ├── phase2_pums_recs_buildings.pkl # Phase 2 sample output
│   │   ├── phase2_sample.csv              # Phase 2 CSV excerpt
│   │   ├── phase2_metadata.json           # Phase 2 processing metadata
│   │   ├── phase2_shards/                 # Phase 2 full data shards + manifest.json
│   │   ├── phase3_pums_recs_atus_buildings.pkl  # Phase 3 sample output
│   │   ├── phase3_sample.csv              # Phase 3 CSV excerpt
│   │   ├── phase3_metadata.json           # Phase 3 processing metadata
│   │   ├── phase3_shards/                 # Phase 3 full data shards + manifest.json
│   │   ├── phase4_final_integrated_buildings.pkl  # Phase 4 sample output
│   │   ├── phase4_metadata.json           # Phase 4 processing metadata
│   │   ├── phase4_shards/                 # Phase 4 full data shards + manifest.json
│   │   └── weather_cache/                 # Cached interpolated weather data
│   │
│   ├── validation/                        # Quality reports (HTML)
│   │   ├── phase1_validation_report.html
│   │   ├── phase2_validation_report.html
│   │   ├── phase3_validation_report.html
│   │   └── phase4_validation_report.html
│   │
│   ├── matching_parameters/               # Learned Fellegi-Sunter parameters (JSON)
│   │   ├── phase2_recs_weights.json
│   │   └── phase3_atus_weights.json
│   │
│   └── checkpoints/                       # Checkpoint files for resume capability
│
├── src/                                   # Source code
│   ├── __init__.py
│   │
│   ├── data_loading/                      # Dataset loaders
│   │   ├── __init__.py
│   │   ├── pums_loader.py                 # PUMS household and person data loading
│   │   ├── recs_loader.py                 # RECS building characteristics loading
│   │   ├── atus_loader.py                 # ATUS activity and respondent data loading
│   │   └── weather_loader.py              # NSRDB weather data loading and interpolation
│   │
│   ├── processing/                        # Phase processing logic
│   │   ├── __init__.py
│   │   ├── phase1_pums_integration.py     # PUMS household-person deterministic merge
│   │   ├── phase2_recs_matching.py        # Probabilistic RECS matching (Fellegi-Sunter)
│   │   ├── phase3_atus_matching.py        # ATUS activity pattern matching (full version)
│   │   ├── phase3_atus_matching_optimized.py  # Optimized k-NN activity matching
│   │   └── phase4_weather_integration.py  # State-level weather integration
│   │
│   ├── matching/                          # Probabilistic record linkage algorithms
│   │   ├── __init__.py
│   │   ├── string_comparators.py          # Jaro-Winkler, edit distance, phonetic matching
│   │   ├── blocking.py                    # Multi-level blocking strategies
│   │   ├── fellegi_sunter.py              # Core Fellegi-Sunter probabilistic matcher
│   │   ├── em_algorithm.py                # EM algorithm for parameter estimation
│   │   ├── advanced_em_algorithm.py        # Extended EM with regularization and early stopping
│   │   ├── match_quality_assessor.py      # Match quality reporting and diagnostics
│   │   ├── quality_metrics.py             # Match quality assessment metrics
│   │   ├── advanced_similarity_metrics.py # Extended similarity measures
│   │   ├── behavioral_embeddings.py       # Activity behavior embedding models
│   │   ├── deep_feature_learning.py       # Autoencoder-based feature extraction
│   │   ├── lsh_blocking.py                # Locality-sensitive hashing for blocking
│   │   ├── household_coordination.py      # Household activity coordination constraints
│   │   ├── probability_calibration.py     # Match probability calibration
│   │   ├── similarity_calculator.py       # (placeholder)
│   │   ├── faiss_matcher.py               # (placeholder)
│   │   └── assignment_optimizer.py        # (placeholder)
│   │
│   ├── utils/                             # Utility modules
│   │   ├── __init__.py
│   │   ├── config_loader.py               # YAML configuration loading and validation
│   │   ├── logging_setup.py               # Phase-specific logging configuration
│   │   ├── data_standardization.py        # Name, address, and field standardization
│   │   ├── feature_engineering.py         # Base feature engineering (200+ features)
│   │   ├── enhanced_feature_engineering.py    # Phase 2 PUMS-RECS feature alignment (100+)
│   │   ├── enhanced_feature_alignment.py      # Phase 3 PUMS-ATUS feature alignment (50+)
│   │   ├── cross_dataset_features.py      # Cross-dataset feature generation
│   │   ├── performance_optimizer.py       # Hardware detection and auto-scaling
│   │   ├── memory_manager.py              # Memory monitoring and garbage collection
│   │   ├── dtype_utils.py                 # DataFrame dtype optimization
│   │   ├── constants.py                   # Project-wide constants and mappings
│   │   ├── file_manager.py                # (placeholder)
│   │   └── json_utils.py                  # (placeholder)
│   │
│   ├── validation/                        # Quality assurance
│   │   ├── __init__.py
│   │   ├── data_validator.py              # General data validation functions
│   │   ├── phase_validators.py            # Phase-specific validation checks
│   │   └── report_generator.py            # HTML validation report generation
│   │
│   └── visualize/                         # Phase 4 visualization modules
│       ├── __init__.py
│       ├── living_system_overview.py      # System-wide Sankey diagrams and metrics
│       ├── building_visualizer.py         # Building characteristic distributions
│       ├── person_visualizer.py           # Demographic analysis and age pyramids
│       ├── activity_visualizer.py         # Daily activity timelines and patterns
│       ├── weather_visualizer.py          # Temperature profiles and weather patterns
│       ├── energy_visualizer.py           # Energy consumption and efficiency analysis
│       ├── household_visualizer.py        # Household dynamics and coordination
│       └── dashboard_generator.py         # HTML dashboard generation
│
├── results/                               # Generated outputs (not tracked in git)
│   └── visualizations/                    # All visualization outputs
│       ├── overview/                      # System overview plots
│       ├── buildings/                     # Building characteristic plots
│       ├── persons/                       # Demographic plots
│       ├── activities/                    # Activity pattern plots
│       ├── weather/                       # Weather plots
│       ├── energy/                        # Energy analysis plots
│       ├── households/                    # Household dynamics plots
│       └── dashboards/                    # Interactive HTML dashboards
│
└── logs/                                  # Runtime logs (not tracked in git)
    ├── main.log                           # Main pipeline log
    ├── phase1.log                         # Phase 1 processing log
    ├── phase2.log                         # Phase 2 processing log
    ├── phase3.log                         # Phase 3 processing log
    ├── phase4.log                         # Phase 4 processing log
    └── matching/                          # Matching diagnostic logs (verbose mode)
```
