# Project Description

## Objective

This project implements a data integration pipeline that constructs behavior-aware synthetic building populations for energy modeling. It combines four U.S. government datasets — PUMS, RECS, ATUS, and NSRDB weather data — using probabilistic record linkage to produce a unified dataset of buildings with realistic occupants, energy characteristics, daily activity schedules, and local weather conditions.

The resulting dataset supports building energy simulation, urban energy planning, and sustainability policy research while preserving individual privacy through the use of synthetic populations derived from real survey microdata.

---

## Theoretical Foundation

### Probabilistic Record Linkage

The core methodology is based on the **Fellegi-Sunter framework** (1969), which treats record linkage as a statistical classification problem. Given two datasets with no shared unique identifier, record pairs are classified into matches and non-matches using likelihood ratios computed from field-level agreement patterns.

For each comparison field, the framework defines two probabilities:

- **m-probability**: the probability of agreement given that the pair is a true match
- **u-probability**: the probability of agreement given that the pair is a non-match (chance agreement)

The match weight for a field is:

```text
Weight = log2(m_probability / u_probability)
```

Total match weights across all fields determine whether a pair is classified as a match, non-match, or uncertain.

### EM Algorithm for Parameter Estimation

The **Expectation-Maximization (EM) algorithm** learns the m and u probabilities without labeled training data. Starting from initial estimates, the algorithm iterates between:

1. **E-step**: Compute posterior match probabilities for each pair given current parameters.
2. **M-step**: Re-estimate m and u probabilities as weighted averages over all pairs.

Iterations continue until parameter changes fall below a convergence threshold. The implementation includes regularization to prevent degenerate solutions and early stopping for efficiency.

### String Similarity Metrics

Several string comparison algorithms support the comparison vectors used in probabilistic matching:

- **Jaro-Winkler**: Effective for names and short strings; accounts for transpositions and rewards matching prefixes.
- **Levenshtein (edit distance)**: Measures minimum character edits; normalized by string length for comparability.
- **Phonetic matching**: Soundex and NYSIIS encode strings by pronunciation, handling spelling variations.
- **Q-gram similarity**: Breaks strings into character subsequences and measures overlap; useful for partial matches.

### Blocking Strategies

To avoid the quadratic cost of comparing all record pairs, the pipeline uses multi-level blocking. Records are grouped by shared attributes (e.g., geographic region, income category, building type), and only pairs within the same block are compared. Multiple blocking passes with progressively relaxed criteria ensure high recall while maintaining computational feasibility.

---

## Data Sources

| Dataset | Description | Records | Year |
| ------- | ----------- | ------- | ---- |
| **PUMS** (American Community Survey Public Use Microdata Sample) | Household demographics, housing characteristics, income, and person-level attributes | 1,444,325 households; 3,405,809 persons | 2023 |
| **RECS** (Residential Energy Consumption Survey) | Detailed building characteristics, appliances, insulation, energy consumption | Smaller national sample used as matching templates | 2020 |
| **ATUS** (American Time Use Survey) | 24-hour activity diaries with demographic profiles | 8,548 respondents; 153,120 activity records | 2023 |
| **NSRDB** (National Solar Radiation Database) | Hourly weather observations by state: temperature, humidity, solar radiation, wind speed | State-level hourly time series | 2023 |

All data sources are real government survey microdata. No synthetic or simulated data is used at any stage.

---

## Pipeline Phases

### Phase 1: PUMS Household-Person Integration

Merges PUMS household and person records using the deterministic SERIALNO key. Feature engineering produces 200+ derived features including household composition categories, income quintiles, building type indicators, energy profile estimates, and geographic blocking keys. Empty households (those without matched persons) are filtered from the output.

### Phase 2: RECS Building Characteristics Matching

Matches each PUMS building to a RECS template using the Fellegi-Sunter probabilistic framework. The EM algorithm estimates m and u probabilities from the data. Enhanced feature engineering generates 100+ comparable features between PUMS and RECS, including multiple categorization levels, composite socioeconomic indices, and interaction terms. A multi-level blocking strategy (7 levels with fallback) ensures all buildings receive a match. The result enriches each building with detailed energy characteristics: square footage, appliance inventory, heating/cooling systems, and energy consumption patterns.

### Phase 3: ATUS Activity Pattern Assignment

Assigns realistic daily activity schedules to each person using optimized k-nearest-neighbor matching against 8,548 real ATUS respondents. Feature alignment generates 50+ comparable features between PUMS persons and ATUS respondents, covering demographics, employment, household context, and time-use predictors. Household coordination logic enforces constraints such as childcare coverage, meal-time overlap, and schedule compatibility among household members. All activity patterns come directly from real time-use diaries.

### Phase 4: Weather Integration

Joins NSRDB weather data to buildings by state. Hourly weather observations are interpolated to align with minute-by-minute activity schedules. The output includes temperature, humidity, solar radiation, wind speed, heating/cooling degree days, and weather condition classifications for each building's location.

---

## Feature Engineering

Feature engineering is central to matching quality across phases:

- **Base features** (`feature_engineering.py`): 200+ features from raw PUMS fields, including household composition, building characteristics, energy profiles, and geographic identifiers.
- **Phase 2 enhanced features** (`enhanced_feature_engineering.py`): 100+ comparable features between PUMS and RECS, with multiple categorization granularities (2-cat through 5-cat), binary indicators, composite indices (SES, density, energy vulnerability), and interaction terms.
- **Phase 3 alignment features** (`enhanced_feature_alignment.py`): 50+ features aligning PUMS persons with ATUS respondents across demographics, employment, household context, and activity-likelihood predictors.
- **Cross-dataset features** (`cross_dataset_features.py`): Additional features bridging dataset-specific variable encodings.

---

## Validation Framework

Each phase produces an HTML validation report covering:

- **Completeness**: Missing value rates and coverage percentages
- **Consistency**: Internal logical checks (e.g., household size vs. person count)
- **Match quality**: Weight distributions, field-level agreement rates, EM convergence diagnostics
- **Coverage**: Percentage of records successfully matched

Validation can run independently of processing (`--validate-only`) and optionally halt the pipeline on quality failures.

---

## Performance Architecture

The pipeline is designed for large-scale data processing:

- **Streaming/sharding**: Each phase reads the prior phase's output as shards and writes its own shard directory with a manifest. This limits peak memory usage regardless of dataset size.
- **Parallel processing**: Enabled by default with auto-detected CPU core count. Configurable via `--workers`.
- **Memory optimization**: Dtype reduction, garbage collection between phases, and configurable memory limits.
- **Checkpointing**: Enabled by default; allows resuming from the last successful phase after failures.
- **Hardware detection**: Automatic calibration of chunk sizes and batch sizes based on available RAM and CPU cores. GPU detection for PyTorch acceleration where applicable.

At full scale (1.4 million buildings), the pipeline processes data in streaming batches to maintain a bounded memory footprint.

---

## Output

The final output is a dataset of buildings where each record contains:

- **Household and person demographics** (from PUMS)
- **Building energy characteristics** (from RECS: square footage, appliances, insulation, fuel type)
- **Daily activity schedules for each occupant** (from ATUS: minute-by-minute time use)
- **Local weather conditions** (from NSRDB: hourly temperature, humidity, solar radiation, wind)

This integrated dataset is suitable for building energy simulation engines, occupant behavior modeling, urban energy demand forecasting, and energy policy analysis.
