"""
Constants for PUMS Enrichment Pipeline.

This module centralizes all magic numbers and constants used throughout the pipeline
for better maintainability and configurability.
"""

# ============================================================================
# MATCHING THRESHOLDS AND PARAMETERS
# ============================================================================

# Fellegi-Sunter Agreement Levels (based on empirical analysis)
SIMILARITY_DISAGREE_THRESHOLD = 0.66  # Below this = disagree
SIMILARITY_PARTIAL_THRESHOLD = 0.88   # Below this = partial agreement
SIMILARITY_MOSTLY_THRESHOLD = 0.94    # Below this = mostly agree
# Above SIMILARITY_MOSTLY_THRESHOLD = full agreement

# EM Algorithm Parameters
EM_DEFAULT_PRIOR_MATCH = 0.1         # Prior probability that a random pair is a match
EM_MIN_PROBABILITY = 0.001           # Minimum m/u probability to avoid log(0)
EM_MAX_PROBABILITY = 0.999           # Maximum m/u probability to avoid log(0)
EM_MIN_SEPARATION = 0.1              # Minimum separation between m and u probabilities
EM_MAX_ITERATIONS = 100              # Maximum EM iterations before stopping
EM_CONVERGENCE_THRESHOLD = 0.0001   # Parameter change threshold for convergence
EM_REGULARIZATION_LAMBDA = 0.01     # Regularization parameter

# Default m and u probabilities for different field types
DEFAULT_M_PROB_NAME = 0.9           # Match probability for names
DEFAULT_M_PROB_GENERAL = 0.8        # Match probability for general fields
DEFAULT_U_PROB = 0.1                # Non-match probability default

# Blocking Parameters
MIN_BLOCK_SIZE = 10                 # Minimum records per block
MAX_BLOCK_SIZE = 10000             # Maximum records per block
BLOCKING_COVERAGE_THRESHOLD = 0.95  # Minimum coverage for blocking strategy

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Sampling and Chunking
DEFAULT_SAMPLE_SIZE = 100           # Default number of buildings if not specified
HOUSEHOLD_CHUNK_SIZE = 10000        # Chunk size for reading household data
PERSON_CHUNK_SIZE_SMALL = 10000     # Person chunk size for small samples (<100)
PERSON_CHUNK_SIZE_MEDIUM = 25000    # Person chunk size for medium samples (<1000)
PERSON_CHUNK_SIZE_LARGE = 50000     # Person chunk size for large samples (>=1000)
SMALL_SAMPLE_THRESHOLD = 100        # Threshold for small sample optimizations
MEDIUM_SAMPLE_THRESHOLD = 1000      # Threshold for medium sample optimizations

# Expected Values for Validation
EXPECTED_PERSONS_PER_HOUSEHOLD = 3  # Typical household size for early stopping
MIN_HOUSEHOLD_SIZE = 1               # Minimum valid household size
MAX_HOUSEHOLD_SIZE = 20              # Maximum reasonable household size
MIN_AGE = 0                         # Minimum valid age
MAX_AGE = 120                       # Maximum reasonable age

# ============================================================================
# PHASE-SPECIFIC PARAMETERS
# ============================================================================

# Phase 1: PUMS Integration
PHASE1_HOUSING_UNIT_FILTER = {'RT': 'H', 'TYPEHUGQ': 1}  # Filter for housing units

# Phase 2: RECS Matching
PHASE2_MIN_MATCH_RATE = 0.95       # Minimum required match rate
PHASE2_WEIGHT_THRESHOLD = 10       # Weight threshold for high-quality matches

# Phase 3: ATUS Matching
PHASE3_KNN_NEIGHBORS = 5           # Number of nearest neighbors for fallback
PHASE3_MIN_ACTIVITY_HOURS = 23.5   # Minimum hours of activities per day
PHASE3_MAX_ACTIVITY_HOURS = 24.5   # Maximum hours of activities per day

# Phase 4: Weather Integration
PHASE4_BASE_TEMPERATURE_F = 65     # Base temperature for degree day calculations

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

LOG_MEMORY_THRESHOLD_MB = 1000     # Warn if memory usage exceeds this
LOG_TIME_THRESHOLD_SEC = 300       # Warn if processing takes longer than this
PROGRESS_BAR_UPDATE_FREQ = 100     # Update progress bar every N records

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

MAX_MISSING_PERCENT = 0.1          # Maximum acceptable missing data percentage
MIN_UNIQUE_VALUES = 2               # Minimum unique values for categorical fields
MAX_DUPLICATE_PERCENT = 0.05       # Maximum acceptable duplicate percentage

# ============================================================================
# FILE FORMATS AND EXTENSIONS
# ============================================================================

VALID_DATA_EXTENSIONS = ['.csv', '.dat', '.txt', '.pkl', '.pickle']
VALID_CONFIG_EXTENSIONS = ['.yaml', '.yml', '.json']
OUTPUT_PICKLE_PROTOCOL = 4         # Pickle protocol version for outputs

# ============================================================================
# THREADING AND PERFORMANCE
# ============================================================================

MAX_THREADS = 4                    # Maximum threads for parallel processing
CACHE_SIZE = 10000                 # LRU cache size for expensive operations