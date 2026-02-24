"""
Phase 2: RECS Matching - Probabilistic linkage of PUMS buildings with RECS templates.

This module implements the second phase of the enrichment pipeline, which matches
PUMS buildings from Phase 1 with RECS energy consumption templates using the
Fellegi-Sunter probabilistic record linkage framework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time

from ..utils.config_loader import get_config
from ..utils.logging_setup import setup_logging, log_execution_time, log_memory_usage
from ..validation.data_validator import (
    validate_dataframe,
    validate_required_columns,
    validate_numeric_range,
    generate_data_quality_report
)
from ..data_loading.recs_loader import load_recs_data, prepare_recs_for_matching
from .phase1_pums_integration import load_phase1_output
from ..matching.blocking import create_standard_blocks, create_blocking_report
from ..matching.fellegi_sunter import FellegiSunterMatcher, ComparisonField, create_comparison_summary
from ..matching.em_algorithm import EMAlgorithm, create_em_summary_report
from ..matching.advanced_em_algorithm import AdvancedEMAlgorithm
from ..matching.lsh_blocking import EnsembleLSHBlocker, LSHConfig
from ..matching.probability_calibration import ProbabilityCalibrator, CalibrationConfig
from ..matching.deep_feature_learning import DeepFeatureLearner, AutoencoderConfig
from ..matching.advanced_similarity_metrics import AdvancedSimilarityCalculator
from ..matching.match_quality_assessor import MatchQualityAssessor
from ..matching.quality_metrics import MatchQualityAssessor as AdvancedQualityAssessor, QualityConfig
from ..utils.enhanced_feature_engineering import get_matching_features_list, align_features_for_matching
from ..utils.cross_dataset_features import enhance_dataset_for_matching

logger = logging.getLogger(__name__)


class Phase2RECSMatcher:
    """Orchestrates Phase 2 RECS matching process."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Phase 2 processor."""
        self.config = config or get_config()
        # Get output directories from config paths - improved path handling
        phase2_output_path = Path(self.config.get_data_path('phase2_output'))
        
        # Better logic to determine if path is file or directory
        # Check if it's an existing file or has a clear file extension
        if phase2_output_path.exists() and phase2_output_path.is_file():
            self.output_dir = phase2_output_path.parent
        elif phase2_output_path.suffix in ['.pkl', '.pickle', '.csv', '.json']:
            self.output_dir = phase2_output_path.parent
        else:
            # Assume it's a directory path
            self.output_dir = phase2_output_path
            
        # Handle validation directory with same improved logic
        try:
            phase2_validation_path = Path(self.config.get_data_path('phase2_validation'))
            if phase2_validation_path.exists() and phase2_validation_path.is_file():
                self.validation_dir = phase2_validation_path.parent
            elif phase2_validation_path.suffix in ['.html', '.json', '.txt', '.md']:
                self.validation_dir = phase2_validation_path.parent
            else:
                self.validation_dir = phase2_validation_path
        except:
            # Fallback if phase2_validation not in config
            self.validation_dir = Path('data/validation')
            
        self.params_dir = Path(self.config.get('data_paths.matching_parameters', 'data/matching_parameters'))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 2 specific paths
        self.phase2_output_path = self.output_dir / 'phase2_pums_recs_buildings.pkl'
        self.phase2_params_path = self.params_dir / 'phase2_recs_weights.json'
        self.phase2_validation_path = self.validation_dir / 'phase2_validation_report.html'
        self.phase2_sample_path = self.output_dir / 'phase2_sample.csv'
        self.phase2_metadata_path = self.output_dir / 'phase2_metadata.json'
        
        # Initialize components
        self.matcher = None
        self.em_algorithm = None
        self.quality_assessor = None
        
        # Initialize advanced components - DEFAULT TO FALSE due to numpy type issues
        self.use_advanced_features = self.config.get('phase2.use_advanced_features', False)
        if self.use_advanced_features:
            self.advanced_em = None
            self.lsh_blocker = None
            self.probability_calibrator = None
            self.feature_learner = None
            self.similarity_calculator = None
            logger.info("Advanced matching features ENABLED: LSH blocking, deep learning, probability calibration")
        
        # Memory/streaming safety knobs
        # Number of candidate pairs to sample for EM training (kept small)
        self.em_pair_sample = int(self.config.get('phase2.em_pair_sample', 50000))
        # Maximum number of RECS candidates to evaluate per PUMS record when classifying
        self.max_candidates_per_record = int(self.config.get('phase2.max_candidates_per_record', 200))
        # Random seed for deterministic sampling
        try:
            self._rand_seed = int(self.config.get_random_seed())
        except Exception:
            self._rand_seed = 42
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'pums_buildings_count': 0,
            'recs_templates_count': 0,
            'matches_found': 0,
            'unmatched_buildings': 0
        }
    
    @log_execution_time(logger)
    @log_memory_usage(logger)
    def run(self, sample_size: Optional[int] = None, input_data: Optional[pd.DataFrame] = None,
            save_output: bool = True) -> pd.DataFrame:
        """
        Execute Phase 2 RECS matching.
        
        Args:
            sample_size: Number of buildings to process (None for all)
            input_data: Optional pre-loaded Phase 1 data (for streaming mode)
            
        Returns:
            DataFrame with matched PUMS-RECS buildings
        """
        self.stats['start_time'] = datetime.now()
        logger.info("Starting Phase 2: RECS Matching")
        
        try:
            # Step 1: Load data
            pums_buildings, recs_templates = self._load_data(sample_size, input_data)
            
            # Step 2: Setup matching configuration
            self._setup_matching_configuration()
            
            # Step 3: Create blocked candidate pairs
            blocks = self._create_blocks(pums_buildings, recs_templates)
            
            # Step 3.5: Ensure all PUMS buildings have candidates
            blocks = self._ensure_full_coverage(blocks, pums_buildings, recs_templates)

            # Step 4: Train EM on a sampled subset of candidate pairs to avoid OOM
            em_pairs = self._sample_pairs_for_em(blocks, max_pairs=self.em_pair_sample)
            if not em_pairs:
                logger.warning("No candidate pairs available for EM training; falling back to emergency sampling")
                # Emergency fallback: pair each PUMS with up to 3 random RECS
                em_pairs = self._emergency_pairs(pums_buildings, recs_templates, per_pums=3)
            logger.info(f"EM training on {len(em_pairs):,} candidate pairs")
            em_comparisons = self._compare_specific_pairs(pums_buildings, recs_templates, em_pairs)

            # Step 5: Learn parameters using EM algorithm (on sampled pairs)
            m_probs, u_probs = self._learn_parameters(em_comparisons)

            # Step 6: Calibrate thresholds using the sampled comparisons
            classified_sample = self._classify_pairs(em_comparisons, m_probs, u_probs)
            logger.info("Calibrated thresholds from EM sample; proceeding to stream classification")

            # Step 7: Stream over blocks, classify in-place, and keep best match per PUMS
            matches = self._stream_classify_and_select(pums_buildings, recs_templates, blocks)
            
            # Step 8: Create matched dataset
            matched_buildings = self._create_matched_dataset(
                pums_buildings, recs_templates, matches
            )
            
            # Step 9: Validate and save results (optional for streaming aggregation)
            self.stats['end_time'] = datetime.now()
            if save_output:
                self._validate_and_save(matched_buildings, matches)
            
            logger.info(f"Phase 2 completed successfully. Matched {len(matches)} buildings.")
            
            return matched_buildings
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {str(e)}")
            raise

    def _sample_pairs_for_em(self, blocks: Dict, max_pairs: int) -> List[Tuple[int, int]]:
        """Sample candidate pairs across blocks for EM training.
        Ensures broad coverage while keeping pair count bounded.
        """
        if max_pairs <= 0:
            return []
        rng = np.random.default_rng(self._rand_seed)
        pairs: List[Tuple[int, int]] = []

        # Shuffle block keys for diversity
        block_keys = list(blocks.keys())
        rng.shuffle(block_keys)

        # Simple round-robin sampling from blocks
        i = 0
        while len(pairs) < max_pairs and i < len(block_keys):
            bk = block_keys[i]
            indices = blocks[bk]
            df1_list = indices.get('df1', [])
            df2_list = indices.get('df2', [])
            if not df1_list or not df2_list:
                i += 1
                continue
            # Sample a few df1 from the block
            n_df1 = min(len(df1_list), max(1, max_pairs // max(1, len(block_keys)) // 10))
            sel_df1 = rng.choice(df1_list, size=n_df1, replace=False)
            # For each selected df1, sample up to k df2
            k = 5  # small per-record sampling for EM
            for idx1 in sel_df1:
                if len(pairs) >= max_pairs:
                    break
                take = min(k, len(df2_list))
                sel_df2 = rng.choice(df2_list, size=take, replace=False)
                for idx2 in sel_df2:
                    pairs.append((idx1, idx2))
                    if len(pairs) >= max_pairs:
                        break
            i += 1
        return pairs

    def _emergency_pairs(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame, per_pums: int = 3) -> List[Tuple[int, int]]:
        """Fallback pair generator if blocks are empty: pair each PUMS with a few RECS."""
        rng = np.random.default_rng(self._rand_seed)
        if len(recs_df) == 0 or len(pums_df) == 0:
            return []
        recs_idx = recs_df.index.values
        pairs: List[Tuple[int, int]] = []
        for idx1 in pums_df.index.values:
            take = min(per_pums, len(recs_idx))
            sel = rng.choice(recs_idx, size=take, replace=False)
            for idx2 in sel:
                pairs.append((idx1, idx2))
        return pairs

    def _compare_specific_pairs(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame,
                                candidate_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
        """Compare a provided list of pairs using the matcher."""
        if not candidate_pairs:
            return pd.DataFrame(columns=['idx1', 'idx2', 'weight', 'probability'])
        # Avoid per-field string 'level' columns to keep memory numeric-only
        return self.matcher.compare_datasets(
            pums_df, recs_df, candidate_pairs, include_levels=False
        )

    def _stream_classify_and_select(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame, blocks: Dict) -> pd.DataFrame:
        """Stream through candidate pairs and keep only the best match per PUMS record.
        Vectorized version: builds candidate pairs per block, compares in batches using
        matcher.compare_datasets (numeric-only), then selects max-weight per idx1.
        """
        rng = np.random.default_rng(self._rand_seed)
        max_pair_batch = int(self.config.get('phase2.max_pair_batch', 200_000))

        all_best: Dict[int, Dict] = {}

        for block_key, indices in blocks.items():
            df1_list = indices.get('df1', [])
            df2_list = indices.get('df2', [])
            if not df1_list or not df2_list:
                continue

            # Build candidate pairs for this block with per-PUMS cap
            pairs: List[Tuple[int, int]] = []
            for idx1 in df1_list:
                candidates = df2_list
                if len(candidates) > self.max_candidates_per_record:
                    candidates = rng.choice(candidates, size=self.max_candidates_per_record, replace=False)
                pairs.extend((idx1, idx2) for idx2 in candidates)

            if not pairs:
                continue

            # Compare in bounded batches to avoid memory spikes
            best_block: Dict[int, Dict] = {}
            for start in range(0, len(pairs), max_pair_batch):
                end = min(start + max_pair_batch, len(pairs))
                batch = pairs[start:end]

                comp = self.matcher.compare_datasets(
                    pums_df, recs_df, batch, include_levels=False
                )
                if comp is None or len(comp) == 0:
                    continue

                # For each idx1 in this batch, keep the row with max weight
                # Ensure required columns exist
                needed_cols = {'idx1', 'idx2', 'weight', 'probability', 'classification'}
                missing = needed_cols - set(comp.columns)
                if missing:
                    # If classification missing, fill with 'match' based on weight threshold
                    for c in missing:
                        if c == 'classification':
                            comp['classification'] = 'unknown'
                        elif c in ('probability', 'weight'):
                            comp[c] = np.nan
                # Groupby-idx1 aggregation: idx with maximum weight
                comp = comp.sort_values(['idx1', 'weight'], ascending=[True, False])
                best_rows = comp.groupby('idx1', as_index=False).head(1)

                for _, row in best_rows.iterrows():
                    cur = best_block.get(row['idx1'])
                    if cur is None or (pd.notna(row.get('weight')) and row['weight'] > cur.get('weight', -np.inf)):
                        best_block[row['idx1']] = {
                            'idx1': int(row['idx1']),
                            'idx2': int(row['idx2']),
                            'weight': float(row.get('weight', np.nan)),
                            'probability': float(row.get('probability', np.nan)) if pd.notna(row.get('probability')) else np.nan,
                            'classification': row.get('classification', 'unknown')
                        }

            # Merge best from this block into global best
            for k, v in best_block.items():
                cur = all_best.get(k)
                if cur is None or (pd.notna(v.get('weight')) and v['weight'] > cur.get('weight', -np.inf)):
                    all_best[k] = v

        if not all_best:
            logger.warning("No best matches found during streaming classification; returning empty result")
            return pd.DataFrame(columns=['idx1', 'idx2', 'weight', 'probability', 'classification'])

        return pd.DataFrame(list(all_best.values()))
    
    def _load_data(self, sample_size: Optional[int], input_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load PUMS buildings from Phase 1 and RECS templates."""
        logger.info("Loading data for Phase 2")
        
        # Load Phase 1 output or use provided data
        if input_data is not None:
            # Use pre-loaded data for streaming mode
            pums_buildings = input_data
            phase1_metadata = {'source': 'streaming'}
            logger.info(f"Using pre-loaded data with {len(pums_buildings)} PUMS buildings")
        else:
            # Load from disk
            pums_buildings, phase1_metadata = load_phase1_output()
            logger.info(f"Loaded {len(pums_buildings)} PUMS buildings from Phase 1")
        
        # Apply sample size if specified
        if sample_size and sample_size < len(pums_buildings):
            logger.info(f"Sampling {sample_size} buildings")
            pums_buildings = pums_buildings.sample(
                n=sample_size, 
                random_state=self.config.get_random_seed()
            )
        
        self.stats['pums_buildings_count'] = len(pums_buildings)
        
        # Load RECS templates
        recs_templates = load_recs_data()
        logger.info(f"Loaded {len(recs_templates)} RECS templates")
        self.stats['recs_templates_count'] = len(recs_templates)
        
        # Prepare RECS data for matching
        recs_templates = prepare_recs_for_matching(
            recs_templates, 
            pums_buildings.columns.tolist()
        )
        
        # Enhance both datasets with cross-dataset features
        logger.info("Creating enhanced cross-dataset features for better matching...")
        pums_buildings = enhance_dataset_for_matching(pums_buildings, 'pums', 'recs')
        recs_templates = enhance_dataset_for_matching(recs_templates, 'recs', 'pums')
        
        # Align features between PUMS and RECS using enhanced feature engineering
        pums_buildings, recs_templates = align_features_for_matching(
            pums_buildings, recs_templates
        )
        logger.info(f"Aligned features between datasets - {len(set(pums_buildings.columns) & set(recs_templates.columns))} common features")
        
        return pums_buildings, recs_templates
    
    def _setup_matching_configuration(self):
        """Configure matching fields and methods with advanced enhancements."""
        logger.info("Setting up enhanced matching configuration")
        
        # Get the enhanced features list from the feature engineering module
        enhanced_features = get_matching_features_list()
        logger.info(f"Using {len(enhanced_features)} enhanced features for matching")
        
        # Create comparison fields from enhanced features
        # Determine which features are numeric vs categorical
        numeric_features = [
            'household_size', 'income_decile', 'total_rooms', 
            'num_bedrooms', 'building_age', 'rooms_per_person'
        ]
        
        comparison_fields = []
        for feature in enhanced_features:
            # Determine field type and comparison method
            if feature in numeric_features:
                field_type = 'numeric'
                comparison_method = 'numeric'
            else:
                field_type = 'categorical'
                comparison_method = 'exact'
            
            # Create comparison field
            comparison_fields.append(
                ComparisonField(
                    name=feature,
                    field_type=field_type,
                    comparison_method=comparison_method,
                    weight=1.0  # Let EM learn actual weights
                )
            )
        
        logger.info(f"Created {len(comparison_fields)} comparison fields from enhanced features")
        
        # Filter to only fields that exist in the data
        # This will be done dynamically based on available columns
        
        # Initialize matcher
        self.matcher = FellegiSunterMatcher(comparison_fields)
        
        # Initialize EM algorithm
        field_names = [field.name for field in comparison_fields]
        
        if self.use_advanced_features:
            # Use advanced EM with Adam optimizer and multiple initializations
            logger.info("Using advanced EM algorithm with Adam optimizer")
            from ..matching.advanced_em_algorithm import EMConfig
            em_config = EMConfig(
                n_initializations=10,
                learning_rate=0.01,
                momentum=0.9,
                adam_beta1=0.9,
                adam_beta2=0.999,
                use_aitken_acceleration=True,
                use_mixture_model=True,
                prior_match=self.config.get('phase2.em_prior_match', 0.1)
            )
            self.advanced_em = AdvancedEMAlgorithm(field_names, em_config)
            self.em_algorithm = self.advanced_em  # Use advanced version
            
            # Initialize advanced similarity calculator
            logger.info("Initializing advanced similarity metrics")
            from ..matching.advanced_similarity_metrics import SimilarityConfig
            sim_config = SimilarityConfig(
                use_neural_similarity=True,
                use_attention=True,
                use_contextual=True,
                ensemble_methods=['neural', 'statistical', 'fuzzy']
            )
            self.similarity_calculator = AdvancedSimilarityCalculator(sim_config)
            
            # Initialize probability calibrator
            logger.info("Setting up probability calibration")
            calib_config = CalibrationConfig(
                method='ensemble',
                ensemble_methods=['platt', 'isotonic', 'beta'],
                cv_folds=5
            )
            self.probability_calibrator = ProbabilityCalibrator(calib_config)
        else:
            # Use standard EM algorithm
            self.em_algorithm = EMAlgorithm(
                field_names,
                prior_match=self.config.get('phase2.em_prior_match', 0.1)
            )
    
    def _create_blocks(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame) -> Dict:
        """Create blocked candidate pairs using LSH if enabled."""
        logger.info("Creating blocking structure")
        
        if self.use_advanced_features and self.config.get('phase2.use_lsh_blocking', True):
            # Use LSH blocking for better candidate generation
            logger.info("Using LSH ensemble blocking for efficient candidate generation")
            
            lsh_config = LSHConfig(
                n_hash_functions=128,
                n_bands=32,
                similarity_threshold=0.3,
                use_ensemble=True,
                ensemble_methods=['minhash', 'simhash', 'random_projection'],
                adaptive_threshold=True,
                min_candidates_per_record=5,
                max_candidates_per_record=100
            )
            
            self.lsh_blocker = EnsembleLSHBlocker(lsh_config)
            
            # Identify text and numeric columns
            text_columns = ['STATE', 'building_type_simple', 'climate_zone']
            numeric_columns = ['household_size', 'income_quintile', 'building_age', 'num_bedrooms']
            
            # Generate candidate pairs using LSH
            candidate_pairs = self.lsh_blocker.generate_candidate_pairs(
                pums_df, recs_df,
                text_columns=text_columns,
                numeric_columns=numeric_columns
            )
            
            # Convert to blocks format
            blocks = self._convert_pairs_to_blocks(candidate_pairs)
            
            logger.info(f"LSH blocking generated {len(candidate_pairs)} candidate pairs")
        else:
            # Use standard blocking
            blocks = create_standard_blocks(
                pums_df, 
                recs_df,
                self.config.get('phase2.blocking', {})
            )
        
        # Log blocking statistics
        report = create_blocking_report(blocks, pums_df, recs_df)
        logger.info(f"Blocking report:\n{report}")
        
        return blocks
    
    def _convert_pairs_to_blocks(self, pairs_df: pd.DataFrame) -> Dict:
        """Convert candidate pairs DataFrame to blocks format."""
        blocks = {}
        
        # Group by first index to create blocks
        for idx1, group in pairs_df.groupby('idx1'):
            block_key = f'lsh_block_{idx1}'
            blocks[block_key] = {
                'df1': [idx1],
                'df2': group['idx2'].tolist()
            }
        
        return blocks
    
    def _ensure_full_coverage(self, blocks: Dict, pums_df: pd.DataFrame, 
                             recs_df: pd.DataFrame) -> Dict:
        """Ensure every PUMS building has at least some RECS candidates."""
        # Get all PUMS indices that are in blocks
        covered_pums = set()
        for block_key, indices in blocks.items():
            covered_pums.update(indices['df1'])
        
        # Find uncovered PUMS buildings
        all_pums = set(pums_df.index)
        uncovered_pums = all_pums - covered_pums
        
        if uncovered_pums:
            logger.warning(f"Found {len(uncovered_pums)} PUMS buildings without candidates - adding fallback block")
            
            # Create a fallback block that pairs uncovered buildings with a sample of RECS
            # Use the most common RECS templates as fallback candidates
            fallback_recs = recs_df.sample(min(10, len(recs_df)), random_state=42).index.tolist()
            
            # Add fallback block
            blocks['_fallback_'] = {
                'df1': list(uncovered_pums),
                'df2': fallback_recs
            }
            
            logger.info(f"Added fallback block with {len(uncovered_pums)} PUMS x {len(fallback_recs)} RECS pairs")
        
        return blocks
    
    def _compare_candidates(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame, 
                           blocks: Dict) -> pd.DataFrame:
        """Compare all candidate pairs with enhanced similarity metrics."""
        logger.info("Comparing candidate pairs with enhanced methods")
        
        # Use deep feature learning if enabled
        if self.use_advanced_features and self.config.get('phase2.use_deep_features', True):
            logger.info("Training deep feature learner for enhanced representations")
            
            # Initialize and train deep feature learner
            if self.feature_learner is None:
                autoencoder_config = AutoencoderConfig(
                    encoding_dim=32,
                    hidden_dims=[128, 64],
                    use_variational=True,
                    use_denoising=True,
                    use_attention=True,
                    n_epochs=50
                )
                self.feature_learner = DeepFeatureLearner(autoencoder_config)
            
            # Train on combined dataset sample
            train_sample = pd.concat([
                pums_df.sample(min(1000, len(pums_df)), random_state=42),
                recs_df.sample(min(1000, len(recs_df)), random_state=42)
            ])
            
            # Select numeric and categorical columns for training
            feature_cols = [col for col in train_sample.columns 
                          if col not in ['persons', 'index', 'SERIALNO']]
            train_data = train_sample[feature_cols].fillna(0)
            
            self.feature_learner.train_autoencoder(train_data)
            
            # Extract learned features for all records
            logger.info("Extracting learned features for matching")
            pums_features = self.feature_learner.extract_features(pums_df[feature_cols].fillna(0))
            recs_features = self.feature_learner.extract_features(recs_df[feature_cols].fillna(0))
            
            # Add learned features to dataframes
            for i in range(pums_features.shape[1]):
                pums_df[f'learned_feature_{i}'] = pums_features[:, i]
                recs_df[f'learned_feature_{i}'] = recs_features[:, i]
        
        # Generate list of candidate pairs from blocks
        candidate_pairs = []
        for block_key, indices in blocks.items():
            for idx1 in indices['df1']:
                for idx2 in indices['df2']:
                    candidate_pairs.append((idx1, idx2))
        
        logger.info(f"Comparing {len(candidate_pairs)} candidate pairs")
        
        # Check if we have pairs to compare
        if len(candidate_pairs) == 0:
            logger.warning("No candidate pairs found in blocks!")
            # Create fallback pairs - pair each PUMS building with at least one RECS template
            logger.info("Creating emergency fallback pairs to ensure 100% match rate")
            
            # Get a representative sample of RECS templates
            recs_sample_size = min(5, len(recs_df))
            recs_sample_indices = recs_df.sample(n=recs_sample_size, random_state=42).index.tolist()
            
            # Pair each PUMS building with the sample of RECS templates
            for pums_idx in pums_df.index:
                for recs_idx in recs_sample_indices:
                    candidate_pairs.append((pums_idx, recs_idx))
            
            logger.info(f"Created {len(candidate_pairs)} emergency fallback pairs")
            
            if len(candidate_pairs) == 0:
                # Last resort - return empty DataFrame with expected structure
                logger.error("Could not create any candidate pairs!")
                return pd.DataFrame(columns=['idx1', 'idx2', 'weight', 'probability'])
        
        # Compare pairs with enhanced similarity metrics if enabled
        if self.use_advanced_features and self.similarity_calculator is not None:
            logger.info("Using advanced similarity metrics for comparison")
            
            # Prepare data for neural similarity
            self.similarity_calculator.prepare_data(pums_df, recs_df)
            
            # Enhance comparison with neural and ensemble methods
            comparison_results = []
            
            for idx1, idx2 in candidate_pairs:
                record1 = pums_df.loc[idx1]
                record2 = recs_df.loc[idx2]
                
                # Calculate advanced similarities
                similarities = self.similarity_calculator.calculate_similarity(
                    record1, record2
                )
                
                # Add to results
                result = {
                    'idx1': idx1,
                    'idx2': idx2,
                    **similarities
                }
                comparison_results.append(result)
            
            comparison_results = pd.DataFrame(comparison_results)
        else:
            # Use standard comparison
            # Use numeric-only comparisons; skip string level columns to reduce memory
            comparison_results = self.matcher.compare_datasets(
                pums_df, recs_df, candidate_pairs, include_levels=False
            )
        
        return comparison_results
    
    def _learn_parameters(self, comparison_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Learn m and u probabilities using advanced EM algorithm."""
        
        if self.use_advanced_features and isinstance(self.em_algorithm, AdvancedEMAlgorithm):
            logger.info("Learning parameters with advanced EM algorithm (Adam optimizer, multiple initializations)")
            
            # Run advanced EM with multiple initializations
            results = self.em_algorithm.fit(comparison_data)
            
            # Extract best parameters
            m_probs = results['m_probabilities']
            u_probs = results['u_probabilities']
            
            logger.info(f"Advanced EM converged in {results['iterations']} iterations")
            logger.info(f"Final log-likelihood: {results['log_likelihood']:.4f}")
        else:
            logger.info("Learning parameters with standard EM algorithm")
            
            # Run standard EM algorithm
            m_probs, u_probs = self.em_algorithm.fit(
                comparison_data,
                max_iterations=self.config.get('phase2.em_max_iterations', 100),
                tolerance=self.config.get('phase2.em_tolerance', 0.0001),
                verbose=True
            )
        
        # Log results
        em_report = create_em_summary_report(self.em_algorithm)
        logger.info(f"EM Algorithm Results:\n{em_report}")
        
        # Save parameters
        self._save_parameters(m_probs, u_probs)
        
        return m_probs, u_probs
    
    def _classify_pairs(self, comparison_data: pd.DataFrame, 
                       m_probs: Dict, u_probs: Dict) -> pd.DataFrame:
        """Apply learned parameters to classify pairs with probability calibration."""
        logger.info("Classifying pairs with learned parameters and calibration")
        
        # Handle empty comparison data
        if len(comparison_data) == 0:
            logger.warning("No comparison data to classify")
            return comparison_data
        
        # Set learned probabilities
        self.matcher.set_probabilities(m_probs, u_probs)
        
        # Recalculate weights with learned parameters
        weights = []
        for idx, row in comparison_data.iterrows():
            agreement_vector = []
            for field in self.em_algorithm.field_names:
                if f'{field}_similarity' in row:
                    agreement_vector.append(row[f'{field}_similarity'])
            
            weight = self.matcher.compute_match_weight(agreement_vector)
            weights.append(weight)
        
        comparison_data['weight'] = weights
        
        # Calculate raw probabilities
        raw_probabilities = comparison_data['weight'].apply(
            lambda w: self.matcher.compute_match_probability(w)
        ).values
        
        # Apply probability calibration if enabled
        if self.use_advanced_features and self.probability_calibrator is not None:
            logger.info("Applying probability calibration for better confidence estimates")
            
            # Create pseudo-labels based on weight thresholds for calibration
            # This is unsupervised - we use high/low confidence samples
            pseudo_labels = (comparison_data['weight'] > 0).astype(int).values
            
            # Fit calibrator on current data
            self.probability_calibrator.fit(raw_probabilities, pseudo_labels)
            
            # Apply calibration
            calibrated_probs, confidence_intervals = self.probability_calibrator.transform(raw_probabilities)
            comparison_data['probability'] = calibrated_probs
            comparison_data['probability_lower'] = confidence_intervals[:, 0]
            comparison_data['probability_upper'] = confidence_intervals[:, 1]
            
            # Get calibration metrics
            calib_report = self.probability_calibrator.get_calibration_report()
            logger.info(f"Calibration metrics: ECE={calib_report['metrics'].get('ece', 0):.4f}")
        else:
            comparison_data['probability'] = raw_probabilities
        
        # Set classification thresholds
        upper, lower = self.matcher.get_optimal_thresholds(
            comparison_data,
            target_precision=self.config.get('phase2.target_precision', 0.95)
        )
        self.matcher.set_thresholds(upper, lower)
        
        # Classify pairs
        comparison_data['classification'] = comparison_data['weight'].apply(
            self.matcher.classify_pair
        )
        
        # Log classification summary
        summary = create_comparison_summary(comparison_data)
        logger.info(f"Classification Summary:\n{summary}")
        
        return comparison_data
    
    def _select_best_matches(self, classified_pairs: pd.DataFrame) -> pd.DataFrame:
        """Select best match for each PUMS building, ensuring 100% match rate."""
        logger.info("Selecting best matches for ALL buildings")
        
        # Use ALL pairs, not just high-scoring ones - we want 100% match rate
        all_pairs = classified_pairs.copy()
        
        if len(all_pairs) == 0:
            logger.warning("No comparison pairs found!")
            return pd.DataFrame()
        
        # Get all unique PUMS building indices
        all_pums_indices = all_pairs['idx1'].unique()
        logger.info(f"Ensuring matches for all {len(all_pums_indices)} PUMS buildings")
        
        # For each PUMS building, select best RECS template (even if low score)
        matches = []
        
        for idx1 in all_pums_indices:
            building_matches = all_pairs[all_pairs['idx1'] == idx1]
            
            # Select match with highest weight (even if it's negative/low)
            best_match = building_matches.loc[building_matches['weight'].idxmax()]
            matches.append(best_match)
            
            # Log if this is a low-quality match
            if best_match['classification'] == 'non-match':
                logger.debug(f"Building {idx1} matched with low confidence (weight: {best_match['weight']:.3f})")
        
        matches_df = pd.DataFrame(matches)
        logger.info(f"Selected {len(matches_df)} matches (100% match rate target)")
        self.stats['matches_found'] = len(matches_df)
        
        # Log match quality distribution
        quality_dist = matches_df['classification'].value_counts()
        logger.info(f"Match quality distribution: {quality_dist.to_dict()}")
        
        return matches_df
    
    def _create_matched_dataset(self, pums_df: pd.DataFrame, recs_df: pd.DataFrame,
                               matches: pd.DataFrame) -> pd.DataFrame:
        """Create final matched PUMS-RECS dataset."""
        logger.info("Creating matched dataset")
        
        # Start with PUMS buildings
        matched_buildings = pums_df.copy()
        
        # Add match indicator
        matched_buildings['has_recs_match'] = False
        matched_buildings['recs_template_id'] = None
        matched_buildings['match_weight'] = None
        matched_buildings['match_probability'] = None
        
        # Process matches
        for _, match in matches.iterrows():
            pums_idx = match['idx1']
            recs_idx = match['idx2']
            
            # Get RECS template data
            recs_record = recs_df.loc[recs_idx]
            
            # Update PUMS building with RECS data
            matched_buildings.loc[pums_idx, 'has_recs_match'] = True
            matched_buildings.loc[pums_idx, 'recs_template_id'] = recs_record['template_id']
            matched_buildings.loc[pums_idx, 'match_weight'] = match['weight']
            matched_buildings.loc[pums_idx, 'match_probability'] = match['probability']
            
            # Add RECS-specific columns
            recs_columns = [
                'square_footage', 'total_rooms', 'energy_use_intensity',
                'electricity_kwh', 'natural_gas_btu', 'total_energy_cost_annual',
                'heating_equipment', 'cooling_type', 'water_heater_fuel'
            ]
            
            for col in recs_columns:
                if col in recs_record:
                    col_name = f'recs_{col}'
                    if col_name not in matched_buildings.columns:
                        matched_buildings[col_name] = None
                    matched_buildings.loc[pums_idx, col_name] = recs_record[col]
        
        # Count unmatched
        self.stats['unmatched_buildings'] = (~matched_buildings['has_recs_match']).sum()
        
        logger.info(f"Matched {matched_buildings['has_recs_match'].sum()} buildings")
        logger.info(f"Unmatched: {self.stats['unmatched_buildings']} buildings")

        # Ensure EV charger features propagate from PUMS through the matched output
        ev_cols = ['ev_charger_prob', 'has_ev_charger', 'charger_level',
                   'charger_capacity_kw', 'ev_charger_level2_prob']
        for col in ev_cols:
            if col in pums_df.columns and col not in matched_buildings.columns:
                matched_buildings[col] = pums_df[col]

        return matched_buildings
    
    def _validate_and_save(self, matched_buildings: pd.DataFrame, matches: pd.DataFrame):
        """Validate results and save outputs with enhanced quality assessment."""
        logger.info("Validating and saving Phase 2 results with advanced quality metrics")
        
        # Use advanced quality assessment if enabled
        if self.use_advanced_features:
            logger.info("Performing advanced quality assessment")
            
            # Configure quality assessment
            quality_config = QualityConfig(
                calculate_all_metrics=True,
                top_k_values=[1, 3, 5, 10],
                confidence_thresholds=[0.5, 0.7, 0.8, 0.9, 0.95],
                generate_plots=True,
                save_reports=True,
                report_format='html',
                output_dir=str(self.validation_dir)
            )
            
            # Initialize advanced quality assessor
            advanced_assessor = AdvancedQualityAssessor(quality_config)
            
            # Evaluate matches
            quality_metrics = advanced_assessor.evaluate_matches(
                matches,
                ground_truth=None,  # No ground truth available
                comparison_data=matches  # Use match data for field analysis
            )
            
            # Log key metrics
            logger.info(f"Match Quality Metrics:")
            logger.info(f"  - Average weight: {quality_metrics['basic']['avg_match_weight']:.3f}")
            logger.info(f"  - Average probability: {quality_metrics['basic']['avg_match_probability']:.3f}")
            logger.info(f"  - Weight entropy: {quality_metrics['distribution']['weight_entropy']:.3f}")
            logger.info(f"  - Uniqueness score: {quality_metrics['cardinality']['uniqueness_score']:.3f}")
            logger.info(f"  - Separation score: {quality_metrics['unsupervised']['separation_score']:.3f}")
            logger.info(f"  - High confidence ratio: {quality_metrics['unsupervised']['high_confidence_ratio']:.3f}")
            
            # Check for quality issues
            if quality_metrics['diagnostics']['n_issues'] > 0:
                logger.warning(f"Found {quality_metrics['diagnostics']['n_issues']} quality issues:")
                for issue in quality_metrics['diagnostics']['issues']:
                    logger.warning(f"  - {issue['type']}: {issue['details']}")
        
        # Initialize quality assessor
        self.quality_assessor = MatchQualityAssessor()
        
        # Assess match quality
        quality_report = self.quality_assessor.assess_matches(
            matched_buildings, matches
        )
        
        # Save main output
        logger.info(f"Saving matched buildings to {self.phase2_output_path}")
        matched_buildings.to_pickle(self.phase2_output_path)
        
        # Save sample for inspection
        sample_size = min(100, len(matched_buildings))
        sample = matched_buildings.head(sample_size)
        sample.to_csv(self.phase2_sample_path, index=False)
        logger.info(f"Saved sample of {sample_size} buildings to {self.phase2_sample_path}")
        
        # Save metadata
        metadata = self._create_metadata(matched_buildings, quality_report)
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        metadata = convert_numpy_types(metadata)
        with open(self.phase2_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate validation report
        self._generate_validation_report(matched_buildings, quality_report)
        
        logger.info("Phase 2 outputs saved successfully")
    
    def _save_parameters(self, m_probs: Dict, u_probs: Dict):
        """Save learned matching parameters."""
        parameters = {
            'phase': 'phase2_recs',
            'timestamp': datetime.now().isoformat(),
            'm_probabilities': m_probs,
            'u_probabilities': u_probs,
            'em_diagnostics': self.em_algorithm.get_convergence_diagnostics()
        }
        
        with open(self.phase2_params_path, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        logger.info(f"Saved matching parameters to {self.phase2_params_path}")
    
    def _create_metadata(self, matched_buildings: pd.DataFrame, 
                        quality_report: Dict) -> Dict:
        """Create metadata for Phase 2 output."""
        metadata = {
            'phase': 'phase2_recs_matching',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            'statistics': {
                'input_pums_buildings': self.stats['pums_buildings_count'],
                'input_recs_templates': self.stats['recs_templates_count'],
                'matched_buildings': self.stats['matches_found'],
                'unmatched_buildings': self.stats['unmatched_buildings'],
                'match_rate': self.stats['matches_found'] / self.stats['pums_buildings_count']
            },
            'quality_metrics': quality_report,
            'output_columns': matched_buildings.columns.tolist(),
            'memory_usage_mb': matched_buildings.memory_usage().sum() / 1024**2
        }
        
        return metadata
    
    def _generate_validation_report(self, matched_buildings: pd.DataFrame,
                                   quality_report: Dict):
        """Generate HTML validation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 2 RECS Matching Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; }}
                .warning {{ color: #ff9800; }}
                .error {{ color: #f44336; }}
                .success {{ color: #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Phase 2: RECS Matching Validation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h2>Summary Statistics</h2>
                <ul>
                    <li>Total PUMS Buildings: {self.stats['pums_buildings_count']:,}</li>
                    <li>Total RECS Templates: {self.stats['recs_templates_count']:,}</li>
                    <li>Matched Buildings: {self.stats['matches_found']:,}</li>
                    <li>Unmatched Buildings: {self.stats['unmatched_buildings']:,}</li>
                    <li>Match Rate: {self.stats['matches_found'] / self.stats['pums_buildings_count']:.1%}</li>
                </ul>
            </div>
            
            <h2>Match Quality Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add quality metrics
        for metric, value in quality_report.items():
            if isinstance(value, (int, float)):
                html_content += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{value:,.2f}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>Processing Details</h2>
            <ul>
                <li>Processing Duration: {:.1f} seconds</li>
                <li>Output File: {}</li>
                <li>Parameters File: {}</li>
            </ul>
        </body>
        </html>
        """.format(
            (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            self.phase2_output_path,
            self.phase2_params_path
        )
        
        with open(self.phase2_validation_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report saved to {self.phase2_validation_path}")


def main():
    """Run Phase 2 RECS matching."""
    # Setup logging
    setup_logging('phase2')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize processor
        processor = Phase2RECSMatcher()
        
        # Run with sample for testing
        sample_size = 100  # Adjust as needed
        logger.info(f"Running Phase 2 with sample size: {sample_size}")
        
        # Execute Phase 2
        matched_buildings = processor.run(sample_size=sample_size)
        
        logger.info("Phase 2 completed successfully")
        
        # Display summary
        print("\nPhase 2 Summary:")
        print(f"Matched buildings: {len(matched_buildings)}")
        print(f"Match rate: {matched_buildings['has_recs_match'].mean():.1%}")
        print(f"Output saved to: {processor.phase2_output_path}")
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
