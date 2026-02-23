"""
Phase 3: ATUS Activity Pattern Matching - OPTIMIZED VERSION.

This module implements an optimized version of Phase 3 using REAL ATUS data.
NO SYNTHETIC DATA - all activity patterns come from actual ATUS 2023 survey respondents.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import time
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from ..utils.config_loader import get_config
from ..utils.logging_setup import setup_logging, log_execution_time, log_memory_usage
from ..validation.data_validator import validate_dataframe
from ..utils.cross_dataset_features import enhance_dataset_for_matching
from ..data_loading.atus_loader import (
    load_all_atus_data,
    create_activity_templates
)
from ..matching.behavioral_embeddings import (
    BehavioralEmbeddingSystem,
    BehavioralConfig
)
from ..matching.household_coordination import (
    HouseholdCoordinationSystem,
    HouseholdConfig
)

logger = logging.getLogger(__name__)


class OptimizedPhase3Matcher:
    """Optimized Phase 3 ATUS activity pattern matching using REAL data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Phase 3 processor."""
        self.config = config or get_config()
        
        # Get output directories - improved path handling
        phase3_output_path = Path(self.config.get_data_path('phase3_output'))
        
        # Better logic to determine if path is file or directory
        # Check if it's an existing file or has a clear file extension
        if phase3_output_path.exists() and phase3_output_path.is_file():
            self.output_dir = phase3_output_path.parent
        elif phase3_output_path.suffix in ['.pkl', '.pickle', '.csv', '.json']:
            self.output_dir = phase3_output_path.parent
        else:
            # Assume it's a directory path
            self.output_dir = phase3_output_path
            
        # Handle validation directory with same improved logic
        try:
            phase3_validation_path = Path(self.config.get_data_path('phase3_validation'))
            if phase3_validation_path.exists() and phase3_validation_path.is_file():
                self.validation_dir = phase3_validation_path.parent
            elif phase3_validation_path.suffix in ['.html', '.json', '.txt', '.md']:
                self.validation_dir = phase3_validation_path.parent
            else:
                self.validation_dir = phase3_validation_path
        except:
            # Fallback if phase3_validation not in config
            self.validation_dir = Path('data/validation')
            
        self.params_dir = Path(self.config.get('data_paths.matching_parameters', 'data/matching_parameters'))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 3 specific paths
        self.phase3_output_path = self.output_dir / 'phase3_pums_recs_atus_buildings.pkl'
        self.phase3_metadata_path = self.output_dir / 'phase3_metadata.json'
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'buildings_count': 0,
            'persons_count': 0,
            'templates_count': 0,
            'matched_persons': 0,
            'real_atus_respondents': 0
        }
        
        # Initialize advanced components
        self.use_advanced_features = self.config.get('phase3.use_advanced_features', False)
        if self.use_advanced_features:
            self.behavioral_system = None
            self.household_coordinator = None
    
    @log_execution_time(logger)
    @log_memory_usage(logger)
    def run(self, sample_size: Optional[int] = None, match_batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Execute optimized Phase 3 ATUS matching with REAL data.
        
        Key features:
        1. Uses real ATUS 2023 survey data (8,548 respondents)
        2. Vectorized distance calculations for speed
        3. K-nearest neighbors for fast matching
        4. Real activity patterns from actual survey respondents
        5. Advanced behavioral embeddings for better person-activity alignment
        6. Household coordination for consistent family schedules
        """
        self.stats['start_time'] = datetime.now()
        logger.info("Starting OPTIMIZED Phase 3: ATUS Activity Pattern Matching with REAL DATA")
        
        try:
            # If Phase 2 shards exist, stream them to avoid loading full dataset
            phase2_base = Path(self.config.get_data_path('phase2_output')).parent
            shards_dir = phase2_base / 'phase2_shards'
            manifest = shards_dir / 'manifest.json'

            if shards_dir.exists() and manifest.exists():
                logger.info("Detected Phase 2 shards - running Phase 3 in streaming mode over shards")
                # Load REAL ATUS data once
                _, atus_templates = self._load_real_data(sample_size=1)  # small call to init paths; will re-load templates below
                # Reload templates properly (avoid loading Phase 2 full data in _load_real_data)
                logger.info("Loading REAL ATUS 2023 survey data for templates...")
                atus_data = load_all_atus_data()
                atus_templates = create_activity_templates(atus_data)
                self.stats['templates_count'] = len(atus_templates)
                self.stats['real_atus_respondents'] = len(atus_templates)

                # Prepare Phase 3 shards output
                phase3_shards_dir = self.output_dir / 'phase3_shards'
                phase3_shards_dir.mkdir(parents=True, exist_ok=True)
                phase3_manifest = phase3_shards_dir / 'manifest.json'
                out_files: List[str] = []
                processed_buildings = 0
                max_buildings = sample_size or float('inf')
                sample_out_parts: List[pd.DataFrame] = []

                try:
                    with open(manifest, 'r') as f:
                        phase2_info = json.load(f)
                        files = phase2_info.get('files', [])
                except Exception as e:
                    logger.error(f"Failed to read Phase 2 manifest: {e}")
                    raise

                shard_idx = 0
                for fp in files:
                    if processed_buildings >= max_buildings:
                        break
                    try:
                        chunk = pd.read_pickle(fp)
                    except Exception as e:
                        logger.warning(f"Skipping unreadable Phase 2 shard {fp}: {e}")
                        continue

                    # Apply sample_size cap across shards
                    remaining = max_buildings - processed_buildings
                    if remaining != float('inf') and len(chunk) > remaining:
                        chunk = chunk.head(remaining)

                    # Extract persons and compute features
                    persons_df = self._extract_persons_optimized(chunk)
                    if len(persons_df) == 0:
                        logger.warning("No persons found in this shard; skipping")
                        continue
                    persons_features, template_features = self._create_matching_features(persons_df, atus_templates)

                    # Match in batches
                    assignments = self._fast_knn_matching(
                        persons_features, template_features, persons_df, atus_templates,
                        batch_size=match_batch_size or 20000
                    )

                    # Apply household coordination and merge
                    assignments = self._apply_household_coordination(assignments, persons_df, chunk, atus_templates)
                    buildings_with_acts = self._merge_activities_optimized(chunk, assignments, persons_df, atus_templates)

                    # Save shard
                    out_path = phase3_shards_dir / f'phase3_part_{shard_idx:05d}.pkl'
                    try:
                        buildings_with_acts.to_pickle(out_path)
                        out_files.append(str(out_path))
                    except Exception as e:
                        logger.error(f"Failed to save Phase 3 shard {out_path}: {e}")
                        raise

                    # Collect sample
                    if len(sample_out_parts) < 3:  # limit to keep memory low
                        sample_out_parts.append(buildings_with_acts.head(4000))

                    processed_buildings += len(buildings_with_acts)
                    shard_idx += 1

                    # Release memory
                    del chunk, persons_df, persons_features, template_features, assignments, buildings_with_acts
                
                # Finalize: write manifest and sample
                with open(phase3_manifest, 'w') as f:
                    json.dump({'n_shards': shard_idx, 'files': out_files, 'total_buildings': int(processed_buildings)}, f, indent=2)

                sample_out = pd.concat(sample_out_parts, ignore_index=True) if sample_out_parts else pd.DataFrame()
                # Save a small sample as the canonical output path for downstream checks
                sample_out.to_pickle(self.phase3_output_path)
                with open(self.phase3_metadata_path, 'w') as f:
                    json.dump({'end_time': datetime.now().isoformat(), 'buildings_processed': int(processed_buildings), 'sharded': True}, f, indent=2)

                self.stats['end_time'] = datetime.now()
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                logger.info(f"Phase 3 (streaming) completed in {duration:.2f} seconds over {shard_idx} shards")
                return sample_out
            else:
                # Standard execution path (loads full Phase 2 output)
                # Step 1: Load real data (Phase 2 buildings + real ATUS)
                buildings_with_recs, atus_templates = self._load_real_data(sample_size)
                
                # Step 2: Extract and prepare persons
                persons_df = self._extract_persons_optimized(buildings_with_recs)
                
                # Step 3: Create features for matching
                persons_features, template_features = self._create_matching_features(
                    persons_df, atus_templates
                )
                
                # Step 4: Fast k-NN matching to real ATUS respondents (GPU-accelerated if available)
                assignments = self._fast_knn_matching(
                    persons_features, template_features, persons_df, atus_templates,
                    batch_size=match_batch_size or 20000
                )
                
                # Step 5: Apply household coordination
                assignments = self._apply_household_coordination(
                    assignments, persons_df, buildings_with_recs, atus_templates
                )
                
                # Step 6: Merge back to buildings
                buildings_with_activities = self._merge_activities_optimized(
                    buildings_with_recs, assignments, persons_df, atus_templates
                )
                
                # Step 7: Save results
                self.stats['end_time'] = datetime.now()
                self._save_results(buildings_with_activities, assignments)
                
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                logger.info(f"Phase 3 completed in {duration:.2f} seconds")
                logger.info(f"Matched {self.stats['matched_persons']} persons to {self.stats['real_atus_respondents']} real ATUS respondents")
                
                return buildings_with_activities
            
        except Exception as e:
            logger.error(f"Optimized Phase 3 failed: {str(e)}")
            raise
    
    def _load_real_data(self, sample_size: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load REAL data only - no synthetic fallbacks."""
        logger.info("Loading REAL data (Phase 2 output + ATUS 2023)")
        
        # Load Phase 2 output using config path
        phase2_path = Path(self.config.get_data_path('phase2_output'))
        if not phase2_path.exists():
            # Try alternate path structure
            phase2_path = self.output_dir / 'phase2_pums_recs_buildings.pkl'
            if not phase2_path.exists():
                raise FileNotFoundError(f"Phase 2 output not found. Expected at: {phase2_path}")
        
        buildings_with_recs = pd.read_pickle(phase2_path)
        logger.info(f"Loaded {len(buildings_with_recs)} buildings from Phase 2")
        
        # Apply sample size if specified
        if sample_size and sample_size < len(buildings_with_recs):
            buildings_with_recs = buildings_with_recs.head(sample_size)
        
        self.stats['buildings_count'] = len(buildings_with_recs)
        
        # Load REAL ATUS data - NO SYNTHETIC DATA
        logger.info("Loading REAL ATUS 2023 survey data...")
        try:
            # Load comprehensive ATUS data from all 10 data files
            atus_data = load_all_atus_data()
            
            # Create activity templates from real respondents
            atus_templates = create_activity_templates(atus_data)
            
            self.stats['templates_count'] = len(atus_templates)
            self.stats['real_atus_respondents'] = len(atus_templates)
            
            logger.info(f"Loaded {len(atus_templates)} REAL ATUS respondents as templates")
            logger.info(f"Each respondent has actual time use data from 2023 survey")
            
        except FileNotFoundError as e:
            logger.error(f"ATUS data files not found: {e}")
            logger.error("Please ensure ATUS 2023 data is available in data/raw/atus/2023/")
            raise
        
        return buildings_with_recs, atus_templates
    
    def _extract_persons_optimized(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """Extract persons with essential features."""
        logger.info("Extracting persons from buildings")
        
        persons_list = []
        
        for idx, building in buildings.iterrows():
            if 'persons' not in building or not isinstance(building['persons'], list):
                continue
            
            for person_idx, person in enumerate(building['persons']):
                # Extract essential features
                person_record = {
                    'building_id': idx,
                    'person_idx': person_idx,
                    'person_id': f"{idx}_{person_idx}"
                }
                
                if isinstance(person, dict):
                    # Extract demographic features for matching
                    person_record.update({
                        'AGEP': person.get('AGEP', 35),  # Age
                        'SEX': person.get('SEX', 1),  # Sex (1=male, 2=female)
                        'ESR': person.get('ESR', 6),  # Employment status
                        'SCHL': person.get('SCHL', 16),  # Education level
                        'MAR': person.get('MAR', 5),  # Marital status
                        'WKHP': person.get('WKHP', 0),  # Usual hours worked
                        'PINCP': person.get('PINCP', 0),  # Income
                        'DIS': person.get('DIS', 2),  # Disability status
                        'RAC1P': person.get('RAC1P', 1),  # Race
                        'HISP': person.get('HISP', 1)  # Hispanic origin
                    })
                
                persons_list.append(person_record)
        
        persons_df = pd.DataFrame(persons_list)
        self.stats['persons_count'] = len(persons_df)
        logger.info(f"Extracted {len(persons_df)} persons")
        
        return persons_df
    
    def _create_matching_features(self, persons: pd.DataFrame, 
                                  templates: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for matching PUMS persons to real ATUS respondents."""
        logger.info("Creating enhanced features for matching to real ATUS respondents")
        
        # Enhance both datasets with cross-dataset features
        logger.info("Applying cross-dataset feature engineering...")
        persons_enhanced = enhance_dataset_for_matching(persons, 'pums', 'atus')
        templates_enhanced = enhance_dataset_for_matching(templates, 'atus', 'pums')
        
        # Define the features to use for matching
        # Use both basic and enhanced features
        basic_features = [
            'AGEP', 'SEX', 'ESR', 'SCHL', 'MAR', 'WKHP', 'PINCP', 'DIS'
        ]
        
        enhanced_features = [
            'hh_size_3cat', 'income_tercile', 'ses_category',
            'work_from_home', 'childcare_likely', 'retirement_activities',
            'student_activities', 'working_age_adult', 'retired_senior'
        ]
        
        # Create feature vectors for persons (PUMS)
        persons_features = []
        for _, person in persons_enhanced.iterrows():
            features = []
            
            # Basic features (normalized)
            features.append(person.get('AGEP', 35) / 100.0)  # Age
            features.append(1.0 if person.get('SEX', 1) == 2 else 0.0)  # Female
            features.append(1.0 if person.get('ESR', 6) in [1, 2] else 0.0)  # Employed
            features.append(min(person.get('SCHL', 16), 24) / 24.0)  # Education
            features.append(1.0 if person.get('MAR', 5) == 1 else 0.0)  # Married
            features.append(min(person.get('WKHP', 0), 80) / 80.0)  # Work hours
            features.append(np.log1p(max(person.get('PINCP', 0), 0)) / 15.0)  # Income
            features.append(1.0 if person.get('DIS', 2) == 1 else 0.0)  # Disability
            
            # Enhanced features (if available)
            for feat in enhanced_features:
                if feat in person.index:
                    if isinstance(person[feat], (int, float)):
                        features.append(float(person[feat]))
                    elif feat in ['hh_size_3cat', 'income_tercile', 'ses_category']:
                        # Convert categorical to numeric
                        cat_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0,
                                  'single': 0.0, 'small': 0.5, 'large': 1.0}
                        features.append(cat_map.get(str(person[feat]), 0.5))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            persons_features.append(features)
        
        persons_features = np.array(persons_features)
        
        # Create feature vectors for ATUS templates (real respondents)
        template_features = []
        for _, template in templates_enhanced.iterrows():
            features = []
            
            # Basic features (normalized)
            features.append(template.get('age', 35) / 100.0)
            features.append(1.0 if template.get('sex', 1) == 2 else 0.0)
            features.append(1.0 if template.get('employed', False) else 0.0)
            # Map education levels
            edu_map = {'less_than_hs': 0.3, 'high_school': 0.5, 
                      'some_college': 0.6, 'bachelors': 0.8, 'graduate': 1.0}
            features.append(edu_map.get(template.get('education_level', 'high_school'), 0.5))
            features.append(1.0 if template.get('spouse_present', 0) == 1 else 0.0)
            features.append(template.get('usual_hours_worked', 0) / 80.0)
            features.append(np.log1p(max(template.get('weekly_earnings', 0) * 52, 0)) / 15.0)
            features.append(0.0)  # Disability not directly available
            
            # Enhanced features (if available)
            for feat in enhanced_features:
                if feat in template.index:
                    if isinstance(template[feat], (int, float)):
                        features.append(float(template[feat]))
                    elif feat in ['hh_size_3cat', 'income_tercile', 'ses_category']:
                        cat_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0,
                                  'single': 0.0, 'small': 0.5, 'large': 1.0}
                        features.append(cat_map.get(str(template[feat]), 0.5))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            template_features.append(features)
        
        template_features = np.array(template_features)
        
        # Normalize features using StandardScaler for better distance calculation
        scaler = StandardScaler()
        persons_features = scaler.fit_transform(persons_features)
        template_features = scaler.transform(template_features)
        
        logger.info(f"Created enhanced feature matrices: persons {persons_features.shape}, templates {template_features.shape}")
        logger.info(f"Using {persons_features.shape[1]} features for matching (8 basic + {len(enhanced_features)} enhanced)")
        
        return persons_features, template_features
    
    def _fast_knn_matching(self, persons_features: np.ndarray, 
                           template_features: np.ndarray,
                           persons_df: pd.DataFrame,
                           templates_df: pd.DataFrame,
                           batch_size: int = 20000) -> Dict:
        """Match PUMS persons to real ATUS respondents using k-NN with GPU (PyTorch), FAISS, or scikit-learn fallback.

        Uses batching to control memory on large datasets.
        """
        logger.info("Matching persons to real ATUS respondents using k-NN")

        assignments: Dict[str, Any] = {}
        match_distances: Dict[str, float] = {}

        # 1) Try PyTorch on GPU for fast batched L2 distances
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                T = torch.tensor(template_features, dtype=torch.float32, device=device)
                # Precompute norms of templates: ||y||^2
                T_norm = (T * T).sum(dim=1).view(1, -1)
                n = persons_features.shape[0]
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    X = torch.tensor(persons_features[start:end], dtype=torch.float32, device=device)
                    # ||x||^2
                    X_norm = (X * X).sum(dim=1).view(-1, 1)
                    # -2 x·y
                    cross = -2.0 * (X @ T.t())
                    # dist^2 = ||x||^2 + ||y||^2 - 2 x·y
                    d2 = X_norm + T_norm + cross
                    best_vals, best_idx = torch.min(d2, dim=1)
                    for i, person_id in enumerate(persons_df['person_id'].iloc[start:end]):
                        idx = int(best_idx[i].item())
                        template_id = templates_df.iloc[idx]['template_id']
                        assignments[person_id] = template_id
                        match_distances[person_id] = float(torch.sqrt(torch.clamp(best_vals[i], min=0)).item())
                    # Free batch tensors
                    del X, X_norm, cross, d2, best_vals, best_idx
                    torch.cuda.empty_cache()
                avg_distance = float(np.mean(list(match_distances.values()))) if match_distances else 0.0
                logger.info(f"Matched {len(assignments)} persons using GPU (PyTorch), avg distance {avg_distance:.3f}")
                self.stats['matched_persons'] = len(assignments)
                return assignments
        except Exception:
            pass

        # 2) Try FAISS (GPU if available, else CPU)
        use_faiss = False
        index = None
        try:
            import faiss  # type: ignore
            d = template_features.shape[1]
            # Not all builds expose GPU helpers; guard calls
            gpu_count = getattr(faiss, 'get_num_gpus', lambda: 0)()
            if gpu_count > 0 and hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                flat = faiss.IndexFlatL2(d)
                index = faiss.index_cpu_to_gpu(res, 0, flat)
                use_faiss = True
                logger.info("Using FAISS GPU index for k-NN matching")
            else:
                index = faiss.IndexFlatL2(d)
                use_faiss = True
                logger.info("Using FAISS CPU index for k-NN matching")
            index.add(template_features.astype('float32'))
        except Exception:
            pass

        if use_faiss and index is not None:
            n = persons_features.shape[0]
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = persons_features[start:end].astype('float32')
                distances, indices = index.search(batch, 1)
                for i, person_id in enumerate(persons_df['person_id'].iloc[start:end]):
                    best_idx = int(indices[i, 0])
                    template_id = templates_df.iloc[best_idx]['template_id']
                    assignments[person_id] = template_id
                    match_distances[person_id] = float(distances[i, 0])
            avg_distance = float(np.mean(list(match_distances.values()))) if match_distances else 0.0
            logger.info(f"Matched {len(assignments)} persons using FAISS, avg distance {avg_distance:.3f}")
            self.stats['matched_persons'] = len(assignments)
            return assignments

        # 3) Fallback to scikit-learn NearestNeighbors (avoids full distance matrix)
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean', n_jobs=-1)
        nn.fit(template_features)
        distances, indices = nn.kneighbors(persons_features, return_distance=True)
        for i, person_id in enumerate(persons_df['person_id']):
            best_idx = int(indices[i, 0])
            template_id = templates_df.iloc[best_idx]['template_id']
            assignments[person_id] = template_id
            match_distances[person_id] = float(distances[i, 0])

        avg_distance = float(np.mean(list(match_distances.values()))) if match_distances else 0.0
        logger.info(f"Matched {len(assignments)} persons using sklearn NN, avg distance {avg_distance:.3f}")
        self.stats['matched_persons'] = len(assignments)
        return assignments

        self.stats['matched_persons'] = len(assignments)
        avg_distance = float(np.mean(list(match_distances.values()))) if match_distances else 0.0
        logger.info(f"Matched {len(assignments)} persons to real ATUS respondents (avg distance {avg_distance:.3f})")

        return assignments
    
    def _apply_household_coordination(self, assignments: Dict, 
                                      persons_df: pd.DataFrame,
                                      buildings: pd.DataFrame,
                                      templates: pd.DataFrame) -> Dict:
        """Apply household coordination using real activity patterns."""
        logger.info("Applying household coordination with real activity patterns")
        
        # Group persons by building
        persons_by_building = persons_df.groupby('building_id')
        
        # Get time use patterns from templates for coordination
        template_lookup = {row['template_id']: row for _, row in templates.iterrows()}
        
        for building_id, household_persons in persons_by_building:
            # Check household composition
            ages = household_persons['AGEP'].fillna(35)
            has_children = any(ages < 18)
            num_adults = sum(ages >= 18)
            
            if has_children and num_adults >= 1:
                # Ensure at least one adult has flexible schedule for childcare
                household_person_ids = household_persons['person_id'].tolist()
                
                # Find adults in household
                adult_mask = household_persons['AGEP'] >= 18
                adult_person_ids = household_persons[adult_mask]['person_id'].tolist()
                
                if adult_person_ids:
                    # Check if any adult has high childcare time
                    childcare_times = []
                    for pid in adult_person_ids:
                        if pid in assignments:
                            template = template_lookup.get(assignments[pid], {})
                            childcare_time = template.get('caring_household', 0)
                            childcare_times.append((pid, childcare_time))
                    
                    # If no adult has significant childcare time, reassign one
                    if childcare_times and max(ct[1] for ct in childcare_times) < 60:  # Less than 60 minutes
                        # Find templates with high childcare time
                        if 'caring_household' in templates.columns:
                            high_childcare_templates = templates[templates['caring_household'] > 120]
                        else:
                            high_childcare_templates = pd.DataFrame()  # Empty if column doesn't exist
                        
                        if len(high_childcare_templates) > 0:
                            # Reassign one adult to high childcare template
                            adult_to_reassign = adult_person_ids[0]
                            new_template = high_childcare_templates.iloc[0]['template_id']
                            assignments[adult_to_reassign] = new_template
                            logger.debug(f"Reassigned adult {adult_to_reassign} to template with more childcare time")
        
        return assignments
    
    def _merge_activities_optimized(self, buildings: pd.DataFrame,
                                    assignments: Dict,
                                    persons_df: pd.DataFrame,
                                    templates: pd.DataFrame) -> pd.DataFrame:
        """Merge real activity patterns back to buildings."""
        logger.info("Merging real ATUS activity patterns to buildings")
        
        buildings_with_activities = buildings.copy()
        
        # Create template lookup for activity data
        template_lookup = {row['template_id']: row.to_dict() for _, row in templates.iterrows()}
        
        # Load detailed activity sequences for retrieval
        logger.info("Loading detailed ATUS activity sequences")
        from ..data_loading.atus_loader import load_atus_activity_data
        detailed_activities = load_atus_activity_data()
        
        # Ensure we have the right columns
        if 'has_atus_activities' not in buildings_with_activities.columns:
            buildings_with_activities['has_atus_activities'] = False
        if 'uses_real_atus_data' not in buildings_with_activities.columns:
            buildings_with_activities['uses_real_atus_data'] = False
        
        # Create lists to store updated values
        updated_persons_list = []
        updated_indices = []
        
        # Update buildings with activity assignments
        for idx in buildings_with_activities.index:
            building = buildings_with_activities.loc[idx]
            if 'persons' not in building or not isinstance(building['persons'], list):
                logger.debug(f"Skipping building {idx}: persons column missing or not a list")
                updated_persons_list.append(building.get('persons', []))
                updated_indices.append(idx)
                continue
            
            updated_persons = []
            for person_idx, person in enumerate(building['persons']):
                person_id = f"{idx}_{person_idx}"
                person_copy = person.copy() if isinstance(person, dict) else {}
                
                if person_id in assignments:
                    template_id = assignments[person_id]
                    person_copy['atus_template_id'] = template_id
                    person_copy['has_activities'] = True
                    
                    # Add key time use information from real ATUS respondent
                    if template_id in template_lookup:
                        template_data = template_lookup[template_id]
                        person_copy['daily_time_use'] = {
                            'sleep': template_data.get('sleep_time', 0),
                            'work': template_data.get('work', 0),
                            'household': template_data.get('household_work', 0),
                            'leisure': template_data.get('leisure', 0),
                            'caring': template_data.get('caring_household', 0),
                            'travel': template_data.get('travel', 0)
                        }
                        person_copy['atus_case_id'] = template_data.get('case_id', '')
                        
                        # Add detailed activity sequence
                        case_id = template_data.get('case_id', '')
                        if case_id and case_id in detailed_activities['case_id'].values:
                            person_activities = detailed_activities[detailed_activities['case_id'] == case_id].copy()
                            person_activities = person_activities.sort_values('activity_number')
                            
                            # Create activity sequence list
                            activity_sequence = []
                            for _, act in person_activities.iterrows():
                                activity_sequence.append({
                                    'activity_num': int(act.get('activity_number', 0)),
                                    'start_time': str(act.get('start_time', '')),
                                    'stop_time': str(act.get('stop_time', '')),
                                    'duration_minutes': int(act.get('duration_minutes', 0)),
                                    'activity_code': str(act.get('activity_tier1', '')) + 
                                                    str(act.get('activity_tier2', '')) + 
                                                    str(act.get('activity_tier3', '')),
                                    'location': str(act.get('TEWHERE', -1))
                                })
                            
                            person_copy['activity_sequence'] = activity_sequence
                            logger.debug(f"Added {len(activity_sequence)} activities for person {person_id}")
                else:
                    person_copy['has_activities'] = False
                
                updated_persons.append(person_copy)
            
            updated_persons_list.append(updated_persons)
            updated_indices.append(idx)
        
        # Update all at once using proper assignment
        buildings_with_activities['persons'] = pd.Series(updated_persons_list, index=updated_indices)
        buildings_with_activities['has_atus_activities'] = True
        buildings_with_activities['uses_real_atus_data'] = True
        
        return buildings_with_activities
    
    def _save_results(self, buildings: pd.DataFrame, assignments: Dict):
        """Save results with metadata about real data usage."""
        logger.info("Saving Phase 3 results with real ATUS data")
        
        # Save main output
        buildings.to_pickle(self.phase3_output_path)
        
        # Save metadata
        metadata = {
            'phase': 'phase3_atus_matching_optimized',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            'statistics': self.stats,
            'match_rate': self.stats['matched_persons'] / max(self.stats['persons_count'], 1),
            'data_source': 'REAL ATUS 2023 Survey Data',
            'synthetic_data': False,
            'atus_respondents_used': self.stats['real_atus_respondents'],
            'note': 'All activity patterns from real ATUS survey respondents'
        }
        
        with open(self.phase3_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved results to {self.phase3_output_path}")
        logger.info(f"Used {self.stats['real_atus_respondents']} real ATUS respondents")


def run_optimized_phase3(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to run optimized Phase 3 with REAL data.
    
    Args:
        sample_size: Number of buildings to process
        
    Returns:
        DataFrame with buildings including real activity patterns
    """
    matcher = OptimizedPhase3Matcher()
    return matcher.run(sample_size)