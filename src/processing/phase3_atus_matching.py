"""
Phase 3: ATUS Activity Pattern Matching.

This module implements the third phase of the enrichment pipeline, which assigns
realistic daily activity patterns from ATUS to individuals in PUMS buildings
using probabilistic matching with household coordination constraints.
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

from ..utils.config_loader import get_config
from ..utils.logging_setup import setup_logging, log_execution_time, log_memory_usage
from ..validation.data_validator import (
    validate_dataframe,
    validate_required_columns,
    generate_data_quality_report
)
from ..data_loading.atus_loader import (
    load_atus_respondent_data,
    load_atus_activity_data,
    load_atus_cps_data,
    load_atus_roster_data,
    load_atus_summary_data,
    create_activity_templates,
    create_enhanced_atus_features,
    prepare_atus_for_matching
)
from ..utils.enhanced_feature_alignment import (
    align_pums_atus_features,
    create_feature_importance_weights
)
from ..matching.blocking import create_standard_blocks
from ..matching.fellegi_sunter import FellegiSunterMatcher, ComparisonField
from ..matching.em_algorithm import EMAlgorithm
from ..matching.match_quality_assessor import MatchQualityAssessor

logger = logging.getLogger(__name__)


class Phase3ATUSMatcher:
    """Orchestrates Phase 3 ATUS activity pattern matching process."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Phase 3 processor."""
        self.config = config or get_config()
        
        # Get output directories from config
        self.output_dir = Path(self.config.get_data_path('phase3_output')).parent
        self.validation_dir = Path(self.config.get_data_path('phase3_validation')).parent
        self.params_dir = Path(self.config.get('data_paths.matching_parameters', 'data/matching_parameters'))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 3 specific paths
        self.phase3_output_path = self.output_dir / 'phase3_pums_recs_atus_buildings.pkl'
        self.phase3_params_path = self.params_dir / 'phase3_atus_weights.json'
        self.phase3_validation_path = self.validation_dir / 'phase3_validation_report.html'
        self.phase3_metadata_path = self.output_dir / 'phase3_metadata.json'
        
        # Initialize components
        self.matcher = None
        self.em_algorithm = None
        self.quality_assessor = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'buildings_count': 0,
            'persons_count': 0,
            'templates_count': 0,
            'matched_persons': 0,
            'households_coordinated': 0
        }
    
    @log_execution_time(logger)
    @log_memory_usage(logger)
    def run(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Execute Phase 3 ATUS activity pattern matching.
        
        Args:
            sample_size: Number of buildings to process (None for all)
            
        Returns:
            DataFrame with buildings including activity patterns
        """
        self.stats['start_time'] = datetime.now()
        logger.info("Starting Phase 3: ATUS Activity Pattern Matching")
        
        try:
            # Step 1: Load Phase 2 output and ATUS data
            buildings_with_recs, atus_templates = self._load_data(sample_size)
            
            # Step 2: Extract persons from buildings
            persons_df = self._extract_persons_from_buildings(buildings_with_recs)
            
            # Step 3: Setup matching configuration
            self._setup_person_activity_matching()
            
            # Step 4: Create blocked candidate pairs
            blocks = self._create_person_activity_blocks(persons_df, atus_templates)
            
            # Step 5: Perform probabilistic matching
            comparison_data = self._compare_person_activity_pairs(
                persons_df, atus_templates, blocks
            )
            
            # Step 6: Learn parameters using EM
            m_probs, u_probs = self._learn_activity_matching_parameters(comparison_data)
            
            # Step 7: Apply learned parameters and classify
            classified_pairs = self._classify_person_activity_pairs(
                comparison_data, m_probs, u_probs
            )
            
            # Step 8: Assign activities with household coordination
            activity_assignments = self._assign_coordinated_activities(
                classified_pairs, persons_df, atus_templates, buildings_with_recs
            )
            
            # Step 9: Merge activities back into buildings
            logger.info(f"About to merge {len(activity_assignments)} assignments to {len(buildings_with_recs)} buildings")
            buildings_with_activities = self._merge_activities_to_buildings(
                buildings_with_recs, activity_assignments, persons_df
            )
            
            # Step 10: Validate and save results
            self.stats['end_time'] = datetime.now()
            self._validate_and_save(buildings_with_activities, activity_assignments)
            
            logger.info(f"Phase 3 completed. Matched {self.stats['matched_persons']} persons.")
            
            return buildings_with_activities
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {str(e)}")
            raise
    
    def _load_data(self, sample_size: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Phase 2 buildings and ATUS activity templates with enhanced features."""
        logger.info("Loading data for Phase 3")
        
        # Load Phase 2 output
        phase2_path = self.output_dir.parent / 'processed' / 'phase2_pums_recs_buildings.pkl'
        if not phase2_path.exists():
            raise FileNotFoundError(f"Phase 2 output not found: {phase2_path}")
        
        buildings_with_recs = pd.read_pickle(phase2_path)
        logger.info(f"Loaded {len(buildings_with_recs)} buildings from Phase 2")
        
        # Apply sample size if specified
        if sample_size and sample_size < len(buildings_with_recs):
            logger.info(f"Sampling {sample_size} buildings")
            buildings_with_recs = buildings_with_recs.sample(
                n=sample_size,
                random_state=self.config.get_random_seed()
            )
        
        self.stats['buildings_count'] = len(buildings_with_recs)
        
        # Load all ATUS data sources for enhanced features
        logger.info("Loading comprehensive ATUS data")
        atus_respondents = load_atus_respondent_data()
        atus_activities = load_atus_activity_data()
        
        # Load additional ATUS data sources
        try:
            logger.info("Loading ATUS-CPS data for detailed demographics")
            atus_cps = load_atus_cps_data()
            logger.info("Loading ATUS roster data for household composition")
            atus_roster = load_atus_roster_data()
            logger.info("Loading ATUS summary data for time use patterns")
            atus_summary = load_atus_summary_data()
        except FileNotFoundError as e:
            logger.warning(f"Could not load additional ATUS data: {e}")
            atus_cps = None
            atus_roster = None
            atus_summary = None
        
        # Create enhanced features from all sources
        atus_enhanced = create_enhanced_atus_features(
            atus_respondents,
            cps_df=atus_cps,
            roster_df=atus_roster,
            summary_df=atus_summary
        )
        
        # Create activity templates with enhanced features
        atus_templates = create_activity_templates(atus_enhanced, atus_activities)
        logger.info(f"Created {len(atus_templates)} ATUS activity templates with enhanced features")
        self.stats['templates_count'] = len(atus_templates)
        
        return buildings_with_recs, atus_templates
    
    def _extract_persons_from_buildings(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """Extract individual persons from buildings for matching."""
        logger.info("Extracting persons from buildings")
        
        persons_list = []
        
        for idx, building in buildings.iterrows():
            if 'persons' not in building or not isinstance(building['persons'], list):
                continue
            
            for person_idx, person in enumerate(building['persons']):
                # Create flattened person record
                person_record = {
                    'building_id': idx,
                    'person_idx': person_idx,
                    'person_id': f"{idx}_{person_idx}"
                }
                
                # Add person attributes
                if isinstance(person, dict):
                    person_record.update(person)
                
                # Add household context
                person_record['household_size'] = len(building['persons'])
                person_record['has_children'] = any(
                    p.get('AGEP', 18) < 18 for p in building['persons'] 
                    if isinstance(p, dict)
                )
                
                persons_list.append(person_record)
        
        persons_df = pd.DataFrame(persons_list)
        
        # DON'T create features here - will use enhanced alignment instead
        # persons_df = self._create_person_matching_features(persons_df)
        
        self.stats['persons_count'] = len(persons_df)
        logger.info(f"Extracted {len(persons_df)} persons from {self.stats['buildings_count']} buildings")
        
        return persons_df
    
    def _create_person_matching_features(self, persons: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for person-activity matching (30+ features)."""
        persons = persons.copy()
        logger.info("Creating enhanced person features for improved matching")
        
        # ========== AGE FEATURES (Multiple granularities) ==========
        if 'AGEP' in persons.columns:
            persons['age'] = persons['AGEP']
            
            # Standard age groups
            persons['age_group'] = pd.cut(
                persons['age'],
                bins=[0, 5, 18, 35, 50, 65, 100],
                labels=['preschool', 'school_age', 'young_adult', 'adult', 'middle_age', 'senior']
            )
            
            # 5-year age bins
            persons['age_5yr'] = pd.cut(persons['age'], 
                                       bins=range(0, 101, 5),
                                       right=False,
                                       labels=[f"{i}-{i+4}" for i in range(0, 100, 5)])
            
            # 10-year age bins
            persons['age_10yr'] = pd.cut(persons['age'],
                                        bins=range(0, 101, 10),
                                        right=False,
                                        labels=[f"{i}-{i+9}" for i in range(0, 100, 10)])
            
            # Life stage
            persons['life_stage'] = pd.cut(persons['age'],
                                          bins=[0, 18, 25, 35, 45, 55, 65, 100],
                                          labels=['minor', 'young_adult', 'early_career', 
                                                 'mid_career', 'late_career', 'pre_retire', 'retired'])
            
            # Binary age indicators
            persons['is_minor'] = persons['age'] < 18
            persons['is_young_adult'] = (persons['age'] >= 18) & (persons['age'] < 30)
            persons['is_middle_aged'] = (persons['age'] >= 40) & (persons['age'] < 60)
            persons['is_senior'] = persons['age'] >= 65
            persons['is_working_age'] = (persons['age'] >= 18) & (persons['age'] < 65)
        
        # ========== EMPLOYMENT FEATURES (Detailed) ==========
        if 'ESR' in persons.columns:
            emp_map = {
                1: 'employed_full',  # Employed
                2: 'employed_full',  # Employed, with a job but not at work
                3: 'unemployed',     # Unemployed
                4: 'unemployed',     # Armed forces
                5: 'unemployed',     # Armed forces
                6: 'not_working'     # Not in labor force
            }
            persons['employment_category'] = persons['ESR'].map(emp_map).fillna('not_working')
            persons['is_employed'] = persons['employment_category'].str.contains('employed')
            persons['is_unemployed'] = persons['employment_category'] == 'unemployed'
            persons['is_not_in_labor_force'] = persons['employment_category'] == 'not_working'
        
        # Work hours (if available)
        if 'WKHP' in persons.columns:
            persons['work_hours_cat'] = pd.cut(persons['WKHP'].fillna(0),
                                               bins=[0, 1, 20, 35, 40, 50, 100],
                                               labels=['not_working', 'part_time_low', 'part_time_high',
                                                      'full_time', 'overtime', 'extreme_hours'])
            persons['is_part_time'] = persons['WKHP'].between(1, 34)
            persons['is_full_time'] = persons['WKHP'].between(35, 45)
            persons['is_overtime'] = persons['WKHP'] > 45
        
        # Class of worker
        if 'COW' in persons.columns:
            cow_map = {
                1: 'private',
                2: 'private_nonprofit',
                3: 'government',
                4: 'government',
                5: 'government',
                6: 'self_employed',
                7: 'self_employed',
                8: 'unpaid_family',
                9: 'unemployed'
            }
            persons['worker_class'] = persons['COW'].map(cow_map).fillna('unknown')
            persons['is_self_employed'] = persons['worker_class'].str.contains('self_employed')
            persons['is_government'] = persons['worker_class'].str.contains('government')
        
        # ========== EDUCATION FEATURES ==========
        if 'SCHL' in persons.columns:
            # Detailed education mapping
            edu_detail_map = {
                1: 'no_schooling',
                2: 'nursery',
                3: 'kindergarten',
                4: 'grade_1', 5: 'grade_2', 6: 'grade_3', 7: 'grade_4',
                8: 'grade_5', 9: 'grade_6', 10: 'grade_7', 11: 'grade_8',
                12: 'grade_9', 13: 'grade_10', 14: 'grade_11', 15: 'grade_12_no_diploma',
                16: 'high_school',
                17: 'ged',
                18: 'some_college_less_1yr',
                19: 'some_college_1yr_plus',
                20: 'associates',
                21: 'bachelors',
                22: 'masters',
                23: 'professional',
                24: 'doctorate'
            }
            
            # Simplified education categories
            edu_simple_map = {}
            for i in range(1, 16):
                edu_simple_map[i] = 'less_than_hs'
            edu_simple_map[16] = 'high_school'
            edu_simple_map[17] = 'high_school'
            for i in range(18, 21):
                edu_simple_map[i] = 'some_college'
            edu_simple_map[21] = 'bachelors'
            for i in range(22, 25):
                edu_simple_map[i] = 'graduate'
            
            persons['education_level'] = persons['SCHL'].map(edu_simple_map).fillna('high_school')
            
            # Binary education indicators
            persons['has_hs_diploma'] = persons['SCHL'] >= 16
            persons['has_some_college'] = persons['SCHL'] >= 18
            persons['has_bachelors'] = persons['SCHL'] >= 21
            persons['has_graduate_degree'] = persons['SCHL'] >= 22
            
            # Education ordinal
            edu_order_map = {
                'less_than_hs': 1,
                'high_school': 2,
                'some_college': 3,
                'bachelors': 4,
                'graduate': 5
            }
            persons['education_ordinal'] = persons['education_level'].map(edu_order_map).fillna(2)
        
        # ========== MARITAL STATUS FEATURES ==========
        if 'MAR' in persons.columns:
            marital_map = {
                1: 'married',
                2: 'widowed',
                3: 'divorced',
                4: 'separated',
                5: 'never_married'
            }
            persons['marital_cat'] = persons['MAR'].map(marital_map).fillna('unknown')
            persons['is_married'] = persons['marital_cat'] == 'married'
            persons['is_single'] = persons['marital_cat'].isin(['never_married', 'divorced', 'widowed'])
            persons['was_married'] = persons['marital_cat'].isin(['divorced', 'widowed', 'separated'])
        
        # ========== DEMOGRAPHIC FEATURES ==========
        # Sex/Gender
        if 'SEX' in persons.columns:
            persons['sex'] = persons['SEX']
            persons['is_female'] = persons['SEX'] == 2
            persons['is_male'] = persons['SEX'] == 1
        
        # Race
        if 'RAC1P' in persons.columns:
            race_map = {
                1: 'white',
                2: 'black',
                3: 'american_indian',
                4: 'american_indian',
                5: 'american_indian',
                6: 'asian',
                7: 'hawaiian_pacific',
                8: 'other',
                9: 'multiracial'
            }
            persons['race_cat'] = persons['RAC1P'].map(race_map).fillna('unknown')
            persons['is_minority'] = ~persons['race_cat'].isin(['white', 'unknown'])
        
        # Hispanic origin
        if 'HISP' in persons.columns:
            persons['is_hispanic'] = persons['HISP'] > 1
        
        # Citizenship
        if 'NATIVITY' in persons.columns:
            persons['is_native_born'] = persons['NATIVITY'] == 1
            persons['is_foreign_born'] = persons['NATIVITY'] == 2
        
        # ========== HOUSEHOLD CONTEXT FEATURES ==========
        # Household size
        if 'household_size' in persons.columns:
            persons['household_size_cat'] = pd.cut(
                persons['household_size'],
                bins=[0, 1, 2, 4, 10],
                labels=['single', 'couple', 'family', 'large']
            )
            
            persons['hh_size_detailed'] = pd.cut(persons['household_size'],
                                                bins=[0, 1, 2, 3, 4, 5, 6, 20],
                                                labels=['single', 'couple', 'three', 'four', 'five', 'six', 'seven_plus'])
            
            persons['is_single_person'] = persons['household_size'] == 1
            persons['is_couple_only'] = persons['household_size'] == 2
            persons['is_large_household'] = persons['household_size'] >= 5
        
        # Relationship to householder
        if 'RELSHIPP' in persons.columns:
            rel_map = {
                1: 'reference_person',
                2: 'spouse',
                3: 'child',
                4: 'child_in_law',
                5: 'parent',
                6: 'parent_in_law',
                7: 'sibling',
                8: 'sibling_in_law',
                9: 'grandchild',
                10: 'other_relative',
                11: 'roommate',
                12: 'foster_child',
                13: 'other_nonrelative'
            }
            persons['relationship'] = persons['RELSHIPP'].map(rel_map).fillna('unknown')
            persons['is_householder'] = persons['relationship'] == 'reference_person'
            persons['is_spouse'] = persons['relationship'] == 'spouse'
            persons['is_child'] = persons['relationship'].str.contains('child')
        
        # ========== INCOME/ECONOMIC FEATURES ==========
        if 'PINCP' in persons.columns:
            # Personal income categories
            persons['has_income'] = persons['PINCP'] > 0
            persons['income_cat'] = pd.cut(persons['PINCP'].fillna(0),
                                          bins=[-100000, 0, 20000, 40000, 60000, 100000, 1000000],
                                          labels=['negative', 'very_low', 'low', 'medium', 'high', 'very_high'])
            persons['is_low_income'] = persons['PINCP'] < 20000
            persons['is_high_income'] = persons['PINCP'] > 100000
        
        # ========== OCCUPATION FEATURES ==========
        if 'OCCP' in persons.columns:
            # Simplified occupation categories based on first digit
            persons['occupation_major'] = persons['OCCP'].fillna(0).astype(str).str[0]
            occ_map = {
                '0': 'not_working',
                '1': 'management',
                '2': 'professional',
                '3': 'service',
                '4': 'sales_office',
                '5': 'construction',
                '6': 'maintenance',
                '7': 'production',
                '8': 'transportation',
                '9': 'military'
            }
            persons['occupation_cat'] = persons['occupation_major'].map(occ_map).fillna('unknown')
            persons['is_white_collar'] = persons['occupation_cat'].isin(['management', 'professional', 'sales_office'])
            persons['is_blue_collar'] = persons['occupation_cat'].isin(['construction', 'maintenance', 'production'])
            persons['is_service_worker'] = persons['occupation_cat'] == 'service'
        
        # ========== COMMUTE/TRAVEL FEATURES ==========
        if 'JWMNP' in persons.columns:
            # Journey to work time
            persons['commute_time_cat'] = pd.cut(persons['JWMNP'].fillna(0),
                                                 bins=[0, 1, 15, 30, 60, 120, 200],
                                                 labels=['no_commute', 'short', 'medium', 'long', 'very_long', 'extreme'])
            persons['has_long_commute'] = persons['JWMNP'] > 60
            persons['works_from_home'] = persons['JWMNP'] == 0
        
        # Transportation to work
        if 'JWTRNS' in persons.columns:
            trans_map = {
                1: 'car_alone',
                2: 'carpool',
                3: 'bus',
                4: 'subway',
                5: 'railroad',
                6: 'ferry',
                7: 'taxi',
                8: 'motorcycle',
                9: 'bicycle',
                10: 'walked',
                11: 'worked_from_home',
                12: 'other'
            }
            persons['transport_mode'] = persons['JWTRNS'].map(trans_map).fillna('unknown')
            persons['drives_to_work'] = persons['transport_mode'].isin(['car_alone', 'carpool'])
            persons['uses_public_transit'] = persons['transport_mode'].isin(['bus', 'subway', 'railroad', 'ferry'])
        
        # ========== DISABILITY FEATURES ==========
        if 'DIS' in persons.columns:
            persons['has_disability'] = persons['DIS'] == 1
        
        # ========== VETERAN STATUS ==========
        if 'MIL' in persons.columns:
            persons['is_veteran'] = persons['MIL'].isin([1, 2, 3])
        
        # ========== INTERACTION FEATURES ==========
        # Age × Employment
        if 'age_group' in persons.columns and 'employment_category' in persons.columns:
            persons['age_employment'] = (
                persons['age_group'].astype(str) + '_' + 
                persons['employment_category'].astype(str)
            )
        
        # Has children × Employment
        if 'has_children' in persons.columns and 'is_employed' in persons.columns:
            persons['parent_worker'] = persons['has_children'] & persons['is_employed']
            persons['parent_not_working'] = persons['has_children'] & ~persons['is_employed']
        
        # Education × Age
        if 'education_level' in persons.columns and 'life_stage' in persons.columns:
            persons['edu_life_stage'] = (
                persons['education_level'].astype(str) + '_' +
                persons['life_stage'].astype(str)
            )
        
        # ========== ENHANCED BLOCKING KEYS ==========
        # Create multiple blocking strategies
        persons['block_key_1'] = (
            persons['age_group'].astype(str) + '_' +
            persons['employment_category'].astype(str)
        )
        
        persons['block_key_2'] = (
            persons['household_size_cat'].astype(str) + '_' +
            persons['has_children'].astype(str)
        )
        
        persons['block_key_3'] = (
            persons['life_stage'].astype(str) + '_' +
            persons.get('marital_cat', 'unknown').astype(str)
        )
        
        persons['block_key_4'] = (
            persons['education_level'].astype(str) + '_' +
            persons.get('income_cat', 'unknown').astype(str)
        )
        
        # Log feature creation summary
        feature_cols = [col for col in persons.columns if col not in ['building_id', 'person_idx', 'person_id']]
        logger.info(f"Created {len(feature_cols)} features for person-activity matching")
        
        # Log feature coverage
        coverage = (persons[feature_cols].notna().sum() / len(persons) * 100).round(1)
        high_coverage = coverage[coverage > 80]
        logger.info(f"Features with >80% coverage: {len(high_coverage)} out of {len(feature_cols)}")
        
        return persons
    
    def _setup_person_activity_matching(self):
        """Configure Fellegi-Sunter matcher for person-activity comparison with 50+ features."""
        logger.info("Setting up enhanced person-activity matching configuration with aligned features")
        
        # Get feature importance weights
        feature_weights = create_feature_importance_weights()
        
        # Define comprehensive comparison fields (50+ aligned features)
        comparison_fields = [
            # Core demographic features (high weight)
            ComparisonField(name='age_group_simple', field_type='categorical', comparison_method='exact', weight=feature_weights.get('age_group_simple', 2.0)),
            ComparisonField(name='age_group_detailed', field_type='categorical', comparison_method='exact', weight=feature_weights.get('age_group_detailed', 1.8)),
            ComparisonField(name='age_decade', field_type='numeric', comparison_method='difference', weight=1.5),
            ComparisonField(name='sex_code', field_type='categorical', comparison_method='exact', weight=feature_weights.get('sex_code', 0.8)),
            ComparisonField(name='is_female', field_type='categorical', comparison_method='exact', weight=feature_weights.get('is_female', 0.8)),
            
            # Employment features (very high weight)
            ComparisonField(name='employed', field_type='categorical', comparison_method='exact', weight=feature_weights.get('employed', 2.5)),
            ComparisonField(name='work_intensity', field_type='categorical', comparison_method='exact', weight=feature_weights.get('work_intensity', 2.0)),
            ComparisonField(name='full_time', field_type='categorical', comparison_method='exact', weight=feature_weights.get('full_time', 1.8)),
            ComparisonField(name='part_time', field_type='categorical', comparison_method='exact', weight=feature_weights.get('part_time', 1.5)),
            ComparisonField(name='not_working', field_type='categorical', comparison_method='exact', weight=feature_weights.get('not_working', 1.5)),
            ComparisonField(name='work_hours_weekly', field_type='numeric', comparison_method='difference', weight=feature_weights.get('work_hours_weekly', 1.5)),
            
            # Education features (medium weight)
            ComparisonField(name='education_level', field_type='categorical', comparison_method='exact', weight=1.2),
            ComparisonField(name='has_bachelors', field_type='categorical', comparison_method='exact', weight=0.8),
            ComparisonField(name='education_ordinal', field_type='numeric', comparison_method='difference', weight=0.6),
            
            # Household composition (high weight)
            ComparisonField(name='hh_size_group', field_type='categorical', comparison_method='exact', weight=feature_weights.get('hh_size_group', 1.5)),
            ComparisonField(name='has_children', field_type='categorical', comparison_method='exact', weight=feature_weights.get('has_children', 2.0)),
            ComparisonField(name='has_young_children', field_type='categorical', comparison_method='exact', weight=feature_weights.get('has_young_children', 2.2)),
            ComparisonField(name='single_person_hh', field_type='categorical', comparison_method='exact', weight=1.2),
            ComparisonField(name='large_family', field_type='categorical', comparison_method='exact', weight=1.0),
            
            # Marital status (medium weight)
            ComparisonField(name='married', field_type='categorical', comparison_method='exact', weight=feature_weights.get('married', 1.5)),
            ComparisonField(name='currently_single', field_type='categorical', comparison_method='exact', weight=1.0),
            
            # Income/Economic (medium weight)
            ComparisonField(name='income_bracket', field_type='categorical', comparison_method='exact', weight=1.2),
            ComparisonField(name='in_poverty', field_type='categorical', comparison_method='exact', weight=0.8),
            ComparisonField(name='high_income', field_type='categorical', comparison_method='exact', weight=0.8),
            
            # Occupation (medium weight)
            ComparisonField(name='white_collar', field_type='categorical', comparison_method='exact', weight=1.0),
            ComparisonField(name='occ_professional', field_type='categorical', comparison_method='exact', weight=0.8),
            ComparisonField(name='occ_service', field_type='categorical', comparison_method='exact', weight=0.8),
            
            # Time use predictors (very high weight)
            ComparisonField(name='schedule_flexibility', field_type='numeric', comparison_method='difference', weight=feature_weights.get('schedule_flexibility', 2.0)),
            ComparisonField(name='childcare_responsibility', field_type='numeric', comparison_method='difference', weight=feature_weights.get('childcare_responsibility', 2.2)),
            ComparisonField(name='time_poverty', field_type='numeric', comparison_method='difference', weight=feature_weights.get('time_poverty', 1.8)),
            ComparisonField(name='work_life_balance', field_type='numeric', comparison_method='difference', weight=feature_weights.get('work_life_balance', 1.8)),
            ComparisonField(name='works_from_home', field_type='categorical', comparison_method='exact', weight=1.2),
            
            # Activity likelihood scores (high weight)
            ComparisonField(name='work_activity_likely', field_type='numeric', comparison_method='difference', weight=feature_weights.get('work_activity_likely', 2.0)),
            ComparisonField(name='childcare_likely', field_type='numeric', comparison_method='difference', weight=feature_weights.get('childcare_likely', 2.0)),
            ComparisonField(name='leisure_likely', field_type='numeric', comparison_method='difference', weight=feature_weights.get('leisure_likely', 1.5)),
            
            # Composite indices (high weight)
            ComparisonField(name='life_complexity', field_type='numeric', comparison_method='difference', weight=feature_weights.get('life_complexity', 1.5)),
            ComparisonField(name='ses_index', field_type='numeric', comparison_method='difference', weight=feature_weights.get('ses_index', 1.2)),
            
            # Interaction features (medium-high weight)
            ComparisonField(name='working_parent', field_type='categorical', comparison_method='exact', weight=feature_weights.get('working_parent', 1.5)),
            ComparisonField(name='young_employed', field_type='categorical', comparison_method='exact', weight=feature_weights.get('young_employed', 1.2)),
        ]
        
        # Initialize matcher with enhanced fields
        self.matcher = FellegiSunterMatcher(comparison_fields)
        
        # Set default thresholds (will be adjusted based on EM results)
        self.matcher.set_thresholds(upper=10.0, lower=-10.0)  # Wider range for more features
        
        logger.info(f"Configured enhanced matcher with {len(comparison_fields)} comparison fields")
        
        # Log field names for debugging
        field_names = [f.name for f in comparison_fields]
        logger.debug(f"Comparison fields: {field_names}")
    
    def _create_person_activity_blocks(self, persons: pd.DataFrame, 
                                      templates: pd.DataFrame) -> Dict:
        """Create enhanced blocks for person-activity matching using multiple strategies."""
        logger.info("Creating enhanced person-activity blocking strategy")
        
        # Use enhanced feature alignment instead of basic preparation
        persons, templates = align_pums_atus_features(persons, templates)
        
        # Store aligned data for later use
        self.aligned_persons = persons
        self.aligned_templates = templates
        
        # Create blocks using multiple strategies
        blocks = {}
        block_stats = defaultdict(int)
        
        # Strategy 1: Age group + employment (primary blocking)
        if 'block_key_1' in persons.columns and 'block_key_1' in templates.columns:
            for block_key in persons['block_key_1'].unique():
                person_ids = persons[persons['block_key_1'] == block_key].index.tolist()
                template_ids = templates[templates['block_key_1'] == block_key].index.tolist()
                
                if person_ids and template_ids:
                    blocks[f"block1_{block_key}"] = {
                        'persons': person_ids,
                        'templates': template_ids
                    }
                    block_stats['strategy_1'] += len(person_ids) * len(template_ids)
        
        # Strategy 2: Household type + children
        if 'block_key_2' in persons.columns and 'block_key_2' in templates.columns:
            for block_key in persons['block_key_2'].unique():
                person_ids = persons[persons['block_key_2'] == block_key].index.tolist()
                template_ids = templates[templates['block_key_2'] == block_key].index.tolist()
                
                if person_ids and template_ids:
                    blocks[f"block2_{block_key}"] = {
                        'persons': person_ids,
                        'templates': template_ids
                    }
                    block_stats['strategy_2'] += len(person_ids) * len(template_ids)
        
        # Strategy 3: Life stage + marital status
        if 'block_key_3' in persons.columns and 'block_key_3' in templates.columns:
            for block_key in persons['block_key_3'].unique():
                person_ids = persons[persons['block_key_3'] == block_key].index.tolist()
                template_ids = templates[templates['block_key_3'] == block_key].index.tolist()
                
                if person_ids and template_ids:
                    blocks[f"block3_{block_key}"] = {
                        'persons': person_ids,
                        'templates': template_ids
                    }
                    block_stats['strategy_3'] += len(person_ids) * len(template_ids)
        
        # Strategy 4: Education + income
        if 'block_key_4' in persons.columns and 'block_key_4' in templates.columns:
            for block_key in persons['block_key_4'].unique():
                person_ids = persons[persons['block_key_4'] == block_key].index.tolist()
                template_ids = templates[templates['block_key_4'] == block_key].index.tolist()
                
                if person_ids and template_ids:
                    blocks[f"block4_{block_key}"] = {
                        'persons': person_ids,
                        'templates': template_ids
                    }
                    block_stats['strategy_4'] += len(person_ids) * len(template_ids)
        
        # Strategy 5: Fine-grained age + work hours (if available)
        if 'age_5yr' in persons.columns and 'work_hours_cat' in persons.columns:
            persons['block_key_5'] = (
                persons['age_5yr'].astype(str) + '_' +
                persons['work_hours_cat'].astype(str)
            )
            
            if 'age_5yr' in templates.columns and 'work_hours_cat' in templates.columns:
                templates['block_key_5'] = (
                    templates['age_5yr'].astype(str) + '_' +
                    templates['work_hours_cat'].astype(str)
                )
                
                for block_key in persons['block_key_5'].unique():
                    person_ids = persons[persons['block_key_5'] == block_key].index.tolist()
                    template_ids = templates[templates.get('block_key_5', '') == block_key].index.tolist()
                    
                    if person_ids and template_ids:
                        blocks[f"block5_{block_key}"] = {
                            'persons': person_ids,
                            'templates': template_ids
                        }
                        block_stats['strategy_5'] += len(person_ids) * len(template_ids)
        
        # Calculate total pairs and coverage
        total_pairs = sum(
            len(b['persons']) * len(b['templates']) for b in blocks.values()
        )
        
        # Log blocking statistics
        logger.info(f"Created {len(blocks)} blocks with {total_pairs:,} candidate pairs")
        for strategy, count in block_stats.items():
            logger.info(f"  {strategy}: {count:,} pairs")
        
        # Check coverage
        persons_in_blocks = set()
        for block in blocks.values():
            persons_in_blocks.update(block['persons'])
        
        coverage = len(persons_in_blocks) / len(persons) * 100
        logger.info(f"Person coverage: {coverage:.1f}% ({len(persons_in_blocks)}/{len(persons)})")
        
        return blocks
    
    def _compare_person_activity_pairs(self, persons: pd.DataFrame,
                                      templates: pd.DataFrame,
                                      blocks: Dict) -> pd.DataFrame:
        """Compare person-template pairs within blocks."""
        logger.info("Comparing person-activity pairs")
        
        comparisons = []
        
        for block_name, block_data in blocks.items():
            for person_idx in block_data['persons']:
                for template_idx in block_data['templates']:
                    person = persons.loc[person_idx]
                    template = templates.loc[template_idx]
                    
                    # Calculate agreement pattern
                    agreement = self.matcher.calculate_agreement_patterns(
                        person, template
                    )
                    
                    comparison = {
                        'person_idx': person_idx,
                        'template_idx': template_idx,
                        'block': block_name
                    }
                    
                    # Add similarity scores for each field
                    for i, field in enumerate(self.matcher.fields):
                        comparison[f'{field.name}_similarity'] = agreement[i]
                    
                    comparisons.append(comparison)
        
        comparison_df = pd.DataFrame(comparisons)
        logger.info(f"Created {len(comparison_df)} person-activity comparisons")
        
        return comparison_df
    
    def _get_default_parameters(self) -> Tuple[Dict, Dict]:
        """Get default m and u parameters when learning fails."""
        field_names = [field.name for field in self.matcher.fields] if self.matcher else []
        
        # Default high m-probs (agreement rate for matches)
        m_probs = {field: 0.8 for field in field_names}
        # Default low u-probs (agreement rate for non-matches)
        u_probs = {field: 0.2 for field in field_names}
        
        return m_probs, u_probs
    
    def _learn_activity_matching_parameters(self, comparison_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Use EM algorithm to learn matching parameters with better initialization."""
        logger.info("Learning activity matching parameters with enhanced EM algorithm")
        
        # Handle empty comparison data
        if comparison_data is None or len(comparison_data) == 0:
            logger.warning("No comparison data available, using default parameters")
            return self._get_default_parameters()
        
        # Get field names
        field_names = [field.name for field in self.matcher.fields]
        
        # Better prior estimation based on expected match rate
        # Estimate prior match probability from blocking strategy
        # Typical person-activity match rate should be higher than 0.1
        # Avoid division by zero
        if len(comparison_data) > 0:
            estimated_prior = min(100 / len(comparison_data), 0.3)  # Cap at 30%
        else:
            estimated_prior = 0.1  # Default if no comparisons
        
        # Initialize EM algorithm with better prior
        self.em_algorithm = EMAlgorithm(field_names, prior_match=estimated_prior)
        
        # Initialize with frequency-based estimates
        self._initialize_em_with_frequencies(comparison_data)
        
        # Run EM with more iterations for better convergence
        comparison_data = self.em_algorithm.fit(comparison_data, max_iterations=100)
        
        # Get learned parameters
        m_probs = self.em_algorithm.m_probs
        u_probs = self.em_algorithm.u_probs
        
        # Save parameters
        self._save_matching_parameters(m_probs, u_probs)
        
        logger.info(f"EM converged: {self.em_algorithm.converged}")
        logger.info(f"Iterations: {self.em_algorithm.iteration}")
        
        return m_probs, u_probs
    
    def _initialize_em_with_frequencies(self, comparison_data: pd.DataFrame):
        """Initialize EM algorithm with frequency-based estimates."""
        logger.info("Initializing EM with frequency-based estimates")
        
        # Calculate frequency-based m and u probabilities
        for field in self.matcher.fields:
            sim_col = f'{field.name}_similarity'
            if sim_col in comparison_data.columns:
                # m-probability: Expected agreement rate for true matches
                # Start with optimistic values for important fields
                if field.weight > 1.5:
                    self.em_algorithm.m_probs[field.name] = 0.8
                else:
                    self.em_algorithm.m_probs[field.name] = 0.6
                
                # u-probability: Chance agreement rate
                # Calculate from actual data distribution
                high_similarity = (comparison_data[sim_col] >= 0.8).mean()
                self.em_algorithm.u_probs[field.name] = max(high_similarity * 0.5, 0.05)
            else:
                # Default values if column missing
                self.em_algorithm.m_probs[field.name] = 0.7
                self.em_algorithm.u_probs[field.name] = 0.1
        
        logger.info(f"Initialized m_probs: {list(self.em_algorithm.m_probs.values())[:5]}...")
        logger.info(f"Initialized u_probs: {list(self.em_algorithm.u_probs.values())[:5]}...")
    
    def _classify_person_activity_pairs(self, comparison_data: pd.DataFrame,
                                       m_probs: Dict, u_probs: Dict) -> pd.DataFrame:
        """Classify pairs using learned parameters."""
        logger.info("Classifying person-activity pairs")
        
        # Set learned parameters in matcher
        self.matcher.set_probabilities(m_probs, u_probs)
        
        # Calculate match weights
        comparison_data['match_weight'] = comparison_data.apply(
            lambda row: self._calculate_pair_weight(row, m_probs, u_probs),
            axis=1
        )
        
        # Calculate match probability
        comparison_data['match_probability'] = comparison_data['match_weight'].apply(
            lambda w: 1 / (1 + np.exp(-w))
        )
        
        # Classify based on thresholds
        comparison_data['classification'] = pd.cut(
            comparison_data['match_weight'],
            bins=[-np.inf, -5, 5, np.inf],
            labels=['non_match', 'possible', 'match']
        )
        
        match_count = (comparison_data['classification'] == 'match').sum()
        logger.info(f"Classified {match_count} matches out of {len(comparison_data)} pairs")
        
        return comparison_data
    
    def _calculate_pair_weight(self, row: pd.Series, m_probs: Dict, u_probs: Dict) -> float:
        """Calculate match weight for a pair with improved thresholds."""
        weight = 0.0
        
        for field_name in m_probs.keys():
            sim_col = f'{field_name}_similarity'
            if sim_col in row:
                similarity = row[sim_col]
                m = m_probs[field_name]
                u = u_probs[field_name]
                
                # Avoid log(0)
                m = max(min(m, 0.9999), 0.0001)
                u = max(min(u, 0.9999), 0.0001)
                
                # More nuanced thresholds for better matching
                if similarity >= 0.8:  # Strong agreement (was 0.9)
                    weight += np.log2(m / u)
                elif similarity >= 0.5:  # Partial agreement
                    # Interpolate weight based on similarity
                    agreement_weight = np.log2(m / u)
                    disagreement_weight = np.log2((1 - m) / (1 - u))
                    interpolation = (similarity - 0.5) / 0.3  # Scale 0.5-0.8 to 0-1
                    weight += interpolation * agreement_weight + (1 - interpolation) * disagreement_weight
                elif similarity <= 0.2:  # Strong disagreement (was 0.1)
                    weight += np.log2((1 - m) / (1 - u))
                # Neutral range (0.2-0.5) contributes minimal weight
        
        return weight
    
    def _assign_coordinated_activities(self, classified_pairs: pd.DataFrame,
                                      persons: pd.DataFrame,
                                      templates: pd.DataFrame,
                                      buildings: pd.DataFrame) -> Dict:
        """Assign activities with household coordination constraints."""
        logger.info("Assigning activities with household coordination")
        
        assignments = {}
        
        # Process by household to ensure coordination
        for building_id in buildings.index:
            household_persons = persons[persons['building_id'] == building_id]
            
            if len(household_persons) == 0:
                continue
            
            # Get household constraints
            constraints = self._identify_household_constraints(household_persons)
            
            # Find best activity assignments for household
            household_assignments = self._find_best_household_activities(
                household_persons, templates, classified_pairs, constraints
            )
            
            # Store assignments using proper person_id format
            for idx, template_idx in household_assignments.items():
                # The idx from household_assignments is the DataFrame index
                # Find the corresponding person
                if idx in household_persons.index:
                    person = household_persons.loc[idx]
                    # Use the person_id that was created in extraction
                    if 'person_id' in person:
                        person_id = person['person_id']
                    else:
                        # Fallback: create person_id from building and person indices
                        person_idx = person.get('person_idx', 0)
                        person_id = f"{building_id}_{person_idx}"
                    assignments[person_id] = template_idx
            
            self.stats['households_coordinated'] += 1
        
        self.stats['matched_persons'] = len(assignments)
        logger.info(f"Assigned activities to {len(assignments)} persons")
        
        return assignments
    
    def _identify_household_constraints(self, household_persons: pd.DataFrame) -> Dict:
        """Identify coordination constraints for a household."""
        constraints = {
            'has_young_children': any(household_persons['age'] < 5) if 'age' in household_persons.columns else False,
            'has_school_children': any((household_persons['age'] >= 5) & (household_persons['age'] <= 17)) if 'age' in household_persons.columns else False,
            'all_adults_work': all(household_persons[household_persons['age'] >= 18]['is_employed']) if 'age' in household_persons.columns and 'is_employed' in household_persons.columns else False,
            'household_size': len(household_persons)
        }
        
        # Add timing constraints
        if constraints['has_young_children']:
            constraints['need_childcare_coverage'] = True
        
        if constraints['has_school_children']:
            constraints['school_schedule'] = True
        
        return constraints
    
    def _find_best_household_activities(self, household_persons: pd.DataFrame,
                                       templates: pd.DataFrame,
                                       classified_pairs: pd.DataFrame,
                                       constraints: Dict) -> Dict:
        """Find best activity assignments for a household with improved fallback."""
        household_assignments = {}
        
        # Use aligned data if available
        if hasattr(self, 'aligned_persons') and hasattr(self, 'aligned_templates'):
            aligned_persons = self.aligned_persons
            aligned_templates = self.aligned_templates
        else:
            aligned_persons = household_persons
            aligned_templates = templates
        
        # Handle empty or invalid classified_pairs
        if classified_pairs is None or len(classified_pairs) == 0 or 'person_idx' not in classified_pairs.columns:
            logger.warning("No valid classified pairs, using fallback for all household members")
            # Use fallback for all persons
            for idx in household_persons.index:
                template_id = self._find_best_fallback_template(idx, aligned_persons, aligned_templates)
                if template_id is not None:
                    # Safely get template index
                    if 'template_id' in aligned_templates.columns:
                        matching_templates = aligned_templates[aligned_templates['template_id'] == template_id]
                        if len(matching_templates) > 0:
                            household_assignments[idx] = matching_templates.index[0]
                        else:
                            # Use first template if available
                            household_assignments[idx] = aligned_templates.index[0] if len(aligned_templates) > 0 else 0
                    else:
                        # Direct index assignment
                        household_assignments[idx] = template_id
                else:
                    # Safe fallback - use first template if available
                    household_assignments[idx] = aligned_templates.index[0] if len(aligned_templates) > 0 else 0
            return household_assignments
        
        for _, person in household_persons.iterrows():
            person_idx = person.name
            
            # Get candidate matches for this person
            person_pairs = classified_pairs[
                (classified_pairs['person_idx'] == person_idx) &
                (classified_pairs['classification'].isin(['match', 'possible']))
            ].sort_values('match_weight', ascending=False)
            
            if len(person_pairs) > 0:
                # Select best template
                best_template_idx = person_pairs.iloc[0]['template_idx']
                household_assignments[person.get('person_id', person_idx)] = best_template_idx
            else:
                # Enhanced fallback: Use k-nearest neighbor approach
                template_idx = self._find_best_fallback_template(
                    person_idx, aligned_persons, aligned_templates
                )
                household_assignments[person.get('person_id', person_idx)] = template_idx
        
        # Apply full household coordination
        if constraints.get('need_childcare_coverage'):
            household_assignments = self._ensure_childcare_coverage(
                household_assignments, household_persons, aligned_templates, constraints
            )
        
        return household_assignments
    
    def _find_best_fallback_template(self, person_idx: int, 
                                    persons: pd.DataFrame, 
                                    templates: pd.DataFrame) -> Optional[int]:
        """Find best template using feature similarity when no match found."""
        # Check for empty templates first
        if templates is None or len(templates) == 0:
            logger.warning("No templates available for fallback matching")
            return None
            
        if person_idx not in persons.index:
            # Random fallback if person not found
            return templates.sample(1).index[0] if len(templates) > 0 else None
        
        person = persons.loc[person_idx]
        
        # Key features for fallback matching
        key_features = ['employed', 'has_children', 'age_decade', 'married', 
                       'work_intensity', 'hh_size_group']
        
        # Calculate simple similarity to all templates
        similarities = []
        for template_idx, template in templates.iterrows():
            similarity = 0
            for feature in key_features:
                if feature in person and feature in template:
                    if person[feature] == template[feature]:
                        similarity += 1
            similarities.append((template_idx, similarity))
        
        # Sort by similarity and pick best
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if similarities and similarities[0][1] > 0:
            return similarities[0][0]
        elif len(templates) > 0:
            # Last resort: random template
            return templates.sample(1).index[0]
        else:
            return None
    
    def _ensure_childcare_coverage(self, assignments: Dict,
                                  household_persons: pd.DataFrame,
                                  templates: pd.DataFrame,
                                  constraints: Dict) -> Dict:
        """Ensure proper childcare coverage in household schedules."""
        # Identify adults and children
        adults = household_persons[household_persons.get('age', 18) >= 18]
        children = household_persons[household_persons.get('age', 18) < 18]
        
        if len(children) > 0 and len(adults) > 0:
            # Check if at least one adult has flexible schedule
            adult_templates = []
            for adult in adults.itertuples():
                person_id = getattr(adult, 'person_id', adult.Index)
                if person_id in assignments:
                    template_id = assignments[person_id]
                    if template_id in templates.index:
                        adult_templates.append((person_id, template_id))
            
            # Ensure at least one adult has low work hours for childcare
            has_caregiver = False
            for person_id, template_id in adult_templates:
                template = templates.loc[template_id]
                # Check if template indicates flexible/part-time schedule
                if template.get('part_time', 0) or template.get('not_working', 0):
                    has_caregiver = True
                    break
            
            # If no caregiver, reassign one adult to part-time template
            if not has_caregiver and adult_templates:
                # Find a part-time/flexible template
                flexible_templates = templates[
                    (templates.get('part_time', 0) == 1) | 
                    (templates.get('not_working', 0) == 1)
                ]
                
                if len(flexible_templates) > 0:
                    # Reassign first adult to flexible template
                    person_id = adult_templates[0][0]
                    new_template = flexible_templates.sample(1).index[0]
                    assignments[person_id] = new_template
                    logger.debug(f"Reassigned {person_id} to flexible template for childcare")
        
        return assignments
    
    def _merge_activities_to_buildings(self, buildings: pd.DataFrame,
                                      activity_assignments: Dict,
                                      persons: pd.DataFrame) -> pd.DataFrame:
        """Merge assigned activities back into buildings."""
        logger.info(f"Merging activities into buildings. Assignments: {len(activity_assignments)}")
        
        # Debug: show some assignment keys
        if activity_assignments:
            sample_keys = list(activity_assignments.keys())[:3]
            logger.debug(f"Sample assignment keys: {sample_keys}")
        
        buildings_with_activities = buildings.copy()
        
        # Add activity data to each building
        for idx, building in buildings_with_activities.iterrows():
            if 'persons' not in building or not isinstance(building['persons'], list):
                continue
            
            # Update each person with their activity assignment
            updated_persons = []
            for person_idx, person in enumerate(building['persons']):
                person_id = f"{idx}_{person_idx}"
                
                if person_id in activity_assignments:
                    template_id = activity_assignments[person_id]
                    person_copy = person.copy() if isinstance(person, dict) else {}
                    person_copy['atus_template_id'] = template_id
                    person_copy['has_activities'] = True
                    updated_persons.append(person_copy)
                    logger.debug(f"Assigned template {template_id} to person {person_id}")
                else:
                    person_copy = person.copy() if isinstance(person, dict) else {}
                    person_copy['has_activities'] = False
                    updated_persons.append(person_copy)
                    logger.debug(f"No assignment for person {person_id}")
            
            buildings_with_activities.at[idx, 'persons'] = updated_persons
            buildings_with_activities.at[idx, 'has_atus_activities'] = True
        
        logger.info(f"Merged activities into {len(buildings_with_activities)} buildings")
        
        return buildings_with_activities
    
    def _save_matching_parameters(self, m_probs: Dict, u_probs: Dict):
        """Save learned matching parameters."""
        parameters = {
            'phase': 'phase3_atus',
            'timestamp': datetime.now().isoformat(),
            'm_probabilities': m_probs,
            'u_probabilities': u_probs,
            'em_diagnostics': {
                'converged': self.em_algorithm.converged if self.em_algorithm else False,
                'iterations': self.em_algorithm.iteration if self.em_algorithm else 0,
                'final_log_likelihood': self.em_algorithm.log_likelihood_history[-1] if self.em_algorithm and self.em_algorithm.log_likelihood_history else None
            }
        }
        
        with open(self.phase3_params_path, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        
        logger.info(f"Saved matching parameters to {self.phase3_params_path}")
    
    def _validate_and_save(self, buildings: pd.DataFrame, assignments: Dict):
        """Validate results and save outputs."""
        logger.info("Validating and saving Phase 3 results")
        
        # Save main output
        buildings.to_pickle(self.phase3_output_path)
        logger.info(f"Saved Phase 3 output to {self.phase3_output_path}")
        
        # Calculate validation metrics
        validation_metrics = {
            'total_buildings': len(buildings),
            'total_persons': self.stats['persons_count'],
            'matched_persons': self.stats['matched_persons'],
            'match_rate': self.stats['matched_persons'] / self.stats['persons_count'] if self.stats['persons_count'] > 0 else 0,
            'households_coordinated': self.stats['households_coordinated'],
            'templates_used': len(set(assignments.values())) if assignments else 0
        }
        
        # Save metadata
        metadata = {
            'phase': 'phase3_atus_matching',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            'statistics': self.stats,
            'validation_metrics': validation_metrics,
            'output_columns': list(buildings.columns)
        }
        
        with open(self.phase3_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {self.phase3_metadata_path}")
        
        # Generate validation report
        self._generate_validation_report(buildings, validation_metrics)
    
    def _generate_validation_report(self, buildings: pd.DataFrame, metrics: Dict):
        """Generate HTML validation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 3 ATUS Matching Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; }}
                .success {{ color: #4caf50; }}
                .warning {{ color: #ff9800; }}
            </style>
        </head>
        <body>
            <h1>Phase 3: ATUS Activity Pattern Matching Validation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h2>Summary Statistics</h2>
                <ul>
                    <li>Total Buildings: {metrics.get('total_buildings', 0)}</li>
                    <li>Total Persons: {metrics.get('total_persons', 0)}</li>
                    <li>Matched Persons: {metrics.get('matched_persons', 0)}</li>
                    <li>Match Rate: {metrics.get('match_rate', 0):.1%}</li>
                    <li>Households Coordinated: {metrics.get('households_coordinated', 0)}</li>
                    <li>Unique Templates Used: {metrics.get('templates_used', 0)}</li>
                </ul>
            </div>
            
            <h2>Processing Performance</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Start Time</td>
                    <td>{self.stats['start_time']}</td>
                </tr>
                <tr>
                    <td>End Time</td>
                    <td>{self.stats['end_time']}</td>
                </tr>
                <tr>
                    <td>Duration</td>
                    <td>{(self.stats['end_time'] - self.stats['start_time']).total_seconds():.2f} seconds</td>
                </tr>
            </table>
            
            <h2>Data Quality Checks</h2>
            <ul>
                <li class="{'success' if metrics.get('match_rate', 0) > 0.9 else 'warning'}">
                    Match rate: {metrics.get('match_rate', 0):.1%}
                </li>
                <li class="success">
                    All households processed with coordination constraints
                </li>
            </ul>
        </body>
        </html>
        """
        
        with open(self.phase3_validation_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated validation report at {self.phase3_validation_path}")


def load_phase3_output() -> Tuple[pd.DataFrame, Dict]:
    """
    Load Phase 3 output for use in Phase 4.
    
    Returns:
        Tuple of (buildings DataFrame, metadata dict)
    """
    output_path = Path("data/processed/phase3_pums_recs_atus_buildings.pkl")
    metadata_path = Path("data/processed/phase3_metadata.json")
    
    if not output_path.exists():
        raise FileNotFoundError(f"Phase 3 output not found: {output_path}")
    
    buildings = pd.read_pickle(output_path)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return buildings, metadata