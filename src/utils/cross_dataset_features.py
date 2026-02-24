"""
Cross-Dataset Feature Engineering for Enhanced Matching.

This module creates advanced features that are comparable across PUMS, RECS, and ATUS datasets
to improve matching quality. Features are designed to capture household characteristics,
demographic patterns, and behavioral indicators that exist in multiple datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class CrossDatasetFeatureEngineer:
    """
    Creates standardized features across PUMS, RECS, and ATUS for improved matching.
    
    This class implements sophisticated feature engineering to create comparable
    features across all three datasets, enabling better probabilistic matching.
    """
    
    def __init__(self):
        """Initialize the feature engineer with standard mappings."""
        self.feature_mappings = self._initialize_mappings()
        self.created_features = set()
    
    def _initialize_mappings(self) -> Dict:
        """Initialize mappings between dataset-specific fields."""
        return {
            'household_size': {
                'pums': 'NP',
                'recs': 'NHSLDMEM',
                'atus': 'HHSIZE'
            },
            'income': {
                'pums': 'HINCP',
                'recs': 'MONEYPY',
                'atus': 'HEFAMINC'
            },
            'age': {
                'pums': 'AGEP',
                'recs': 'HHAGE',
                'atus': 'TEAGE'
            },
            'state': {
                'pums': 'STATE',
                'recs': 'STATE_FIPS',
                'atus': 'GESTFIPS'
            }
        }
    
    def create_universal_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Create universal features that exist across all datasets.
        
        Args:
            df: Input dataframe from PUMS, RECS, or ATUS
            dataset_type: One of 'pums', 'recs', 'atus'
            
        Returns:
            DataFrame with additional universal features
        """
        df = df.copy()
        dataset_type = dataset_type.lower()
        
        # Household size features
        df = self._create_household_size_features(df, dataset_type)
        
        # Income features
        df = self._create_income_features(df, dataset_type)
        
        # Geographic features
        df = self._create_geographic_features(df, dataset_type)
        
        # Temporal features
        df = self._create_temporal_features(df, dataset_type)
        
        # Composite indices
        df = self._create_composite_indices(df, dataset_type)
        
        # Interaction features
        df = self._create_interaction_features(df, dataset_type)
        
        # Energy-related features (for PUMS-RECS matching)
        df = self._create_energy_proxy_features(df, dataset_type)
        
        # Activity-related features (for PUMS-ATUS matching)
        df = self._create_activity_proxy_features(df, dataset_type)
        
        logger.info(f"Created {len(self.created_features)} cross-dataset features for {dataset_type}")
        
        return df
    
    def _create_household_size_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create standardized household size features."""
        size_col = self.feature_mappings['household_size'].get(dataset_type)
        
        if size_col and size_col in df.columns:
            # Basic size categories
            df['hh_size_binary'] = (df[size_col] > 1).astype(int)
            
            # Handle size categories with proper checks
            if df[size_col].notna().any():
                try:
                    df['hh_size_3cat'] = pd.cut(df[size_col], 
                                                bins=[0, 1, 3, 100],
                                                labels=['single', 'small', 'large'])
                except (ValueError, TypeError):
                    df['hh_size_3cat'] = 'small'
                    
                try:
                    df['hh_size_5cat'] = pd.cut(df[size_col],
                                                bins=[0, 1, 2, 3, 5, 100],
                                                labels=['single', 'couple', 'small_family', 
                                                       'medium_family', 'large_family'])
                except (ValueError, TypeError):
                    df['hh_size_5cat'] = 'small_family'
            else:
                df['hh_size_3cat'] = 'small'
                df['hh_size_5cat'] = 'small_family'
            
            # Numeric transformations
            df['hh_size_log'] = np.log1p(df[size_col])
            df['hh_size_sqrt'] = np.sqrt(df[size_col])
            df['hh_size_squared'] = df[size_col] ** 2
            
            # Statistical bins (only if enough data variety)
            if len(df) > 100 and df[size_col].notna().any() and df[size_col].nunique() > 1:
                try:
                    df['hh_size_quartile'] = pd.qcut(df[size_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                except (ValueError, TypeError):
                    # Not enough unique values for quartiles
                    pass
                try:
                    df['hh_size_decile'] = pd.qcut(df[size_col], q=10, labels=False, duplicates='drop')
                except (ValueError, TypeError):
                    # Not enough unique values for deciles
                    pass
            
            # Flags
            df['is_single_person'] = (df[size_col] == 1).astype(int)
            df['is_large_household'] = (df[size_col] >= 5).astype(int)
            df['is_typical_family'] = df[size_col].isin([3, 4]).astype(int)
            
            self.created_features.update([
                'hh_size_binary', 'hh_size_3cat', 'hh_size_5cat',
                'hh_size_log', 'hh_size_sqrt', 'hh_size_squared',
                'is_single_person', 'is_large_household', 'is_typical_family'
            ])
        
        return df
    
    def _create_income_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create standardized income features."""
        income_col = self.feature_mappings['income'].get(dataset_type)
        
        if income_col and income_col in df.columns:
            # Handle missing/negative values
            # Handle income with proper NaN checks
            median_income = df[income_col].median()
            if pd.isna(median_income):
                # If all incomes are NaN, use a default value
                median_income = 50000  # Approximate US median household income
            
            df['income_clean'] = df[income_col].fillna(median_income)
            df['income_clean'] = df['income_clean'].clip(lower=0)
            
            # Income categories - check if we have valid data
            if df['income_clean'].nunique() > 1 and not df['income_clean'].isna().all():
                df['income_binary'] = (df['income_clean'] > df['income_clean'].median()).astype(int)
                
                try:
                    df['income_tercile'] = pd.qcut(df['income_clean'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                except (ValueError, TypeError):
                    # Not enough unique values or other issues
                    try:
                        df['income_tercile'] = pd.cut(df['income_clean'], bins=3, labels=['low', 'medium', 'high'])
                    except (ValueError, TypeError):
                        # If still failing, assign default
                        df['income_tercile'] = 'medium'
                
                try:
                    df['income_quintile'] = pd.qcut(df['income_clean'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                except (ValueError, TypeError):
                    # Not enough unique values or other issues
                    try:
                        df['income_quintile'] = pd.cut(df['income_clean'], bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                    except (ValueError, TypeError):
                        # If still failing, assign default
                        df['income_quintile'] = 'Q3'
            else:
                # If no valid income data, set defaults
                df['income_binary'] = 0
                df['income_tercile'] = 'medium'
                df['income_quintile'] = 'Q3'
            
            # Poverty indicators (using federal poverty guidelines approximation)
            if 'hh_size_clean' in df.columns or self.feature_mappings['household_size'].get(dataset_type) in df.columns:
                size_col = 'hh_size_clean' if 'hh_size_clean' in df.columns else self.feature_mappings['household_size'].get(dataset_type)
                poverty_threshold = 12880 + (df[size_col] - 1) * 4540  # 2023 guidelines
                df['below_poverty'] = (df['income_clean'] < poverty_threshold).astype(int)
                df['poverty_ratio'] = df['income_clean'] / poverty_threshold.clip(lower=1)  # Avoid division by zero
                
                # Handle poverty category with proper checks
                if df['poverty_ratio'].notna().any() and df['poverty_ratio'].nunique() > 1:
                    try:
                        df['poverty_category'] = pd.cut(df['poverty_ratio'],
                                                       bins=[0, 0.5, 1.0, 2.0, 100],
                                                       labels=['extreme_poverty', 'poverty', 'near_poverty', 'above_poverty'])
                    except (ValueError, TypeError):
                        df['poverty_category'] = 'above_poverty'
                else:
                    df['poverty_category'] = 'above_poverty'
            
            # Income transformations
            df['income_log'] = np.log1p(df['income_clean'])
            df['income_sqrt'] = np.sqrt(df['income_clean'])
            
            # Income per capita
            if self.feature_mappings['household_size'].get(dataset_type) in df.columns:
                size_col = self.feature_mappings['household_size'].get(dataset_type)
                df['income_per_capita'] = df['income_clean'] / df[size_col].clip(lower=1)
                
                # Handle income per capita categories with proper checks
                if df['income_per_capita'].notna().any() and df['income_per_capita'].nunique() > 1:
                    try:
                        df['income_per_capita_cat'] = pd.qcut(df['income_per_capita'], 
                                                              q=4, 
                                                              labels=['low', 'med_low', 'med_high', 'high'],
                                                              duplicates='drop')
                    except (ValueError, TypeError):
                        # Not enough unique values or other issues
                        try:
                            df['income_per_capita_cat'] = pd.cut(df['income_per_capita'], 
                                                                 bins=4, 
                                                                 labels=['low', 'med_low', 'med_high', 'high'])
                        except (ValueError, TypeError):
                            df['income_per_capita_cat'] = 'med_low'
                else:
                    df['income_per_capita_cat'] = 'med_low'
            
            self.created_features.update([
                'income_binary', 'income_tercile', 'income_quintile',
                'income_log', 'income_sqrt', 'income_per_capita'
            ])
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create standardized geographic features."""
        state_col = self.feature_mappings['state'].get(dataset_type)
        
        if state_col and state_col in df.columns:
            # Regional groupings
            df['census_region'] = self._map_state_to_region(df[state_col])
            df['census_division'] = self._map_state_to_division(df[state_col])
            
            # Climate zones (simplified)
            df['climate_zone'] = self._map_state_to_climate(df[state_col])
            
            # Urban/rural proxy (if available)
            if 'PUMA' in df.columns:  # PUMS specific
                df['urban_rural_proxy'] = self._create_urban_rural_proxy(df)
            elif 'UATYP10' in df.columns:  # RECS specific
                df['urban_rural_proxy'] = df['UATYP10'].map({
                    'U': 'urban', 'C': 'urban', 'R': 'rural'
                })
            
            # State characteristics
            df['is_coastal'] = self._is_coastal_state(df[state_col])
            df['is_southern'] = (df['census_region'] == 'South').astype(int)
            df['is_cold_climate'] = df['climate_zone'].isin(['very_cold', 'cold']).astype(int)
            
            self.created_features.update([
                'census_region', 'census_division', 'climate_zone',
                'is_coastal', 'is_southern', 'is_cold_climate'
            ])
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create temporal features if date/time information available."""
        # Year built features (PUMS/RECS)
        if 'YRBLT' in df.columns:  # PUMS
            current_year = 2023
            df['building_age'] = current_year - df['YRBLT']
            df['building_decade'] = (df['YRBLT'] // 10) * 10
            df['is_new_building'] = (df['building_age'] <= 10).astype(int)
            df['is_old_building'] = (df['building_age'] >= 50).astype(int)
            
            # Handle building era with proper checks
            if df['YRBLT'].notna().any():
                try:
                    df['building_era'] = pd.cut(df['YRBLT'],
                                               bins=[0, 1950, 1980, 2000, 2010, 2030],
                                               labels=['pre1950', '1950-1980', '1980-2000', '2000-2010', 'post2010'])
                except (ValueError, TypeError):
                    df['building_era'] = '1980-2000'
            else:
                df['building_era'] = '1980-2000'
        elif 'YEARMADERANGE' in df.columns:  # RECS
            df['building_era'] = df['YEARMADERANGE']
            df['is_new_building'] = df['YEARMADERANGE'].isin(['2010-2015', '2016-2020']).astype(int)
            df['is_old_building'] = df['YEARMADERANGE'].isin(['Before 1950', '1950-1959']).astype(int)
        
        # Day of week features (ATUS)
        if 'TUDIARYDAY' in df.columns:
            df['is_weekend'] = df['TUDIARYDAY'].isin([1, 7]).astype(int)
            df['is_weekday'] = (~df['TUDIARYDAY'].isin([1, 7])).astype(int)
        
        self.created_features.update(['building_age', 'building_era', 'is_new_building', 'is_old_building'])
        
        return df
    
    def _create_composite_indices(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create composite indices combining multiple features."""
        # Socioeconomic Status (SES) Index
        ses_components = []
        
        if 'income_quintile' in df.columns:
            if hasattr(df['income_quintile'], 'cat'):
                ses_components.append(df['income_quintile'].cat.codes)
            else:
                # If not categorical, map the labels
                label_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'Q5': 4}
                ses_components.append(df['income_quintile'].map(label_map).fillna(2))
        if 'SCHL' in df.columns:  # Education in PUMS
            schl_values = df['SCHL'].fillna(0)
            if schl_values.nunique() > 1 and not schl_values.isna().all():
                try:
                    ses_components.append(pd.qcut(schl_values, q=5, labels=False, duplicates='drop'))
                except (ValueError, TypeError):
                    try:
                        ses_components.append(pd.cut(schl_values, bins=5, labels=False))
                    except (ValueError, TypeError):
                        ses_components.append(np.ones(len(df)) * 2)  # Default middle value
            else:
                ses_components.append(np.ones(len(df)) * 2)
        if 'ESR' in df.columns:  # Employment status in PUMS
            ses_components.append((df['ESR'].isin([1, 2])).astype(int) * 5)
        
        if ses_components:
            df['ses_index'] = np.mean(ses_components, axis=0)
            if df['ses_index'].nunique() > 1 and not df['ses_index'].isna().all():
                try:
                    df['ses_category'] = pd.qcut(df['ses_index'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                except (ValueError, TypeError):
                    try:
                        df['ses_category'] = pd.cut(df['ses_index'], bins=3, labels=['low', 'medium', 'high'])
                    except (ValueError, TypeError):
                        df['ses_category'] = 'medium'
            else:
                df['ses_category'] = 'medium'
        
        # Housing Quality Index
        quality_components = []
        
        if 'building_age' in df.columns:
            age_values = df['building_age'].fillna(50)
            if age_values.nunique() > 1 and not age_values.isna().all():
                try:
                    quality_components.append(5 - pd.qcut(age_values, q=5, labels=False, duplicates='drop'))
                except (ValueError, TypeError):
                    try:
                        quality_components.append(5 - pd.cut(age_values, bins=5, labels=False))
                    except (ValueError, TypeError):
                        quality_components.append(np.ones(len(df)) * 2.5)
            else:
                quality_components.append(np.ones(len(df)) * 2.5)
        if 'RMSP' in df.columns:  # Number of rooms
            room_values = df['RMSP'].fillna(4)
            if room_values.nunique() > 1 and not room_values.isna().all():
                try:
                    quality_components.append(pd.qcut(room_values, q=5, labels=False, duplicates='drop'))
                except (ValueError, TypeError):
                    try:
                        quality_components.append(pd.cut(room_values, bins=5, labels=False))
                    except (ValueError, TypeError):
                        quality_components.append(np.ones(len(df)) * 2)
            else:
                quality_components.append(np.ones(len(df)) * 2)
        if 'BDSP' in df.columns:  # Number of bedrooms
            bed_values = df['BDSP'].fillna(2)
            if bed_values.nunique() > 1 and not bed_values.isna().all():
                try:
                    quality_components.append(pd.qcut(bed_values, q=5, labels=False, duplicates='drop'))
                except (ValueError, TypeError):
                    try:
                        quality_components.append(pd.cut(bed_values, bins=5, labels=False))
                    except (ValueError, TypeError):
                        quality_components.append(np.ones(len(df)) * 2)
            else:
                quality_components.append(np.ones(len(df)) * 2)
        
        if quality_components:
            df['housing_quality_index'] = np.mean(quality_components, axis=0)
            if df['housing_quality_index'].nunique() > 1 and not df['housing_quality_index'].isna().all():
                try:
                    df['housing_quality_cat'] = pd.qcut(df['housing_quality_index'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                except (ValueError, TypeError):
                    try:
                        df['housing_quality_cat'] = pd.cut(df['housing_quality_index'], bins=3, labels=['low', 'medium', 'high'])
                    except (ValueError, TypeError):
                        df['housing_quality_cat'] = 'medium'
            else:
                df['housing_quality_cat'] = 'medium'
        
        # Energy Vulnerability Index
        vuln_score = 0
        vuln_factors = 0
        
        if 'income_tercile' in df.columns:
            vuln_score += (df['income_tercile'] == 'low').astype(int)
            vuln_factors += 1
        if 'is_old_building' in df.columns:
            vuln_score += df['is_old_building']
            vuln_factors += 1
        if 'climate_zone' in df.columns:
            vuln_score += df['climate_zone'].isin(['very_cold', 'hot_humid']).astype(int)
            vuln_factors += 1
        if 'is_large_household' in df.columns:
            vuln_score += df['is_large_household']
            vuln_factors += 1
        
        if vuln_factors > 0:
            df['energy_vulnerability'] = vuln_score / vuln_factors
            df['high_energy_vulnerable'] = (df['energy_vulnerability'] > 0.5).astype(int)
        
        self.created_features.update(['ses_index', 'housing_quality_index', 'energy_vulnerability'])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create interaction features between key variables."""
        # Income × Household Size
        if 'income_tercile' in df.columns and 'hh_size_3cat' in df.columns:
            df['income_size_interaction'] = (
                df['income_tercile'].astype(str) + '_' + 
                df['hh_size_3cat'].astype(str)
            )
        
        # Age × Employment (for person-level matching)
        if 'AGEP' in df.columns and 'ESR' in df.columns:
            df['working_age_adult'] = (
                (df['AGEP'].between(25, 65)) & 
                (df['ESR'].isin([1, 2]))
            ).astype(int)
            df['retired_senior'] = (
                (df['AGEP'] >= 65) & 
                (df['ESR'].isin([6]))
            ).astype(int)
        
        # Geographic × Income
        if 'census_region' in df.columns and 'income_quintile' in df.columns:
            df['region_income'] = (
                df['census_region'].astype(str) + '_' + 
                df['income_quintile'].astype(str)
            )
        
        # Building Age × Climate
        if 'building_era' in df.columns and 'climate_zone' in df.columns:
            df['building_climate_risk'] = (
                df['building_era'].isin(['pre1950', '1950-1980']).astype(int) * 
                df['climate_zone'].isin(['very_cold', 'hot_humid']).astype(int)
            )
        
        self.created_features.update(['income_size_interaction', 'region_income'])
        
        return df
    
    def _create_energy_proxy_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create energy-related proxy features for PUMS-RECS matching."""
        # Heating/Cooling needs proxy
        if 'climate_zone' in df.columns:
            df['high_heating_need'] = df['climate_zone'].isin(['very_cold', 'cold']).astype(int)
            df['high_cooling_need'] = df['climate_zone'].isin(['hot_humid', 'hot_dry']).astype(int)
        
        # Energy intensity proxy (based on building characteristics)
        intensity_score = 0
        intensity_factors = 0
        
        if 'building_age' in df.columns:
            intensity_score += (df['building_age'] > 30).astype(int)
            intensity_factors += 1
        if 'hh_size_3cat' in df.columns:
            intensity_score += (df['hh_size_3cat'] == 'large').astype(int)
            intensity_factors += 1
        if 'RMSP' in df.columns:  # More rooms = more energy
            intensity_score += (df['RMSP'] > df['RMSP'].median()).astype(int)
            intensity_factors += 1
        
        if intensity_factors > 0:
            df['energy_intensity_proxy'] = intensity_score / intensity_factors
        
        # Appliance proxy (based on income and household size)
        if 'income_quintile' in df.columns and 'hh_size_3cat' in df.columns:
            df['high_appliance_use'] = (
                (df['income_quintile'].isin(['Q4', 'Q5'])) & 
                (df['hh_size_3cat'].isin(['small', 'large']))
            ).astype(int)
        
        # Electric vehicle charger derivation (literature-calibrated logistic model)
        # Apply the full 3-stage model for PUMS datasets
        if dataset_type == 'pums' and 'ev_charger_prob' not in df.columns:
            try:
                from .enhanced_feature_engineering import derive_ev_charger_features
                df = derive_ev_charger_features(df, dataset_type='pums')
            except Exception as e:
                logger.warning(f"EV charger derivation failed, using fallback: {e}")

        # Set ev_likely from calibrated probability if available, otherwise heuristic
        if 'ev_charger_prob' in df.columns:
            df['ev_likely'] = (df['ev_charger_prob'] > 0.10).astype(int)
        elif 'income_quintile' in df.columns and 'is_new_building' in df.columns:
            # Fallback: simplified heuristic if full model not yet applied
            df['ev_likely'] = (
                (df['income_quintile'].isin(['Q4', 'Q5', 'q4', 'q5'])) &
                (df['is_new_building'].astype(bool))
            ).astype(int)

        self.created_features.update([
            'high_heating_need', 'high_cooling_need', 'energy_intensity_proxy',
            'ev_likely', 'ev_charger_prob', 'has_ev_charger', 'charger_level', 'charger_capacity_kw'
        ])
        
        return df
    
    def _create_activity_proxy_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Create activity-related proxy features for PUMS-ATUS matching."""
        # Work from home proxy
        if 'ESR' in df.columns and 'JWTR' in df.columns:  # PUMS
            df['work_from_home'] = (
                (df['ESR'].isin([1, 2])) & 
                (df['JWTR'] == 0)  # JWTR=0 means work from home
            ).astype(int)
        
        # Childcare responsibility proxy
        if 'hh_size_3cat' in df.columns:
            has_children = False
            if 'HUPAC' in df.columns:  # PUMS children presence
                has_children = df['HUPAC'].isin([1, 2, 3])
            elif 'TRCHILDNUM' in df.columns:  # ATUS
                has_children = df['TRCHILDNUM'] > 0
            
            if isinstance(has_children, pd.Series):
                df['childcare_likely'] = has_children.astype(int)
        
        # Retirement activity proxy
        if 'AGEP' in df.columns:
            df['retirement_activities'] = (df['AGEP'] >= 65).astype(int)
        
        # Student activity proxy
        if 'SCHL' in df.columns and 'AGEP' in df.columns:
            df['student_activities'] = (
                (df['AGEP'].between(18, 25)) & 
                (df['SCHL'].between(18, 21))  # College level education
            ).astype(int)
        
        # Commute time proxy (affects daily schedule)
        if 'JWTR' in df.columns and 'JWMNP' in df.columns:
            df['long_commute'] = (df['JWMNP'] > 45).astype(int)  # >45 min commute
            df['no_commute'] = (df['JWTR'] == 0).astype(int)  # Work from home
        
        self.created_features.update(['work_from_home', 'childcare_likely', 'retirement_activities'])
        
        return df
    
    def _map_state_to_region(self, state_series: pd.Series) -> pd.Series:
        """Map state codes to census regions."""
        # Simplified mapping - would need complete FIPS to region mapping
        northeast = ['09', '23', '25', '33', '44', '50', '34', '36', '42']
        midwest = ['17', '18', '26', '39', '55', '19', '20', '27', '29', '31', '38', '46']
        south = ['10', '11', '12', '13', '24', '37', '45', '51', '54', '01', '21', '28', '47', 
                '05', '22', '40', '48']
        west = ['04', '08', '16', '30', '32', '35', '49', '56', '02', '06', '15', '41', '53']
        
        def map_region(state):
            state_str = str(state).zfill(2)
            if state_str in northeast:
                return 'Northeast'
            elif state_str in midwest:
                return 'Midwest'
            elif state_str in south:
                return 'South'
            elif state_str in west:
                return 'West'
            else:
                return 'Unknown'
        
        return state_series.apply(map_region)
    
    def _map_state_to_division(self, state_series: pd.Series) -> pd.Series:
        """Map state codes to census divisions."""
        # Simplified - would need complete mapping
        division_map = {
            '09': 'New England', '23': 'New England', '25': 'New England',
            '33': 'New England', '44': 'New England', '50': 'New England',
            '34': 'Middle Atlantic', '36': 'Middle Atlantic', '42': 'Middle Atlantic',
            '17': 'East North Central', '18': 'East North Central', '26': 'East North Central',
            '39': 'East North Central', '55': 'East North Central',
            '19': 'West North Central', '20': 'West North Central', '27': 'West North Central',
            '29': 'West North Central', '31': 'West North Central', '38': 'West North Central',
            '46': 'West North Central',
            '10': 'South Atlantic', '11': 'South Atlantic', '12': 'South Atlantic',
            '13': 'South Atlantic', '24': 'South Atlantic', '37': 'South Atlantic',
            '45': 'South Atlantic', '51': 'South Atlantic', '54': 'South Atlantic',
            '01': 'East South Central', '21': 'East South Central', '28': 'East South Central',
            '47': 'East South Central',
            '05': 'West South Central', '22': 'West South Central', '40': 'West South Central',
            '48': 'West South Central',
            '04': 'Mountain', '08': 'Mountain', '16': 'Mountain', '30': 'Mountain',
            '32': 'Mountain', '35': 'Mountain', '49': 'Mountain', '56': 'Mountain',
            '02': 'Pacific', '06': 'Pacific', '15': 'Pacific', '41': 'Pacific', '53': 'Pacific'
        }
        
        return state_series.astype(str).str.zfill(2).map(division_map).fillna('Unknown')
    
    def _map_state_to_climate(self, state_series: pd.Series) -> pd.Series:
        """Map state codes to simplified climate zones."""
        # Very simplified climate mapping
        very_cold = ['02', '23', '25', '33', '44', '50', '26', '27', '55', '30', '38', '46', '56']
        cold = ['09', '34', '36', '42', '17', '18', '39', '19', '20', '29', '31', '08', '16', '49']
        mixed = ['10', '11', '24', '51', '54', '21', '47', '37', '40', '35']
        hot_humid = ['12', '13', '45', '01', '28', '05', '22', '48']
        hot_dry = ['04', '06', '32', '35']
        marine = ['41', '53', '15']
        
        def map_climate(state):
            state_str = str(state).zfill(2)
            if state_str in very_cold:
                return 'very_cold'
            elif state_str in cold:
                return 'cold'
            elif state_str in mixed:
                return 'mixed'
            elif state_str in hot_humid:
                return 'hot_humid'
            elif state_str in hot_dry:
                return 'hot_dry'
            elif state_str in marine:
                return 'marine'
            else:
                return 'mixed'
        
        return state_series.apply(map_climate)
    
    def _is_coastal_state(self, state_series: pd.Series) -> pd.Series:
        """Identify coastal states."""
        coastal = ['06', '41', '53', '02', '15',  # Pacific
                  '09', '23', '25', '33', '44', '34', '36',  # Atlantic Northeast
                  '10', '11', '24', '51', '37', '45', '13', '12',  # Atlantic South
                  '01', '22', '48', '28']  # Gulf
        
        return state_series.astype(str).str.zfill(2).isin(coastal).astype(int)
    
    def _create_urban_rural_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Create urban/rural proxy from available data."""
        # For PUMS, use PUMA characteristics as proxy
        # This is simplified - would need PUMA to urban/rural mapping
        if 'PUMA' in df.columns:
            # PUMAs starting with certain codes tend to be urban
            # This is a very rough approximation
            urban_indicator = df['PUMA'].astype(str).str[:2].isin(['01', '02', '03'])
            return urban_indicator.map({True: 'urban', False: 'rural'})
        else:
            return pd.Series(['unknown'] * len(df))
    
    def get_matching_features(self, dataset_type: str) -> List[str]:
        """
        Get list of features suitable for matching based on dataset type.
        
        Args:
            dataset_type: One of 'pums_recs', 'pums_atus', 'recs_atus'
            
        Returns:
            List of feature names to use for matching
        """
        base_features = [
            'hh_size_binary', 'hh_size_3cat', 'hh_size_5cat',
            'income_binary', 'income_tercile', 'income_quintile',
            'census_region', 'census_division', 'climate_zone',
            'is_single_person', 'is_large_household'
        ]
        
        if dataset_type == 'pums_recs':
            # Add energy-related features for PUMS-RECS matching
            base_features.extend([
                'building_era', 'is_new_building', 'is_old_building',
                'high_heating_need', 'high_cooling_need',
                'energy_intensity_proxy', 'energy_vulnerability',
                'housing_quality_cat', 'ses_category'
            ])
        elif dataset_type == 'pums_atus':
            # Add activity-related features for PUMS-ATUS matching
            base_features.extend([
                'work_from_home', 'childcare_likely',
                'retirement_activities', 'student_activities',
                'working_age_adult', 'retired_senior',
                'ses_category', 'income_size_interaction'
            ])
        elif dataset_type == 'recs_atus':
            # Features for potential RECS-ATUS matching
            base_features.extend([
                'ses_category', 'energy_vulnerability',
                'climate_zone', 'income_size_interaction'
            ])
        
        # Filter to only return features that were actually created
        return [f for f in base_features if f in self.created_features]
    
    def create_matching_weights(self, features: List[str]) -> Dict[str, float]:
        """
        Create importance weights for matching features.
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary of feature weights
        """
        weights = {}
        
        # Higher weights for more discriminative features
        high_weight = ['income_quintile', 'hh_size_5cat', 'census_division', 'building_era']
        medium_weight = ['income_tercile', 'hh_size_3cat', 'climate_zone', 'ses_category']
        
        for feature in features:
            if feature in high_weight:
                weights[feature] = 1.5
            elif feature in medium_weight:
                weights[feature] = 1.2
            else:
                weights[feature] = 1.0
        
        return weights


def enhance_dataset_for_matching(df: pd.DataFrame, 
                                dataset_type: str,
                                matching_target: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to enhance a dataset with cross-dataset features.
    
    Args:
        df: Input dataframe
        dataset_type: One of 'pums', 'recs', 'atus'
        matching_target: Optional target dataset for specific feature selection
        
    Returns:
        Enhanced dataframe with additional features
    """
    engineer = CrossDatasetFeatureEngineer()
    
    # Create universal features
    df_enhanced = engineer.create_universal_features(df, dataset_type)
    
    # Get appropriate matching features
    if matching_target:
        matching_type = f"{dataset_type}_{matching_target}"
        features = engineer.get_matching_features(matching_type)
        logger.info(f"Created {len(features)} features for {matching_type} matching")
    
    return df_enhanced