"""
Feature engineering utilities for PUMS Enrichment Pipeline.

This module provides functions to create derived features for improved
matching accuracy and energy modeling insights.
"""

import pandas as pd
import numpy as np
import jellyfish
from typing import Dict, List, Optional, Tuple
import logging

from .data_standardization import create_blocking_key
from .config_loader import get_config
from .dtype_utils import safe_cut, safe_qcut, ensure_string_dtype

logger = logging.getLogger(__name__)


def ensure_consistent_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent data types across the DataFrame to avoid downstream issues.
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        DataFrame with consistent data types
    """
    # Convert all categorical columns that were created with pd.cut to strings
    categorical_columns = df.select_dtypes(include=['category']).columns
    for col in categorical_columns:
        df[col] = df[col].astype(str)
    
    # Ensure integer columns are properly typed
    int_columns = ['NP', 'has_vehicle', 'multi_vehicle', 'has_children', 
                   'has_seniors', 'multigenerational', 'is_owner', 
                   'high_tech', 'is_employed', 'high_energy_burden']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    # Ensure float columns are properly typed
    float_columns = ['HINCP', 'YRBLT', 'BDSP', 'RMSP', 'VEH', 'ELEP', 
                     'GASP', 'FULP', 'energy_burden', 'total_energy_cost']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # Ensure string columns are properly typed
    string_columns = ['SERIALNO', 'STATE', 'PUMA', 'building_id']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


def create_household_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for household data.
    
    Args:
        df: Household DataFrame
        
    Returns:
        DataFrame with additional engineered features
        
    Raises:
        ValueError: If input DataFrame is invalid
    """
    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Check required columns
    required_cols = ['SERIALNO', 'NP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Creating household features")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Household size category with better NaN handling
    np_values = df['NP'].fillna(1).clip(lower=1)  # Ensure minimum value of 1
    df['household_size_cat'] = safe_cut(
        np_values,
        bins=[0, 1, 2, 4, 6, 100],
        labels=['single', 'couple', 'small_family', 'large_family', 'very_large'],
        include_lowest=True
    )  # safe_cut automatically returns string dtype
    
    # Income quintiles (handle missing values)
    if 'HINCP' in df.columns:
        # Replace negative income with 0
        income_clean = df['HINCP'].fillna(0)
        income_clean = income_clean.clip(lower=0)
        
        # Create quintiles only for positive incomes
        if (income_clean > 0).any():
            try:
                income_quintiles = pd.qcut(
                    income_clean[income_clean > 0],
                    q=5,
                    labels=['q1_lowest', 'q2_low', 'q3_medium', 'q4_high', 'q5_highest'],
                    duplicates='drop'
                ).astype(str)  # Convert to string to avoid categorical issues
                # Reindex to include all rows
                df['income_quintile'] = income_quintiles.reindex(df.index, fill_value='q1_lowest')
            except:
                # If quintiles can't be created (e.g., too few unique values)
                df['income_quintile'] = 'q3_medium'
        else:
            df['income_quintile'] = 'q1_lowest'
    else:
        df['income_quintile'] = 'unknown'
    
    # Building age category with better NaN handling
    if 'YRBLT' in df.columns:
        config = get_config()
        current_year = config.get('processing.data_year', 2023)  # Default to 2023 if not set
        # Handle missing YRBLT values
        valid_yrblt = df['YRBLT'].fillna(1980)  # Use median year as default
        df['building_age'] = current_year - valid_yrblt
        df['building_age_cat'] = pd.cut(
            df['building_age'].fillna(40),  # Default to moderate age
            bins=[-np.inf, 10, 30, 50, 100, np.inf],
            labels=['new', 'recent', 'moderate', 'old', 'very_old']
        ).astype(str)
    else:
        df['building_age_cat'] = 'unknown'
    
    # Building type from BLD (units in structure)
    if 'BLD' in df.columns:
        df['building_type'] = df['BLD'].map({
            1: 'mobile_home',
            2: 'single_family_detached',
            3: 'single_family_attached',
            4: 'duplex',
            5: 'small_multi_3_4',
            6: 'small_multi_5_9', 
            7: 'medium_multi_10_19',
            8: 'medium_multi_20_49',
            9: 'large_multi_50_plus',
            10: 'other'
        }).fillna('unknown')
        
        # Simplified building type for matching
        df['building_type_simple'] = df['building_type'].map({
            'mobile_home': 'mobile',
            'single_family_detached': 'single_family',
            'single_family_attached': 'single_family',
            'duplex': 'small_multi',
            'small_multi_3_4': 'small_multi',
            'small_multi_5_9': 'small_multi',
            'medium_multi_10_19': 'large_multi',
            'medium_multi_20_49': 'large_multi',
            'large_multi_50_plus': 'large_multi',
            'other': 'other',
            'unknown': 'unknown'
        })
    else:
        df['building_type'] = 'unknown'
        df['building_type_simple'] = 'unknown'
    
    # Urban/Rural indicator (simplified from PUMA characteristics)
    # This is a placeholder - in production would use PUMA-to-urban mapping
    df['urban_rural'] = 'urban'  # Default, would need external data for accurate classification
    
    # Geographic blocking keys
    df['geo_block_state_income'] = (
        df['STATE'].astype(str) + '_' + 
        df['income_quintile'].astype(str)
    )
    
    df['geo_block_state_size'] = (
        df['STATE'].astype(str) + '_' + 
        df['household_size_cat'].astype(str)
    )
    
    df['geo_block_puma_type'] = (
        df['PUMA'].astype(str) + '_' + 
        df['building_type_simple'].astype(str)
    )
    
    # Energy-relevant features
    
    # Number of bedrooms category with NaN handling
    if 'BDSP' in df.columns:
        bdsp_values = df['BDSP'].fillna(1).clip(lower=0)  # Default to 1 bedroom
        df['bedroom_cat'] = pd.cut(
            bdsp_values,
            bins=[-np.inf, 0, 1, 2, 3, 4, np.inf],
            labels=['studio', 'one_br', 'two_br', 'three_br', 'four_br', 'five_plus_br']
        ).astype(str)
    else:
        df['bedroom_cat'] = 'unknown'
    
    # Number of rooms category with NaN handling
    if 'RMSP' in df.columns:
        rmsp_values = df['RMSP'].fillna(4).clip(lower=1)  # Default to 4 rooms
        df['room_cat'] = pd.cut(
            rmsp_values,
            bins=[-np.inf, 3, 5, 7, 9, np.inf],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        ).astype(str)
    else:
        df['room_cat'] = 'unknown'
    
    # Vehicle ownership with NaN handling
    if 'VEH' in df.columns:
        veh_values = df['VEH'].fillna(0).clip(lower=0)  # Default to no vehicle
        df['has_vehicle'] = (veh_values > 0).astype(int)
        df['multi_vehicle'] = (veh_values > 1).astype(int)
        df['vehicle_cat'] = pd.cut(
            veh_values,
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=['no_vehicle', 'one_vehicle', 'two_vehicles', 'three_plus_vehicles']
        ).astype(str)
    else:
        df['vehicle_cat'] = 'unknown'
    
    # Technology adoption indicators
    tech_indicators = ['BROADBND', 'HISPEED', 'LAPTOP', 'SMARTPHONE', 'TABLET']
    available_tech = [col for col in tech_indicators if col in df.columns]
    if available_tech:
        df['tech_score'] = df[available_tech].sum(axis=1)
        df['high_tech'] = (df['tech_score'] >= len(available_tech) * 0.6).astype(int)
    else:
        df['tech_score'] = 0
        df['high_tech'] = 0
    
    # Appliance indicators
    appliance_indicators = ['REFR', 'STOV', 'HOTWAT']
    available_appliances = [col for col in appliance_indicators if col in df.columns]
    if available_appliances:
        df['appliance_score'] = df[available_appliances].sum(axis=1)
    else:
        df['appliance_score'] = 0
    
    # Energy cost burden (if available)
    if all(col in df.columns for col in ['ELEP', 'GASP', 'FULP', 'HINCP']):
        # Calculate total monthly energy cost
        df['total_energy_cost'] = (
            df['ELEP'].fillna(0) + 
            df['GASP'].fillna(0) + 
            (df['FULP'].fillna(0) / 12)  # Convert yearly to monthly
        )
        
        # Energy burden as percentage of income
        monthly_income = df['HINCP'] / 12
        df['energy_burden'] = np.where(
            monthly_income > 0,
            (df['total_energy_cost'] / monthly_income) * 100,
            np.nan
        )
        
        df['high_energy_burden'] = (df['energy_burden'] > 6).astype(int)
    else:
        df['energy_burden'] = np.nan
        df['high_energy_burden'] = 0
    
    # Household composition features
    df['has_children'] = df['R18'].fillna(0).astype(int)
    df['has_seniors'] = df['R65'].fillna(0).astype(int)
    df['multigenerational'] = df['MULTG'].fillna(0).astype(int)
    
    # Create household type based on composition
    df['household_composition'] = 'other'
    df.loc[(df['NP'] == 1) & (df['has_seniors'] == 0), 'household_composition'] = 'single_adult'
    df.loc[(df['NP'] == 1) & (df['has_seniors'] == 1), 'household_composition'] = 'single_senior'
    df.loc[(df['NP'] == 2) & (df['has_children'] == 0) & (df['has_seniors'] == 0), 'household_composition'] = 'couple_no_children'
    df.loc[(df['NP'] >= 2) & (df['has_children'] == 1), 'household_composition'] = 'family_with_children'
    df.loc[df['multigenerational'] == 1, 'household_composition'] = 'multigenerational'
    
    # Heating fuel type category
    if 'HFL' in df.columns:
        df['heating_fuel'] = df['HFL'].map({
            1: 'gas',
            2: 'electricity',
            3: 'fuel_oil',
            4: 'coal',
            5: 'wood',
            6: 'solar',
            7: 'other',
            8: 'none'
        }).fillna('unknown')
    else:
        df['heating_fuel'] = 'unknown'
    
    # Tenure type
    if 'TEN' in df.columns:
        df['tenure_type'] = df['TEN'].map({
            1: 'owned_with_mortgage',
            2: 'owned_free_clear',
            3: 'rented',
            4: 'occupied_no_rent'
        }).fillna('unknown')
        
        df['is_owner'] = df['TEN'].isin([1, 2]).astype(int)
    else:
        df['tenure_type'] = 'unknown'
        df['is_owner'] = 0
    
    logger.info(f"Created {len(df.columns)} household features")
    
    # Ensure consistent data types
    df = ensure_consistent_dtypes(df)
    
    return df


def create_person_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for person data.
    
    Args:
        df: Person DataFrame
        
    Returns:
        DataFrame with additional engineered features
        
    Raises:
        ValueError: If input DataFrame is invalid
    """
    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Check required columns
    required_cols = ['SERIALNO', 'AGEP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Creating person features")
    
    # Create a copy
    df = df.copy()
    
    # Age groups with NaN handling and consistent data types
    age_values = df['AGEP'].fillna(35).clip(lower=0, upper=120)  # Default to adult age
    df['age_group'] = pd.cut(
        age_values,
        bins=[0, 5, 12, 17, 25, 34, 44, 54, 64, 74, 120],
        labels=['infant_toddler', 'child', 'teen', 'young_adult', 'adult', 
                'middle_age', 'older_adult', 'young_senior', 'senior', 'elderly'],
        include_lowest=True
    ).astype(str)  # Ensure string type
    
    # Simplified age groups for matching
    df['age_group_simple'] = pd.cut(
        age_values,
        bins=[0, 18, 35, 50, 65, 120],
        labels=['child', 'young_adult', 'adult', 'middle_age', 'senior'],
        include_lowest=True
    ).astype(str)  # Ensure string type
    
    # Employment status
    if 'ESR' in df.columns:
        df['employment_status'] = df['ESR'].map({
            1: 'employed',
            2: 'employed_not_at_work',
            3: 'unemployed',
            4: 'armed_forces',
            5: 'armed_forces_not_at_work',
            6: 'not_in_labor_force'
        }).fillna('unknown')
        
        df['is_employed'] = df['ESR'].isin([1, 2, 4, 5]).astype(int)
    else:
        df['employment_status'] = 'unknown'
        df['is_employed'] = 0
    
    # Education level
    if 'SCHL' in df.columns:
        def map_education(schl):
            if pd.isna(schl):
                return 'unknown'
            elif schl <= 11:
                return 'less_than_hs'
            elif schl <= 15:
                return 'some_hs'
            elif schl <= 19:
                return 'hs_grad'
            elif schl == 20:
                return 'some_college'
            elif schl == 21:
                return 'bachelors'
            elif schl <= 24:
                return 'graduate'
            else:
                return 'graduate'
                
        df['education_level'] = df['SCHL'].apply(map_education)
    else:
        df['education_level'] = 'unknown'
    
    # Marital status
    if 'MAR' in df.columns:
        df['marital_status'] = df['MAR'].map({
            1: 'married',
            2: 'widowed',
            3: 'divorced',
            4: 'separated',
            5: 'never_married'
        }).fillna('unknown')
        
        df['is_married'] = (df['MAR'] == 1).astype(int)
    else:
        df['marital_status'] = 'unknown'
        df['is_married'] = 0
    
    # Work from home indicator
    if 'JWTRNS' in df.columns:
        df['works_from_home'] = (df['JWTRNS'] == 80).astype(int)  # 80 = worked at home
        
        # Commute method
        df['commute_method'] = 'unknown'
        df.loc[df['JWTRNS'] == 1, 'commute_method'] = 'car_alone'
        df.loc[df['JWTRNS'].isin([2, 3, 4, 5, 6, 7]), 'commute_method'] = 'carpool'
        df.loc[df['JWTRNS'].isin([8, 9, 10, 11, 12]), 'commute_method'] = 'public_transit'
        df.loc[df['JWTRNS'] == 13, 'commute_method'] = 'walk'
        df.loc[df['JWTRNS'] == 14, 'commute_method'] = 'bike'
        df.loc[df['JWTRNS'] == 80, 'commute_method'] = 'work_from_home'
    else:
        df['works_from_home'] = 0
        df['commute_method'] = 'unknown'
    
    # Income level
    if 'PINCP' in df.columns:
        # Handle negative income
        income_clean = df['PINCP'].fillna(0).clip(lower=0)
        
        def map_income_level(income):
            if income <= 15000:
                return 'very_low'
            elif income <= 30000:
                return 'low'
            elif income <= 50000:
                return 'moderate'
            elif income <= 75000:
                return 'middle'
            elif income <= 100000:
                return 'high'
            else:
                return 'very_high'
                
        df['income_level'] = income_clean.apply(map_income_level)
    else:
        df['income_level'] = 'unknown'
    
    # Create person type for activity matching
    df['person_type'] = 'other'
    
    # Children
    df.loc[df['AGEP'] < 5, 'person_type'] = 'preschool_child'
    df.loc[(df['AGEP'] >= 5) & (df['AGEP'] <= 17), 'person_type'] = 'school_age_child'
    
    # Working adults
    df.loc[(df['AGEP'] >= 18) & (df['is_employed'] == 1), 'person_type'] = 'working_adult'
    df.loc[(df['AGEP'] >= 18) & (df['is_employed'] == 1) & 
           (df['works_from_home'] == 1), 'person_type'] = 'work_from_home_adult'
    
    # Non-working adults
    df.loc[(df['AGEP'] >= 18) & (df['AGEP'] < 65) & 
           (df['is_employed'] == 0), 'person_type'] = 'non_working_adult'
    
    # Seniors
    df.loc[(df['AGEP'] >= 65) & (df['is_employed'] == 1), 'person_type'] = 'working_senior'
    df.loc[(df['AGEP'] >= 65) & (df['is_employed'] == 0), 'person_type'] = 'retired_senior'
    
    # Sex
    if 'SEX' in df.columns:
        df['sex'] = df['SEX'].map({1: 'male', 2: 'female'}).fillna('unknown')
    else:
        df['sex'] = 'unknown'
    
    # Create blocking keys for person matching
    df['person_block_age_emp'] = (
        df['age_group_simple'].astype(str) + '_' + 
        df['employment_status'].astype(str)
    )
    
    df['person_block_type'] = df['person_type']
    
    # Phonetic codes for names (placeholder for future use)
    # In real implementation, these would be based on actual names
    df['first_name_soundex'] = ''
    df['last_name_soundex'] = ''
    df['first_name_nysiis'] = ''
    df['last_name_nysiis'] = ''
    
    logger.info(f"Created {len(df.columns)} person features")
    
    # Ensure consistent data types
    df = ensure_consistent_dtypes(df)
    
    return df


def create_matching_features(households: pd.DataFrame, persons: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create features specifically for matching in Phase 2-3.
    
    Args:
        households: Household DataFrame
        persons: Person DataFrame
        
    Returns:
        Tuple of (households, persons) with matching features
        
    Raises:
        ValueError: If input DataFrames are invalid
    """
    # Input validation
    if households is None or households.empty:
        raise ValueError("Households DataFrame is None or empty")
    
    if persons is None or persons.empty:
        raise ValueError("Persons DataFrame is None or empty")
    
    if not isinstance(households, pd.DataFrame):
        raise ValueError(f"Expected households to be DataFrame, got {type(households)}")
    
    if not isinstance(persons, pd.DataFrame):
        raise ValueError(f"Expected persons to be DataFrame, got {type(persons)}")
    logger.info("Creating matching features")
    
    # Aggregate person features to household level
    person_agg = persons.groupby('SERIALNO').agg({
        'AGEP': ['mean', 'min', 'max'],
        'is_employed': 'sum',
        'works_from_home': 'sum',
        'person_id': 'count'
    })
    
    person_agg.columns = ['avg_age', 'min_age', 'max_age', 
                         'num_employed', 'num_work_from_home', 'person_count']
    
    # Add person type counts
    person_type_counts = persons.groupby(['SERIALNO', 'person_type'], observed=True).size().unstack(fill_value=0)
    person_agg = person_agg.join(person_type_counts, how='left')
    
    # Join to households
    households = households.join(person_agg, on='SERIALNO', how='left')
    
    # Fill missing values
    for col in person_agg.columns:
        if col in households.columns:
            households[col] = households[col].fillna(0)
    
    # Create composite matching keys
    households['match_key_1'] = (
        households['STATE'].astype(str) + '_' +
        households['household_size_cat'].astype(str) + '_' +
        households['income_quintile'].astype(str)
    )
    
    households['match_key_2'] = (
        households['building_type_simple'].astype(str) + '_' +
        households['household_composition'].astype(str)
    )
    
    # Create numeric features for similarity calculation
    numeric_features = [
        'NP', 'avg_age', 'num_employed', 'HINCP', 'BDSP', 'RMSP', 'VEH'
    ]
    
    available_numeric = [col for col in numeric_features if col in households.columns]
    
    # Normalize numeric features
    for col in available_numeric:
        if households[col].std() > 0:
            households[f'{col}_normalized'] = (
                (households[col] - households[col].mean()) / households[col].std()
            )
        else:
            households[f'{col}_normalized'] = 0
    
    logger.info("Matching features created")
    
    return households, persons


def create_energy_profile_features(households: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that predict energy consumption patterns.
    
    Args:
        households: Household DataFrame with engineered features
        
    Returns:
        DataFrame with energy profile features
        
    Raises:
        ValueError: If input DataFrame is invalid
    """
    # Input validation
    if households is None or households.empty:
        raise ValueError("Households DataFrame is None or empty")
    
    if not isinstance(households, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(households)}")
    
    logger.info("Creating energy profile features")
    
    df = households.copy()
    
    # Base load indicator (always-on consumption)
    df['base_load_score'] = 0
    
    # More people = higher base load
    if 'NP' in df.columns:
        df['base_load_score'] += df['NP'] * 0.2
    
    # Technology adds to base load
    if 'tech_score' in df.columns:
        df['base_load_score'] += df['tech_score'] * 0.1
    
    # HVAC load indicator
    df['hvac_load_score'] = 0
    
    # Larger homes need more HVAC
    if 'RMSP' in df.columns:
        df['hvac_load_score'] += df['RMSP'] * 0.1
    
    # Older homes are less efficient
    if 'building_age' in df.columns:
        df['hvac_load_score'] += (df['building_age'] / 50).clip(upper=2)
    
    # Presence of seniors affects thermostat settings
    if 'has_seniors' in df.columns:
        df['hvac_load_score'] += df['has_seniors'] * 0.3
    
    # Peak load indicator
    df['peak_load_score'] = 0
    
    # Employed people create morning/evening peaks
    if 'num_employed' in df.columns:
        df['peak_load_score'] += df['num_employed'] * 0.3
    
    # Children create afternoon peaks
    if 'has_children' in df.columns:
        df['peak_load_score'] += df['has_children'] * 0.2
    
    # Work from home reduces peaks
    if 'num_work_from_home' in df.columns:
        df['peak_load_score'] -= df['num_work_from_home'] * 0.1
    
    # Normalize scores
    for score_col in ['base_load_score', 'hvac_load_score', 'peak_load_score']:
        if score_col in df.columns:
            df[score_col] = df[score_col].clip(lower=0)
            if df[score_col].max() > 0:
                df[score_col] = df[score_col] / df[score_col].max()
    
    # Create overall energy intensity category
    df['energy_intensity'] = (
        df['base_load_score'] * 0.3 +
        df['hvac_load_score'] * 0.5 +
        df['peak_load_score'] * 0.2
    )
    
    try:
        df['energy_intensity_cat'] = pd.qcut(
            df['energy_intensity'],
            q=5,
            labels=['very_low', 'low', 'moderate', 'high', 'very_high'],
            duplicates='drop'
        ).astype(str)
    except:
        # If can't create quintiles, use simple binning
        df['energy_intensity_cat'] = pd.cut(
            df['energy_intensity'],
            bins=5,
            labels=['very_low', 'low', 'moderate', 'high', 'very_high']
        ).astype(str)
    
    logger.info("Energy profile features created")
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Create sample data
    sample_households = pd.DataFrame({
        'SERIALNO': ['001', '002', '003'],
        'STATE': ['01', '01', '02'],
        'PUMA': ['001', '001', '002'],
        'NP': [1, 4, 2],
        'HINCP': [50000, 75000, 30000],
        'YRBLT': [2010, 1980, 2000],
        'BLD': [2, 2, 4],
        'BDSP': [1, 3, 2],
        'RMSP': [4, 8, 5],
        'VEH': [1, 2, 0],
        'R18': [0, 1, 0],
        'R65': [0, 0, 1],
        'MULTG': [0, 0, 0],
        'HFL': [1, 2, 1],
        'TEN': [3, 1, 3],
        'BROADBND': [1, 1, 0],
        'LAPTOP': [1, 1, 0],
        'SMARTPHONE': [1, 1, 1]
    })
    
    sample_persons = pd.DataFrame({
        'SERIALNO': ['001', '002', '002', '002', '002', '003', '003'],
        'SPORDER': [1, 1, 2, 3, 4, 1, 2],
        'AGEP': [35, 40, 38, 10, 8, 70, 68],
        'SEX': [1, 1, 2, 1, 2, 1, 2],
        'ESR': [1, 1, 1, np.nan, np.nan, 6, 6],
        'MAR': [5, 1, 1, 5, 5, 1, 1],
        'SCHL': [21, 22, 20, 8, 6, 19, 18],
        'JWTRNS': [1, 1, 80, np.nan, np.nan, np.nan, np.nan],
        'PINCP': [50000, 60000, 40000, 0, 0, 20000, 18000]
    })
    
    # Test household features
    print("Testing household features...")
    households_with_features = create_household_features(sample_households)
    print(f"Created features: {[col for col in households_with_features.columns if col not in sample_households.columns]}")
    
    # Test person features
    print("\nTesting person features...")
    persons_with_features = create_person_features(sample_persons)
    print(f"Created features: {[col for col in persons_with_features.columns if col not in sample_persons.columns]}")
    
    # Test matching features
    print("\nTesting matching features...")
    households_match, persons_match = create_matching_features(
        households_with_features, persons_with_features
    )
    print(f"Household matching features: {[col for col in households_match.columns if 'match' in col or 'normalized' in col]}")
    
    # Test energy profile
    print("\nTesting energy profile features...")
    households_energy = create_energy_profile_features(households_match)
    print(f"Energy features: {[col for col in households_energy.columns if 'energy' in col or 'load' in col]}")
    
    print("\nFeature engineering tests completed!")