"""
RECS data loader for PUMS Enrichment Pipeline.

This module provides functions to load and prepare RECS (Residential Energy
Consumption Survey) data for probabilistic matching with PUMS buildings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

from ..utils.config_loader import get_config
from ..utils.data_standardization import standardize_field
from ..utils.logging_setup import log_execution_time, log_memory_usage
from ..utils.enhanced_feature_engineering import create_comprehensive_matching_features

logger = logging.getLogger(__name__)


class RECSDataError(Exception):
    """Raised when there's an issue with RECS data loading."""
    pass


@log_execution_time(logger)
@log_memory_usage(logger)
def load_recs_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and clean RECS microdata.
    
    Args:
        sample_size: Number of RECS records to load (None for all)
        
    Returns:
        DataFrame with cleaned RECS data
        
    Raises:
        RECSDataError: If data cannot be loaded
    """
    config = get_config()
    recs_file = config.get_data_path('recs_data')
    
    logger.info(f"Loading RECS data from: {recs_file}")
    
    # Check if file exists
    if not Path(recs_file).exists():
        raise RECSDataError(f"RECS data file not found: {recs_file}")
    
    try:
        # Define columns to keep (key RECS variables for matching and building characteristics)
        recs_columns = [
            'DOEID',  # Unique identifier
            'REGIONC', 'DIVISION', 'STATE_FIPS', 'UATYP10',  # Geographic
            'TYPEHUQ', 'YEARMADERANGE', 'SQFTEST', 'TOTROOMS', 'BEDROOMS',  # Building + BEDROOMS added
            'NHSLDMEM', 'MONEYPY',  # Household (NHSLDMEM is household size)
            'NUMCHILD', 'NUMADULT65', 'KOWNRENT',  # Household composition + tenure
            'FUELHEAT', 'EQUIPM', 'AIRCOND', 'FUELH2O',  # Equipment
            'KWH', 'BTUEL', 'BTUNG', 'BTULP', 'BTUFO',  # Energy
            'DOLLAREL', 'DOLLARNG', 'DOLLARLP', 'DOLLARFO',  # Energy costs
            'NWEIGHT'  # Survey weight
        ]
        
        # Read RECS data
        df = pd.read_csv(recs_file, usecols=lambda x: x in recs_columns, low_memory=False)
        
        # Apply sample size if specified
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} RECS records")
            df = df.sample(n=sample_size, random_state=config.get_random_seed())
        
        logger.info(f"Loaded {len(df)} RECS records")
        
        # Data cleaning and standardization
        df = clean_recs_data(df)
        
        # Create template ID (use recs_id after renaming)
        df['template_id'] = 'RECS_' + df['recs_id'].astype(str)
        
        # Log summary statistics
        logger.info("RECS data summary:")
        logger.info(f"  Total templates: {len(df)}")
        logger.info(f"  States represented: {df['STATE'].nunique()}")
        logger.info(f"  Average household size: {df['household_size'].mean():.2f}")
        logger.info(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        raise RECSDataError(f"Error loading RECS data: {str(e)}")


def clean_recs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize RECS data for matching.
    
    Args:
        df: Raw RECS DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning RECS data")
    
    # Create a copy
    df = df.copy()
    
    # Rename columns to match PUMS naming conventions
    column_mapping = {
        'DOEID': 'recs_id',
        'STATE_FIPS': 'STATE',
        'REGIONC': 'REGION',
        'DIVISION': 'DIVISION',
        'UATYP10': 'urban_rural_raw',
        'TYPEHUQ': 'building_type_raw',
        'YEARMADERANGE': 'year_built_range',
        'SQFTEST': 'square_footage',  # Changed from TOTSQFT_EN
        'TOTROOMS': 'total_rooms',
        'BEDROOMS': 'num_bedrooms',  # Added bedroom mapping
        'NHSLDMEM': 'household_size',  # This is the household size in RECS
        'MONEYPY': 'household_income',
        'NUMCHILD': 'num_children',  # Number of children
        'NUMADULT65': 'num_seniors',  # Number of adults 65+
        'KOWNRENT': 'tenure_raw',  # Own/rent status
        'FUELHEAT': 'heating_fuel_raw',
        'EQUIPM': 'heating_equipment',
        'AIRCOND': 'has_air_conditioning',  # Changed from COOLTYPE
        'FUELH2O': 'water_heater_fuel',
        'KWH': 'electricity_kwh',
        'BTUEL': 'electricity_btu',
        'BTUNG': 'natural_gas_btu',
        'BTULP': 'propane_btu',
        'BTUFO': 'fuel_oil_btu',
        'DOLLAREL': 'electricity_cost',
        'DOLLARNG': 'natural_gas_cost',
        'DOLLARLP': 'propane_cost',
        'DOLLARFO': 'fuel_oil_cost',
        'NWEIGHT': 'survey_weight'
    }
    
    # Rename available columns
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Ensure STATE is string with leading zeros
    if 'STATE' in df.columns:
        df['STATE'] = df['STATE'].astype(str).str.zfill(2)
    else:
        df['STATE'] = 'XX'  # Unknown state
    
    # Handle missing values
    df = handle_recs_missing_values(df)
    
    # Create derived features
    df = create_recs_features(df)
    
    # Apply comprehensive feature engineering
    df = create_comprehensive_matching_features(df, dataset_type='recs')
    
    return df


def handle_recs_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in RECS data with appropriate defaults.
    
    Args:
        df: RECS DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    # Numeric columns - fill with 0 or median
    numeric_fills = {
        'square_footage': df['square_footage'].median() if 'square_footage' in df.columns else 1500,
        'total_rooms': df['total_rooms'].median() if 'total_rooms' in df.columns else 6,
        'num_bedrooms': df['num_bedrooms'].median() if 'num_bedrooms' in df.columns else 3,
        'household_size': 1,
        'household_income': df['household_income'].median() if 'household_income' in df.columns else 50000,
        'num_children': 0,
        'num_seniors': 0,
        'electricity_kwh': 0,
        'electricity_cost': 0,
        'natural_gas_cost': 0,
        'propane_cost': 0,
        'fuel_oil_cost': 0
    }
    
    for col, fill_value in numeric_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    
    # Categorical columns - fill with mode or 'unknown'
    categorical_fills = {
        'urban_rural_raw': 'U',  # Default to urban
        'building_type_raw': 2,  # Single-family detached
        'heating_fuel_raw': 1,  # Natural gas
        'has_air_conditioning': 1,  # Has AC
        'tenure_raw': 1  # Default to owned
    }
    
    for col, fill_value in categorical_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    
    return df


def create_recs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from RECS data that align with PUMS features.
    
    Args:
        df: RECS DataFrame
        
    Returns:
        DataFrame with additional features
    """
    logger.info("Creating RECS features for matching")
    
    # Income quintiles (matching PUMS)
    if 'household_income' in df.columns:
        # Handle negative incomes
        income_clean = df['household_income'].clip(lower=0)
        
        # Create quintiles
        try:
            from ..utils.dtype_utils import safe_qcut
            df['income_quintile'] = safe_qcut(
                income_clean[income_clean > 0],
                q=5,
                labels=['q1_lowest', 'q2_low', 'q3_medium', 'q4_high', 'q5_highest']
            ).reindex(df.index, fill_value='q1_lowest')
        except:
            df['income_quintile'] = 'q3_medium'
    else:
        df['income_quintile'] = 'unknown'
    
    # Household size category (matching PUMS)
    if 'household_size' in df.columns:
        from ..utils.dtype_utils import safe_cut
        df['household_size_cat'] = safe_cut(
            df['household_size'],
            bins=[0, 1, 2, 4, 6, 100],
            labels=['single', 'couple', 'small_family', 'large_family', 'very_large']
        )
    else:
        df['household_size_cat'] = 'unknown'
    
    # Building type (simplified to match PUMS)
    if 'building_type_raw' in df.columns:
        building_type_map = {
            1: 'mobile',  # Mobile home
            2: 'single_family',  # Single-family detached
            3: 'single_family',  # Single-family attached
            4: 'small_multi',  # Apartment 2-4 units
            5: 'large_multi'  # Apartment 5+ units
        }
        df['building_type_simple'] = df['building_type_raw'].map(building_type_map).fillna('unknown')
    else:
        df['building_type_simple'] = 'unknown'
    
    # Urban/rural indicator
    if 'urban_rural_raw' in df.columns:
        # RECS UATYP10: U=Urban Area, C=Urban Cluster, R=Rural
        # First ensure it's a string
        df['urban_rural_raw'] = df['urban_rural_raw'].astype(str).str.upper()
        urban_rural_map = {
            'U': 'urban',
            'C': 'urban',  # Urban cluster is still urban
            'R': 'rural',
            '1': 'urban',  # Sometimes coded as numbers
            '2': 'urban',
            '3': 'rural'
        }
        df['urban_rural'] = df['urban_rural_raw'].map(urban_rural_map).fillna('urban')
        
        # Also create binary indicators matching PUMS
        df['is_urban'] = (df['urban_rural'] == 'urban').astype(int)
        df['is_rural'] = (df['urban_rural'] == 'rural').astype(int)
    else:
        df['urban_rural'] = 'urban'
        df['is_urban'] = 1
        df['is_rural'] = 0
    
    # Year built to building age (approximate)
    if 'year_built_range' in df.columns:
        # RECS year ranges: 1=Before 1950, 2=1950-1959, ..., 8=2010-2015, 9=2016-2020
        year_map = {
            1: 1940,  # Before 1950
            2: 1955,  # 1950-1959
            3: 1965,  # 1960-1969
            4: 1975,  # 1970-1979
            5: 1985,  # 1980-1989
            6: 1995,  # 1990-1999
            7: 2005,  # 2000-2009
            8: 2013,  # 2010-2015
            9: 2018   # 2016-2020
        }
        config = get_config()
        current_year = config.get('processing.data_year', 2023)
        df['year_built_est'] = df['year_built_range'].map(year_map).fillna(1980)
        df['building_age'] = current_year - df['year_built_est']
        
        from ..utils.dtype_utils import safe_cut
        df['building_age_cat'] = safe_cut(
            df['building_age'],
            bins=[-np.inf, 10, 30, 50, 100, np.inf],
            labels=['new', 'recent', 'moderate', 'old', 'very_old']
        )
    else:
        df['building_age_cat'] = 'unknown'
    
    # Heating fuel (simplified)
    if 'heating_fuel_raw' in df.columns:
        fuel_map = {
            1: 'gas',  # Natural gas
            2: 'gas',  # Propane/LPG
            3: 'fuel_oil',  # Fuel oil
            4: 'electricity',
            5: 'electricity',  # Heat pump
            7: 'wood',
            8: 'solar',
            9: 'other',
            21: 'other',  # Other
            -2: 'none'  # Not applicable
        }
        df['heating_fuel'] = df['heating_fuel_raw'].map(fuel_map).fillna('unknown')
    else:
        df['heating_fuel'] = 'unknown'
    
    # Room categories
    if 'total_rooms' in df.columns:
        from ..utils.dtype_utils import safe_cut
        df['room_cat'] = safe_cut(
            df['total_rooms'],
            bins=[-np.inf, 3, 5, 7, 9, np.inf],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
    else:
        df['room_cat'] = 'unknown'
    
    # Energy intensity (if we have consumption data)
    if all(col in df.columns for col in ['electricity_btu', 'natural_gas_btu', 'propane_btu', 
                                         'fuel_oil_btu', 'square_footage']):
        # Total energy consumption in BTU (sum all available fuel types)
        df['total_energy_btu'] = (
            df['electricity_btu'].fillna(0) +
            df['natural_gas_btu'].fillna(0) +
            df['propane_btu'].fillna(0) +
            df['fuel_oil_btu'].fillna(0)
        )
        
        # Energy use intensity (BTU per square foot)
        df['energy_use_intensity'] = np.where(
            df['square_footage'] > 0,
            df['total_energy_btu'] / df['square_footage'],
            0
        )
    
    # Total energy cost
    cost_columns = ['electricity_cost', 'natural_gas_cost', 'propane_cost', 'fuel_oil_cost']
    available_cost_cols = [col for col in cost_columns if col in df.columns]
    if available_cost_cols:
        df['total_energy_cost_annual'] = df[available_cost_cols].fillna(0).sum(axis=1)
        df['total_energy_cost_monthly'] = df['total_energy_cost_annual'] / 12
    
    # Number of people (for matching with PUMS NP)
    if 'household_size' in df.columns:
        df['NP'] = df['household_size'].astype(int)
    
    # Household composition features
    if 'num_children' in df.columns:
        df['has_children'] = (df['num_children'] > 0).astype(int)
        df['R18'] = df['num_children']  # Match PUMS R18 field
    else:
        df['has_children'] = 0
        df['R18'] = 0
    
    if 'num_seniors' in df.columns:
        df['has_seniors'] = (df['num_seniors'] > 0).astype(int)
        df['R65'] = df['num_seniors']  # Match PUMS R65 field
    else:
        df['has_seniors'] = 0
        df['R65'] = 0
    
    # Tenure (ownership) features
    if 'tenure_raw' in df.columns:
        # RECS: 1=Owned, 2=Rented, 3=Occupied without payment
        df['is_owner'] = (df['tenure_raw'] == 1).astype(int)
        df['is_renter'] = (df['tenure_raw'] == 2).astype(int)
        # Map to match PUMS tenure_type values
        tenure_map = {
            1: 'owned_mortgage',  # Assuming most owned homes have mortgages
            2: 'rented',
            3: 'occupied_no_rent'
        }
        df['tenure_type'] = df['tenure_raw'].map(tenure_map).fillna('owned_mortgage')
    else:
        df['is_owner'] = 1  # Default to owned
        df['is_renter'] = 0
        df['tenure_type'] = 'owned_mortgage'
    
    # Create matching keys (same as PUMS)
    df['match_key_1'] = (
        df['STATE'].astype(str) + '_' +
        df['household_size_cat'].astype(str) + '_' +
        df['income_quintile'].astype(str)
    )
    
    df['match_key_2'] = (
        df['building_type_simple'].astype(str) + '_' +
        df['urban_rural'].astype(str)
    )
    
    return df


def prepare_recs_for_matching(df: pd.DataFrame, pums_columns: List[str]) -> pd.DataFrame:
    """
    Prepare RECS data to have compatible columns with PUMS for matching.
    
    Args:
        df: RECS DataFrame
        pums_columns: List of columns from PUMS data
        
    Returns:
        RECS DataFrame with columns aligned for matching
    """
    # Identify common columns
    common_columns = [col for col in pums_columns if col in df.columns]
    
    # Add placeholders for PUMS columns that don't exist in RECS
    for col in pums_columns:
        if col not in df.columns:
            # Add appropriate default based on column type
            if 'normalized' in col:
                df[col] = 0.0
            elif col in ['has_vehicle', 'high_tech', 'is_owner']:
                df[col] = 0
            elif col in ['vehicle_cat', 'tech_score']:
                df[col] = 'unknown'
            # Skip person-specific columns as RECS doesn't have individual person data
    
    return df


def create_recs_summary_report(df: pd.DataFrame) -> Dict[str, any]:
    """
    Create summary statistics for RECS data.
    
    Args:
        df: RECS DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'states_covered': df['STATE'].nunique() if 'STATE' in df.columns else 0,
        'building_types': df['building_type_simple'].value_counts().to_dict() if 'building_type_simple' in df.columns else {},
        'household_sizes': df['household_size_cat'].value_counts().to_dict() if 'household_size_cat' in df.columns else {},
        'income_distribution': df['income_quintile'].value_counts().to_dict() if 'income_quintile' in df.columns else {},
        'heating_fuels': df['heating_fuel'].value_counts().to_dict() if 'heating_fuel' in df.columns else {},
        'avg_square_footage': df['square_footage'].mean() if 'square_footage' in df.columns else None,
        'avg_energy_cost_monthly': df['total_energy_cost_monthly'].mean() if 'total_energy_cost_monthly' in df.columns else None
    }
    
    return summary


def calculate_recs_weights(df: pd.DataFrame) -> pd.Series:
    """
    Calculate normalized weights for RECS templates.
    
    Args:
        df: RECS DataFrame with survey weights
        
    Returns:
        Series of normalized weights
    """
    if 'survey_weight' in df.columns:
        # Normalize weights to sum to 1
        weights = df['survey_weight'].fillna(1.0)
        return weights / weights.sum()
    else:
        # Equal weights if no survey weights available
        return pd.Series(1.0 / len(df), index=df.index)


if __name__ == "__main__":
    # Test RECS data loading
    try:
        logger.info("Testing RECS data loader")
        
        # Load sample of RECS data
        recs_data = load_recs_data(sample_size=100)
        
        print(f"\nLoaded {len(recs_data)} RECS records")
        print(f"Columns: {len(recs_data.columns)}")
        print(f"\nColumn names: {list(recs_data.columns[:20])}...")  # First 20 columns
        
        # Show summary
        summary = create_recs_summary_report(recs_data)
        print(f"\nRECS Data Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        # Check matching features
        matching_features = ['STATE', 'income_quintile', 'household_size_cat', 
                           'building_type_simple', 'urban_rural']
        print(f"\nMatching features present: {[f for f in matching_features if f in recs_data.columns]}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise