"""
Enhanced feature engineering for improved PUMS-RECS matching.

This module creates many comparable features between PUMS and RECS datasets
to improve probabilistic matching quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from .dtype_utils import safe_cut, safe_qcut

logger = logging.getLogger(__name__)


def create_comprehensive_matching_features(df: pd.DataFrame, 
                                          dataset_type: str = 'pums',
                                          current_year: int = 2023) -> pd.DataFrame:
    """
    Create comprehensive features for matching between PUMS and RECS.
    
    This function creates many comparable features that exist in both datasets,
    improving the quality of probabilistic matching.
    
    Args:
        df: Input DataFrame (PUMS or RECS)
        dataset_type: 'pums' or 'recs' to handle dataset-specific logic
        current_year: Current year for age calculations
        
    Returns:
        DataFrame with many engineered features for matching
    """
    df = df.copy()
    logger.info(f"Creating comprehensive features for {dataset_type} dataset")
    
    # ========== HOUSEHOLD SIZE FEATURES (Critical for matching) ==========
    if dataset_type == 'pums':
        if 'NP' in df.columns:
            df['household_size'] = df['NP']
    else:  # RECS
        if 'NHSLDMEM' in df.columns:
            df['household_size'] = df['NHSLDMEM']
    
    if 'household_size' in df.columns:
        # Multiple categorizations for flexible matching
        df['hh_size_2cat'] = safe_cut(df['household_size'], 
                                    bins=[0, 2, 100], 
                                    labels=['small', 'large'])
        
        df['hh_size_3cat'] = safe_cut(df['household_size'], 
                                    bins=[0, 1, 3, 100], 
                                    labels=['single', 'small', 'large'])
        
        df['hh_size_4cat'] = safe_cut(df['household_size'], 
                                    bins=[0, 1, 2, 4, 100], 
                                    labels=['single', 'couple', 'small_family', 'large_family'])
        
        df['hh_size_5cat'] = safe_cut(df['household_size'], 
                                    bins=[0, 1, 2, 3, 5, 100], 
                                    labels=['one', 'two', 'three', 'four_five', 'six_plus'])
        
        # Binary features
        df['is_single_person'] = (df['household_size'] == 1).astype(int)
        df['is_couple'] = (df['household_size'] == 2).astype(int)
        df['is_large_household'] = (df['household_size'] >= 5).astype(int)
        df['has_3plus_people'] = (df['household_size'] >= 3).astype(int)
    
    # ========== INCOME FEATURES (Important for energy consumption) ==========
    if dataset_type == 'pums':
        income_col = 'HINCP' if 'HINCP' in df.columns else 'household_income'
    else:  # RECS
        income_col = 'MONEYPY' if 'MONEYPY' in df.columns else 'household_income'
    
    if income_col in df.columns and df[income_col].notna().any():
        # Clean income data
        income_clean = df[income_col].fillna(0).clip(lower=0)
        
        if (income_clean > 0).any():
            # Log transformation for better distribution
            df['income_log'] = np.log1p(income_clean)
            
            # Multiple income categorizations
            try:
                df['income_quintile'] = safe_qcut(income_clean[income_clean > 0], 
                                               q=5, 
                                               labels=['q1', 'q2', 'q3', 'q4', 'q5'],
                                               duplicates='drop').reindex(df.index, fill_value='q1')
                
                df['income_tercile'] = safe_qcut(income_clean[income_clean > 0], 
                                              q=3, 
                                              labels=['low', 'medium', 'high'],
                                              duplicates='drop').reindex(df.index, fill_value='low')
                
                df['income_decile'] = safe_qcut(income_clean[income_clean > 0], 
                                             q=10, 
                                             labels=False,
                                             duplicates='drop').reindex(df.index, fill_value=0)
            except:
                df['income_quintile'] = 'q3'
                df['income_tercile'] = 'medium'
                df['income_decile'] = 5
            
            # Binary income features
            median_income = income_clean[income_clean > 0].median()
            df['above_median_income'] = (income_clean > median_income).astype(int)
            df['low_income'] = (income_clean < income_clean.quantile(0.3)).astype(int)
            df['high_income'] = (income_clean > income_clean.quantile(0.7)).astype(int)
            
            # Income per capita
            if 'household_size' in df.columns:
                df['income_per_capita'] = income_clean / df['household_size'].clip(lower=1)
                df['income_per_capita_cat'] = safe_qcut(df['income_per_capita'][df['income_per_capita'] > 0], 
                                                      q=3, 
                                                      labels=['low', 'medium', 'high'],
                                                      duplicates='drop').reindex(df.index, fill_value='low')
    
    # ========== ROOMS AND SPACE FEATURES ==========
    if dataset_type == 'pums':
        rooms_col = 'RMSP' if 'RMSP' in df.columns else None
        bedrooms_col = 'BDSP' if 'BDSP' in df.columns else None
    else:  # RECS
        rooms_col = 'TOTROOMS' if 'TOTROOMS' in df.columns else 'total_rooms'
        bedrooms_col = 'BEDROOMS' if 'BEDROOMS' in df.columns else 'num_bedrooms'
    
    if rooms_col and rooms_col in df.columns:
        df['total_rooms'] = df[rooms_col]
        
        # Multiple room categorizations
        df['rooms_3cat'] = safe_cut(df['total_rooms'], 
                                  bins=[0, 4, 7, 100], 
                                  labels=['small', 'medium', 'large'])
        
        df['rooms_4cat'] = safe_cut(df['total_rooms'], 
                                  bins=[0, 3, 5, 8, 100], 
                                  labels=['tiny', 'small', 'medium', 'large'])
        
        df['rooms_5cat'] = safe_cut(df['total_rooms'], 
                                  bins=[0, 3, 5, 7, 9, 100], 
                                  labels=['tiny', 'small', 'medium', 'large', 'xlarge'])
        
        # Binary room features
        df['has_many_rooms'] = (df['total_rooms'] >= 7).astype(int)
        df['has_few_rooms'] = (df['total_rooms'] <= 4).astype(int)
        
        # Rooms per person (crowding indicator)
        if 'household_size' in df.columns:
            df['rooms_per_person'] = df['total_rooms'] / df['household_size'].clip(lower=1)
            df['rooms_pp_cat'] = safe_cut(df['rooms_per_person'], 
                                       bins=[0, 1.5, 2.5, 100], 
                                       labels=['crowded', 'normal', 'spacious'])
            df['crowding_index'] = df['household_size'] / df['total_rooms'].clip(lower=1)
            df['is_crowded'] = (df['rooms_per_person'] < 1.5).astype(int)
    
    if bedrooms_col and bedrooms_col in df.columns:
        df['num_bedrooms'] = df[bedrooms_col]
        
        # Bedroom categorizations
        df['bedrooms_3cat'] = safe_cut(df['num_bedrooms'], 
                                     bins=[-1, 1, 3, 100], 
                                     labels=['small', 'medium', 'large'])
        
        df['bedrooms_4cat'] = safe_cut(df['num_bedrooms'], 
                                     bins=[-1, 0, 2, 4, 100], 
                                     labels=['studio', 'small', 'medium', 'large'])
        
        # Binary bedroom features
        df['studio_flag'] = (df['num_bedrooms'] == 0).astype(int)
        df['one_bedroom'] = (df['num_bedrooms'] == 1).astype(int)
        df['multi_bedroom'] = (df['num_bedrooms'] >= 3).astype(int)
        
        # Bedrooms per person
        if 'household_size' in df.columns:
            df['bedrooms_per_person'] = df['num_bedrooms'] / df['household_size'].clip(lower=1)
            df['bedroom_deficit'] = (df['household_size'] - df['num_bedrooms'] - 1).clip(lower=0)
    
    # ========== BUILDING AGE FEATURES ==========
    if dataset_type == 'pums':
        if 'YRBLT' in df.columns:
            df['building_age'] = current_year - df['YRBLT']
    else:  # RECS
        if 'YEARMADERANGE' in df.columns:
            # RECS uses ranges, map to approximate year
            year_map = {
                1: 1945,  # Before 1950
                2: 1955,  # 1950-1959
                3: 1965,  # 1960-1969
                4: 1975,  # 1970-1979
                5: 1985,  # 1980-1989
                6: 1995,  # 1990-1999
                7: 2005,  # 2000-2009
                8: 2015   # 2010+
            }
            df['building_age'] = current_year - df['YEARMADERANGE'].map(year_map).fillna(2000)
    
    if 'building_age' in df.columns:
        # Multiple age categorizations
        df['age_3cat'] = safe_cut(df['building_age'], 
                               bins=[0, 20, 50, 200], 
                               labels=['new', 'medium', 'old'])
        
        df['age_4cat'] = safe_cut(df['building_age'], 
                               bins=[0, 10, 30, 60, 200], 
                               labels=['new', 'recent', 'mature', 'old'])
        
        df['age_5cat'] = safe_cut(df['building_age'], 
                               bins=[0, 10, 25, 50, 75, 200], 
                               labels=['new', 'modern', 'mature', 'old', 'historic'])
        
        # Decade built
        df['decade_built'] = ((current_year - df['building_age']) // 10) * 10
        
        # Binary age features
        df['is_new_building'] = (df['building_age'] < 15).astype(int)
        df['is_old_building'] = (df['building_age'] > 50).astype(int)
        df['is_pre_1980'] = (df['building_age'] > 43).astype(int)  # Energy code watershed
        df['is_21st_century'] = (df['building_age'] < 23).astype(int)
    
    # ========== VEHICLE FEATURES (PUMS specific but important) ==========
    if dataset_type == 'pums' and 'VEH' in df.columns:
        df['num_vehicles'] = df['VEH']
        
        # Vehicle categorizations
        df['vehicle_cat'] = safe_cut(df['VEH'], 
                                  bins=[-1, 0, 1, 2, 100], 
                                  labels=['none', 'one', 'two', 'many'])
        
        # Binary vehicle features
        df['has_vehicle'] = (df['VEH'] > 0).astype(int)
        df['no_vehicle'] = (df['VEH'] == 0).astype(int)
        df['multi_vehicle'] = (df['VEH'] >= 2).astype(int)
        df['many_vehicles'] = (df['VEH'] >= 3).astype(int)
        
        # Vehicles per person
        if 'household_size' in df.columns:
            df['vehicles_per_adult'] = df['VEH'] / df['household_size'].clip(lower=1)
            df['vehicle_sufficient'] = (df['VEH'] >= (df['household_size'] - 1)).astype(int)
    
    # ========== TENURE AND OWNERSHIP ==========
    if dataset_type == 'pums' and 'TEN' in df.columns:
        df['is_owner'] = df['TEN'].isin([1, 2]).astype(int)
        df['is_renter'] = df['TEN'].isin([3]).astype(int)
        df['tenure_type'] = df['TEN'].map({
            1: 'owned_clear', 
            2: 'owned_mortgage', 
            3: 'rented', 
            4: 'other'
        }).fillna('unknown')
    elif dataset_type == 'recs' and 'KOWNRENT' in df.columns:
        df['is_owner'] = (df['KOWNRENT'] == 1).astype(int)
        df['is_renter'] = (df['KOWNRENT'] == 2).astype(int)
        df['tenure_type'] = df['KOWNRENT'].map({
            1: 'owned',
            2: 'rented',
            3: 'other'
        }).fillna('unknown')
    
    # ========== CLIMATE AND GEOGRAPHIC FEATURES ==========
    if 'DIVISION' in df.columns:
        # Map divisions to climate zones
        df['climate_zone'] = df['DIVISION'].map({
            1: 'cold',       # New England
            2: 'cold',       # Middle Atlantic
            3: 'moderate',   # East North Central
            4: 'moderate',   # West North Central
            5: 'hot_humid',  # South Atlantic
            6: 'hot_humid',  # East South Central
            7: 'mixed',      # West South Central
            8: 'mixed',      # Mountain
            9: 'moderate'    # Pacific
        }).fillna('moderate')
        
        # Binary climate features
        df['cold_climate'] = (df['climate_zone'] == 'cold').astype(int)
        df['hot_climate'] = df['climate_zone'].isin(['hot_humid', 'hot_dry']).astype(int)
        df['moderate_climate'] = (df['climate_zone'] == 'moderate').astype(int)
    
    if 'REGION' in df.columns:
        # Regional features
        df['is_northeast'] = (df['REGION'] == 1).astype(int)
        df['is_midwest'] = (df['REGION'] == 2).astype(int)
        df['is_south'] = (df['REGION'] == 3).astype(int)
        df['is_west'] = (df['REGION'] == 4).astype(int)
    
    # ========== URBAN/RURAL FEATURES ==========
    if dataset_type == 'recs':
        # RECS has explicit urban/rural data
        if 'urban_rural_raw' in df.columns:
            df['urban_rural'] = df['urban_rural_raw'].map({
                'U': 'urban',
                'C': 'urban',  # Urban cluster
                'R': 'rural'
            }).fillna('urban')
        elif 'urban_rural' not in df.columns:
            df['urban_rural'] = 'urban'  # Default
        
        df['is_urban'] = (df['urban_rural'] == 'urban').astype(int)
        df['is_rural'] = (df['urban_rural'] == 'rural').astype(int)
    elif dataset_type == 'pums':
        # PUMS: Use population density proxy based on PUMA size and state
        # This is a simplified heuristic - larger PUMAs tend to be more rural
        if 'PUMA' in df.columns and 'STATE' in df.columns:
            # States with large rural areas (simplified mapping)
            rural_states = ['02', '08', '16', '30', '38', '46', '50', '56', '69']  # AK, CO, ID, MT, ND, SD, VT, WY, etc
            
            # Simple heuristic: if state is rural-heavy and PUMA < 1000, likely rural
            df['is_likely_rural'] = (
                (df['STATE'].isin(rural_states)) | 
                (df['PUMA'].str[:1].isin(['0', '1']))  # Low PUMA numbers often rural
            ).astype(int)
            
            df['urban_rural'] = np.where(df['is_likely_rural'], 'rural', 'urban')
            df['is_urban'] = (df['urban_rural'] == 'urban').astype(int)
            df['is_rural'] = (df['urban_rural'] == 'rural').astype(int)
            df.drop('is_likely_rural', axis=1, inplace=True)
        else:
            # Default if we don't have the fields
            df['urban_rural'] = 'urban'
            df['is_urban'] = 1
            df['is_rural'] = 0
    
    # ========== HOUSEHOLD COMPOSITION ==========
    if dataset_type == 'pums':
        if 'R65' in df.columns:
            df['has_seniors'] = (df['R65'] > 0).astype(int)
            df['num_seniors'] = df['R65']
            df['all_seniors'] = ((df['R65'] == df.get('NP', 1)) & (df['R65'] > 0)).astype(int)
        
        if 'R18' in df.columns:
            df['has_children'] = (df['R18'] > 0).astype(int)
            df['num_children'] = df['R18']
            df['children_ratio'] = df['R18'] / df.get('NP', 1).clip(lower=1)
            df['many_children'] = (df['R18'] >= 3).astype(int)
    
    # ========== COMPOSITE INDICES ==========
    
    # Socioeconomic status index
    ses_components = []
    if 'income_decile' in df.columns:
        ses_components.append(df['income_decile'] / 10)
    if 'rooms_per_person' in df.columns:
        ses_components.append(df['rooms_per_person'].clip(upper=3) / 3)
    if 'is_owner' in df.columns:
        ses_components.append(df['is_owner'])
    
    if ses_components:
        # Use pd.DataFrame to properly calculate mean across columns
        df['ses_index'] = pd.DataFrame(ses_components).T.mean(axis=1).values
        df['ses_category'] = safe_qcut(df['ses_index'], 
                                    q=3, 
                                    labels=['low', 'medium', 'high'],
                                    duplicates='drop')
    
    # Housing density index
    density_components = []
    if 'household_size' in df.columns:
        density_components.append(df['household_size'] / 5)  # Normalize to 0-1 range
    if 'crowding_index' in df.columns:
        density_components.append(df['crowding_index'].clip(upper=1))
    
    if density_components:
        # Use pd.DataFrame to properly calculate mean across columns
        df['density_index'] = pd.DataFrame(density_components).T.mean(axis=1).values
        df['density_category'] = safe_qcut(df['density_index'], 
                                        q=3, 
                                        labels=['low', 'medium', 'high'],
                                        duplicates='drop')
    
    # Energy vulnerability index
    energy_vuln = []
    if 'low_income' in df.columns:
        energy_vuln.append(df['low_income'])
    if 'is_old_building' in df.columns:
        energy_vuln.append(df['is_old_building'])
    if 'is_large_household' in df.columns:
        energy_vuln.append(df['is_large_household'])
    
    if energy_vuln:
        # Use pd.DataFrame to properly calculate sum across columns
        df['energy_vulnerability'] = pd.DataFrame(energy_vuln).T.sum(axis=1).values
        df['high_energy_vulnerable'] = (df['energy_vulnerability'] >= 2).astype(int)
    
    # ========== INTERACTION FEATURES ==========
    
    # Income-size interaction
    if 'income_tercile' in df.columns and 'hh_size_3cat' in df.columns:
        df['income_size_group'] = df['income_tercile'] + '_' + df['hh_size_3cat']
    
    # Age-income interaction
    if 'income_tercile' in df.columns and 'age_3cat' in df.columns:
        df['income_age_group'] = df['income_tercile'] + '_' + df['age_3cat']
    
    # Climate-size interaction
    if 'climate_zone' in df.columns and 'hh_size_3cat' in df.columns:
        df['climate_size_group'] = df['climate_zone'] + '_' + df['hh_size_3cat']
    
    # ========== ENERGY-RELATED FEATURES ==========
    
    # Building efficiency proxy (newer buildings are generally more efficient)
    if 'building_age' in df.columns:
        df['efficiency_proxy'] = safe_cut(df['building_age'],
                                       bins=[0, 10, 25, 50, 200],
                                       labels=['high', 'medium', 'low', 'very_low'])
        df['is_efficient_building'] = (df['building_age'] < 20).astype(int)
    
    # Collect all new columns to add at once (avoid fragmentation)
    new_columns = {}
    
    # Energy intensity categories (if available from PUMS or RECS)
    if 'ELEP' in df.columns:  # PUMS electricity cost
        # Create electricity burden (cost relative to income)
        if 'HINCP' in df.columns:
            new_columns['electricity_burden'] = np.where(
                df['HINCP'] > 0,
                (df['ELEP'] * 12) / df['HINCP'],  # Annual elec cost / income
                0
            )
            new_columns['high_elec_burden'] = (new_columns['electricity_burden'] > 0.05).astype(int)
    
    # Heating/cooling needs based on climate
    if 'climate_zone' in df.columns:
        new_columns['high_heating_need'] = df['climate_zone'].isin(['cold']).astype(int)
        new_columns['high_cooling_need'] = df['climate_zone'].isin(['hot_humid', 'hot_dry']).astype(int)
        new_columns['balanced_hvac_need'] = df['climate_zone'].isin(['moderate', 'mixed']).astype(int)
    
    # Occupancy intensity (people per room)
    if 'household_size' in df.columns and 'total_rooms' in df.columns:
        new_columns['occupancy_intensity'] = np.where(
            df['total_rooms'] > 0,
            df['household_size'] / df['total_rooms'],
            df['household_size']
        )
        new_columns['high_occupancy'] = (new_columns['occupancy_intensity'] > 0.75).astype(int)
    
    # Work from home potential (affects daytime energy use)
    if dataset_type == 'pums':
        # Check for work from home indicators
        if 'JWTR' in df.columns:  # Journey to work
            new_columns['likely_wfh'] = (df['JWTR'] == 11).astype(int)  # 11 = worked at home
        else:
            new_columns['likely_wfh'] = 0
    elif dataset_type == 'recs':
        # For RECS, we might need to infer from other variables
        new_columns['likely_wfh'] = 0  # Default, could be enhanced with more data
    
    # Building envelope quality proxy
    if 'building_age' in df.columns and 'climate_zone' in df.columns:
        # Older buildings in harsh climates likely have worse envelopes
        new_columns['poor_envelope'] = (
            (df['building_age'] > 40) & 
            df['climate_zone'].isin(['cold', 'hot_humid'])
        ).astype(int)
    
    # Technology adoption indicator
    if dataset_type == 'pums':
        tech_indicators = []
        if 'HISPEED' in df.columns:
            tech_indicators.append(df['HISPEED'])
        if 'SMARTPHONE' in df.columns:
            tech_indicators.append(df['SMARTPHONE'])
        if 'LAPTOP' in df.columns:
            tech_indicators.append(df['LAPTOP'])
        
        if tech_indicators:
            new_columns['tech_adoption'] = pd.DataFrame(tech_indicators).T.sum(axis=1).values
            # Only create high_tech if it doesn't already exist
            if 'high_tech' not in df.columns:
                new_columns['high_tech'] = (new_columns['tech_adoption'] >= 2).astype(int)
    
    # Size-adjusted energy indicators
    if 'household_size' in df.columns:
        # Large families likely use more energy
        new_columns['size_energy_factor'] = safe_cut(df['household_size'],
                                          bins=[0, 1, 2, 4, 100],
                                          labels=['low', 'medium', 'high', 'very_high'])
    
    # Add all new columns at once to avoid fragmentation
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    logger.info(f"Created {len(df.columns)} total features for {dataset_type} dataset")
    
    return df


def align_features_for_matching(pums_df: pd.DataFrame, recs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align features between PUMS and RECS for optimal matching.
    
    This function ensures both datasets have the same features for comparison.
    
    Args:
        pums_df: PUMS DataFrame with features
        recs_df: RECS DataFrame with features
        
    Returns:
        Tuple of (aligned_pums, aligned_recs) with matching columns
    """
    # Find common columns
    common_cols = set(pums_df.columns) & set(recs_df.columns)
    
    # Always keep identification columns
    keep_cols = ['building_id', 'SERIALNO', 'template_id', 'recs_id', 'DOEID']
    for col in keep_cols:
        if col in pums_df.columns or col in recs_df.columns:
            common_cols.add(col)
    
    # Ensure both have the same columns
    for col in common_cols:
        if col not in pums_df.columns:
            pums_df[col] = np.nan
        if col not in recs_df.columns:
            recs_df[col] = np.nan
    
    logger.info(f"Aligned datasets with {len(common_cols)} common features")
    
    return pums_df, recs_df


def get_matching_features_list() -> List[str]:
    """
    Get list of features to use for matching.
    
    Returns:
        List of feature names suitable for matching
    """
    return [
        # Geographic
        'REGION', 'DIVISION', 'climate_zone',
        
        # Household size variations
        'household_size', 'hh_size_2cat', 'hh_size_3cat', 'hh_size_4cat',
        'is_single_person', 'is_large_household',
        
        # Income variations
        'income_quintile', 'income_tercile', 'income_decile',
        'low_income', 'high_income', 'above_median_income',
        
        # Rooms and space
        'total_rooms', 'rooms_3cat', 'rooms_4cat',
        'num_bedrooms', 'bedrooms_3cat',
        'rooms_per_person', 'rooms_pp_cat',
        
        # Building age
        'building_age', 'age_3cat', 'age_4cat',
        'is_new_building', 'is_old_building',
        
        # Other characteristics
        'is_owner', 'is_renter', 'tenure_type',
        'urban_rural', 'is_urban', 'is_rural',
        
        # Household composition
        'has_children', 'has_seniors',
        
        # Energy-related features
        'efficiency_proxy', 'is_efficient_building',
        'high_heating_need', 'high_cooling_need', 'balanced_hvac_need',
        'high_occupancy', 'poor_envelope', 'size_energy_factor',
        
        # Composite indices
        'ses_category', 'density_category',
        'high_energy_vulnerable',
        
        # Interaction features
        'income_size_group', 'income_age_group'
    ]