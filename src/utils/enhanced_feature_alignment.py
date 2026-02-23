"""
Enhanced Feature Alignment Module for PUMS-ATUS Matching.

This module creates 50+ aligned features between PUMS persons and ATUS respondents
to dramatically improve probabilistic matching quality in Phase 3.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def align_pums_atus_features(pums_df: pd.DataFrame, atus_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aligned features between PUMS persons and ATUS respondents.
    
    This function ensures both datasets have identical feature sets with
    consistent encoding for improved matching quality.
    
    Args:
        pums_df: PUMS person data
        atus_df: ATUS respondent data
        
    Returns:
        Tuple of (aligned_pums, aligned_atus) with 50+ comparable features
    """
    logger.info("Starting enhanced feature alignment between PUMS and ATUS")
    
    # Create copies to avoid modifying originals
    pums = pums_df.copy()
    atus = atus_df.copy()
    
    # ========== DEMOGRAPHIC ALIGNMENT ==========
    pums, atus = align_age_features(pums, atus)
    pums, atus = align_sex_features(pums, atus)
    pums, atus = align_race_ethnicity_features(pums, atus)
    
    # ========== EMPLOYMENT ALIGNMENT ==========
    pums, atus = align_employment_features(pums, atus)
    pums, atus = align_work_schedule_features(pums, atus)
    pums, atus = align_occupation_features(pums, atus)
    
    # ========== EDUCATION ALIGNMENT ==========
    pums, atus = align_education_features(pums, atus)
    
    # ========== HOUSEHOLD ALIGNMENT ==========
    pums, atus = align_household_features(pums, atus)
    pums, atus = align_marital_features(pums, atus)
    pums, atus = align_children_features(pums, atus)
    
    # ========== ECONOMIC ALIGNMENT ==========
    pums, atus = align_income_features(pums, atus)
    
    # ========== TIME USE PREDICTORS ==========
    pums, atus = create_time_use_predictors(pums, atus)
    
    # ========== ACTIVITY LIKELIHOOD SCORES ==========
    pums, atus = create_activity_likelihood_scores(pums, atus)
    
    # ========== COMPOSITE INDICES ==========
    pums, atus = create_composite_indices(pums, atus)
    
    # ========== INTERACTION FEATURES ==========
    pums, atus = create_interaction_features(pums, atus)
    
    # Ensure all features are present in both datasets
    pums, atus = ensure_feature_completeness(pums, atus)
    
    # Log alignment summary
    common_features = set(pums.columns) & set(atus.columns)
    logger.info(f"Feature alignment complete: {len(common_features)} common features created")
    
    return pums, atus


def align_age_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align age-related features between datasets."""
    
    # PUMS age field
    if 'AGEP' in pums.columns:
        pums['age'] = pums['AGEP']
    elif 'age' not in pums.columns:
        pums['age'] = 35  # Default adult age
    
    # ATUS age field (already standardized in loader)
    if 'age' not in atus.columns:
        atus['age'] = 35  # Default adult age
    
    # Create identical age categorizations
    age_bins = [0, 5, 13, 18, 25, 35, 45, 55, 65, 75, 100]
    age_labels = ['infant', 'child', 'teen', 'young_adult', 'adult', 
                  'middle_age', 'late_middle', 'senior', 'elderly', 'very_elderly']
    
    pums['age_group_detailed'] = pd.cut(pums['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    atus['age_group_detailed'] = pd.cut(atus['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # Simple age groups
    simple_bins = [0, 18, 35, 50, 65, 100]
    simple_labels = ['minor', 'young_adult', 'adult', 'middle_age', 'senior']
    
    pums['age_group_simple'] = pd.cut(pums['age'], bins=simple_bins, labels=simple_labels, include_lowest=True)
    atus['age_group_simple'] = pd.cut(atus['age'], bins=simple_bins, labels=simple_labels, include_lowest=True)
    
    # Age decades
    pums['age_decade'] = (pums['age'] // 10) * 10
    atus['age_decade'] = (atus['age'] // 10) * 10
    
    # Binary age indicators
    for threshold in [18, 21, 25, 30, 40, 50, 60, 65, 70]:
        pums[f'age_over_{threshold}'] = (pums['age'] >= threshold).astype(int)
        atus[f'age_over_{threshold}'] = (atus['age'] >= threshold).astype(int)
    
    # Life stage indicators
    pums['is_school_age'] = ((pums['age'] >= 5) & (pums['age'] <= 17)).astype(int)
    atus['is_school_age'] = ((atus['age'] >= 5) & (atus['age'] <= 17)).astype(int)
    
    pums['is_college_age'] = ((pums['age'] >= 18) & (pums['age'] <= 24)).astype(int)
    atus['is_college_age'] = ((atus['age'] >= 18) & (atus['age'] <= 24)).astype(int)
    
    pums['is_prime_working'] = ((pums['age'] >= 25) & (pums['age'] <= 54)).astype(int)
    atus['is_prime_working'] = ((atus['age'] >= 25) & (atus['age'] <= 54)).astype(int)
    
    pums['is_retirement_age'] = (pums['age'] >= 65).astype(int)
    atus['is_retirement_age'] = (atus['age'] >= 65).astype(int)
    
    return pums, atus


def align_sex_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align sex/gender features."""
    
    # PUMS: SEX field (1=Male, 2=Female)
    if 'SEX' in pums.columns:
        pums['is_female'] = (pums['SEX'] == 2).astype(int)
        pums['is_male'] = (pums['SEX'] == 1).astype(int)
        pums['sex_code'] = pums['SEX']
    else:
        pums['is_female'] = 0
        pums['is_male'] = 1
        pums['sex_code'] = 1
    
    # ATUS: sex field (1=Male, 2=Female) 
    if 'sex' in atus.columns:
        atus['is_female'] = (atus['sex'] == 2).astype(int)
        atus['is_male'] = (atus['sex'] == 1).astype(int)
        atus['sex_code'] = atus['sex']
    else:
        atus['is_female'] = 0
        atus['is_male'] = 1
        atus['sex_code'] = 1
    
    return pums, atus


def align_race_ethnicity_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align race and ethnicity features."""
    
    # PUMS race
    if 'RAC1P' in pums.columns:
        pums['race_white'] = (pums['RAC1P'] == 1).astype(int)
        pums['race_black'] = (pums['RAC1P'] == 2).astype(int)
        pums['race_asian'] = (pums['RAC1P'] == 6).astype(int)
        pums['race_other'] = (~pums['RAC1P'].isin([1, 2, 6])).astype(int)
    else:
        pums['race_white'] = 1
        pums['race_black'] = 0
        pums['race_asian'] = 0
        pums['race_other'] = 0
    
    # ATUS race (from race_cat if available)
    if 'race_cat' in atus.columns:
        atus['race_white'] = (atus['race_cat'] == 'white').astype(int)
        atus['race_black'] = (atus['race_cat'] == 'black').astype(int)
        atus['race_asian'] = (atus['race_cat'] == 'asian').astype(int)
        atus['race_other'] = (~atus['race_cat'].isin(['white', 'black', 'asian'])).astype(int)
    else:
        atus['race_white'] = 1
        atus['race_black'] = 0
        atus['race_asian'] = 0
        atus['race_other'] = 0
    
    # Hispanic ethnicity
    if 'HISP' in pums.columns:
        pums['is_hispanic'] = (pums['HISP'] > 1).astype(int)
    else:
        pums['is_hispanic'] = 0
    
    if 'is_hispanic' not in atus.columns:
        atus['is_hispanic'] = 0
    else:
        atus['is_hispanic'] = atus['is_hispanic'].astype(int)
    
    # Minority status
    pums['is_minority'] = ((pums['race_white'] == 0) | (pums['is_hispanic'] == 1)).astype(int)
    atus['is_minority'] = ((atus['race_white'] == 0) | (atus['is_hispanic'] == 1)).astype(int)
    
    return pums, atus


def align_employment_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align employment status features."""
    
    # PUMS employment
    if 'ESR' in pums.columns:
        pums['employed'] = pums['ESR'].isin([1, 2]).astype(int)
        pums['unemployed'] = pums['ESR'].isin([3]).astype(int)
        pums['not_in_labor_force'] = pums['ESR'].isin([6]).astype(int)
    else:
        pums['employed'] = 0
        pums['unemployed'] = 0
        pums['not_in_labor_force'] = 1
    
    # ATUS employment (from employment_category if available)
    if 'employment_category' in atus.columns:
        atus['employed'] = atus['employment_category'].str.contains('employed', na=False).astype(int)
        atus['unemployed'] = (atus['employment_category'] == 'unemployed').astype(int)
        atus['not_in_labor_force'] = (atus['employment_category'] == 'not_working').astype(int)
    else:
        atus['employed'] = 0
        atus['unemployed'] = 0
        atus['not_in_labor_force'] = 1
    
    # Employment type
    pums['emp_status_code'] = pums['employed'] * 2 + pums['unemployed']
    atus['emp_status_code'] = atus['employed'] * 2 + atus['unemployed']
    
    return pums, atus


def align_work_schedule_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align work schedule and hours features."""
    
    # PUMS work hours
    if 'WKHP' in pums.columns:
        pums['work_hours_weekly'] = pums['WKHP'].fillna(0)
    else:
        pums['work_hours_weekly'] = 0
    
    # ATUS work hours (from usual_hours_worked or work_hours_cat)
    if 'usual_hours_worked' in atus.columns:
        atus['work_hours_weekly'] = atus['usual_hours_worked'].fillna(0)
    elif 'work_main_job' in atus.columns:
        # Convert minutes to weekly hours (assuming 5 day work week)
        atus['work_hours_weekly'] = (atus['work_main_job'] / 60 * 5).fillna(0)
    else:
        atus['work_hours_weekly'] = 0
    
    # Work schedule categories
    for df in [pums, atus]:
        df['not_working'] = (df['work_hours_weekly'] == 0).astype(int)
        df['part_time'] = ((df['work_hours_weekly'] > 0) & (df['work_hours_weekly'] < 35)).astype(int)
        df['full_time'] = ((df['work_hours_weekly'] >= 35) & (df['work_hours_weekly'] <= 45)).astype(int)
        df['overtime'] = (df['work_hours_weekly'] > 45).astype(int)
        
        # Work intensity
        df['work_intensity'] = pd.cut(
            df['work_hours_weekly'],
            bins=[0, 1, 20, 35, 40, 50, 100],
            labels=['none', 'minimal', 'part_time', 'standard', 'full_plus', 'excessive'],
            include_lowest=True
        )
    
    # Work from home
    if 'JWTRNS' in pums.columns:
        pums['works_from_home'] = (pums['JWTRNS'] == 11).astype(int)
    elif 'works_from_home' in pums.columns:
        pums['works_from_home'] = pums['works_from_home'].astype(int)
    else:
        pums['works_from_home'] = 0
    
    if 'works_from_home' not in atus.columns:
        atus['works_from_home'] = 0
    else:
        atus['works_from_home'] = atus['works_from_home'].astype(int)
    
    return pums, atus


def align_occupation_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align occupation features."""
    
    # PUMS occupation
    if 'OCCP' in pums.columns:
        pums['occ_management'] = pums['OCCP'].astype(str).str.startswith('1').astype(int)
        pums['occ_professional'] = pums['OCCP'].astype(str).str.startswith('2').astype(int)
        pums['occ_service'] = pums['OCCP'].astype(str).str.startswith('3').astype(int)
        pums['occ_sales'] = pums['OCCP'].astype(str).str.startswith('4').astype(int)
        pums['occ_blue_collar'] = pums['OCCP'].astype(str).str[0].isin(['5', '6', '7']).astype(int)
    else:
        pums['occ_management'] = 0
        pums['occ_professional'] = 0
        pums['occ_service'] = 0
        pums['occ_sales'] = 0
        pums['occ_blue_collar'] = 0
    
    # ATUS occupation (from occupation_cat if available)
    if 'occupation_cat' in atus.columns:
        atus['occ_management'] = (atus['occupation_cat'] == 'management').astype(int)
        atus['occ_professional'] = (atus['occupation_cat'] == 'professional').astype(int)
        atus['occ_service'] = (atus['occupation_cat'] == 'service').astype(int)
        atus['occ_sales'] = (atus['occupation_cat'] == 'sales').astype(int)
        atus['occ_blue_collar'] = atus['occupation_cat'].isin(['construction', 'maintenance', 'production']).astype(int)
    else:
        atus['occ_management'] = 0
        atus['occ_professional'] = 0
        atus['occ_service'] = 0
        atus['occ_sales'] = 0
        atus['occ_blue_collar'] = 0
    
    # White collar indicator
    pums['white_collar'] = (pums['occ_management'] | pums['occ_professional'] | pums['occ_sales']).astype(int)
    atus['white_collar'] = (atus['occ_management'] | atus['occ_professional'] | atus['occ_sales']).astype(int)
    
    return pums, atus


def align_education_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align education features."""
    
    # PUMS education
    if 'SCHL' in pums.columns:
        pums['edu_less_than_hs'] = (pums['SCHL'] < 16).astype(int)
        pums['edu_high_school'] = pums['SCHL'].isin([16, 17]).astype(int)
        pums['edu_some_college'] = pums['SCHL'].isin([18, 19, 20]).astype(int)
        pums['edu_bachelors'] = (pums['SCHL'] == 21).astype(int)
        pums['edu_graduate'] = (pums['SCHL'] >= 22).astype(int)
        
        # Education years (approximate)
        edu_years_map = {i: min(i-3, 8) for i in range(1, 16)}  # Less than HS
        edu_years_map.update({16: 12, 17: 12})  # HS
        edu_years_map.update({18: 13, 19: 14, 20: 14})  # Some college
        edu_years_map.update({21: 16, 22: 18, 23: 20, 24: 22})  # Bachelor+
        pums['education_years'] = pums['SCHL'].map(edu_years_map).fillna(12)
    else:
        pums['edu_less_than_hs'] = 0
        pums['edu_high_school'] = 1
        pums['edu_some_college'] = 0
        pums['edu_bachelors'] = 0
        pums['edu_graduate'] = 0
        pums['education_years'] = 12
    
    # ATUS education
    if 'education_level' in atus.columns:
        atus['edu_less_than_hs'] = (atus['education_level'] == 'less_than_hs').astype(int)
        atus['edu_high_school'] = (atus['education_level'] == 'high_school').astype(int)
        atus['edu_some_college'] = (atus['education_level'] == 'some_college').astype(int)
        atus['edu_bachelors'] = (atus['education_level'] == 'bachelors').astype(int)
        atus['edu_graduate'] = (atus['education_level'] == 'graduate').astype(int)
        
        # Map to years
        edu_years_map = {
            'less_than_hs': 10,
            'high_school': 12,
            'some_college': 14,
            'bachelors': 16,
            'graduate': 18
        }
        atus['education_years'] = atus['education_level'].map(edu_years_map).fillna(12)
    else:
        atus['edu_less_than_hs'] = 0
        atus['edu_high_school'] = 1
        atus['edu_some_college'] = 0
        atus['edu_bachelors'] = 0
        atus['edu_graduate'] = 0
        atus['education_years'] = 12
    
    # College indicator
    pums['has_college'] = (pums['edu_bachelors'] | pums['edu_graduate']).astype(int)
    atus['has_college'] = (atus['edu_bachelors'] | atus['edu_graduate']).astype(int)
    
    return pums, atus


def align_household_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align household composition features."""
    
    # PUMS household size
    if 'household_size' in pums.columns:
        pums['hh_size'] = pums['household_size']
    elif 'NP' in pums.columns:
        pums['hh_size'] = pums['NP']
    else:
        pums['hh_size'] = 2
    
    # ATUS household size
    if 'household_size' in atus.columns:
        atus['hh_size'] = atus['household_size']
    elif 'hh_size_roster' in atus.columns:
        atus['hh_size'] = atus['hh_size_roster']
    else:
        atus['hh_size'] = 2
    
    # Household size categories
    for df in [pums, atus]:
        df['single_person_hh'] = (df['hh_size'] == 1).astype(int)
        df['two_person_hh'] = (df['hh_size'] == 2).astype(int)
        df['small_family'] = ((df['hh_size'] >= 3) & (df['hh_size'] <= 4)).astype(int)
        df['large_family'] = (df['hh_size'] >= 5).astype(int)
        
        # Household size grouped
        df['hh_size_group'] = pd.cut(
            df['hh_size'],
            bins=[0, 1, 2, 4, 6, 20],
            labels=['single', 'couple', 'small', 'medium', 'large'],
            include_lowest=True
        )
    
    return pums, atus


def align_marital_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align marital status features."""
    
    # PUMS marital status
    if 'MAR' in pums.columns:
        pums['married'] = (pums['MAR'] == 1).astype(int)
        pums['never_married'] = (pums['MAR'] == 5).astype(int)
        pums['divorced_separated'] = pums['MAR'].isin([3, 4]).astype(int)
        pums['widowed'] = (pums['MAR'] == 2).astype(int)
    else:
        pums['married'] = 0
        pums['never_married'] = 1
        pums['divorced_separated'] = 0
        pums['widowed'] = 0
    
    # ATUS marital status
    if 'marital_cat' in atus.columns:
        atus['married'] = (atus['marital_cat'] == 'married').astype(int)
        atus['never_married'] = (atus['marital_cat'] == 'never_married').astype(int)
        atus['divorced_separated'] = atus['marital_cat'].isin(['divorced', 'separated']).astype(int)
        atus['widowed'] = (atus['marital_cat'] == 'widowed').astype(int)
    elif 'is_married' in atus.columns:
        atus['married'] = atus['is_married'].astype(int)
        atus['never_married'] = 0
        atus['divorced_separated'] = 0
        atus['widowed'] = 0
    else:
        atus['married'] = 0
        atus['never_married'] = 1
        atus['divorced_separated'] = 0
        atus['widowed'] = 0
    
    # Currently single
    pums['currently_single'] = (1 - pums['married']).astype(int)
    atus['currently_single'] = (1 - atus['married']).astype(int)
    
    return pums, atus


def align_children_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align children-related features."""
    
    # PUMS children (from has_children flag or household context)
    if 'has_children' not in pums.columns:
        pums['has_children'] = 0
    else:
        pums['has_children'] = pums['has_children'].astype(int)
    
    # ATUS children
    if 'has_children' not in atus.columns:
        if 'num_children' in atus.columns:
            atus['has_children'] = (atus['num_children'] > 0).astype(int)
        elif 'num_children_hh' in atus.columns:
            atus['has_children'] = (atus['num_children_hh'] > 0).astype(int)
        else:
            atus['has_children'] = 0
    else:
        atus['has_children'] = atus['has_children'].astype(int)
    
    # Young children indicator
    if 'has_young_children' in pums.columns:
        pums['has_young_children'] = pums['has_young_children'].astype(int)
    else:
        pums['has_young_children'] = 0
    
    if 'has_young_children' in atus.columns:
        atus['has_young_children'] = atus['has_young_children'].astype(int)
    else:
        atus['has_young_children'] = 0
    
    # School age children
    if 'has_school_age_children' in pums.columns:
        pums['has_school_children'] = pums['has_school_age_children'].astype(int)
    else:
        pums['has_school_children'] = 0
    
    if 'has_school_age_children' in atus.columns:
        atus['has_school_children'] = atus['has_school_age_children'].astype(int)
    else:
        atus['has_school_children'] = 0
    
    return pums, atus


def align_income_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align income features."""
    
    # PUMS income
    if 'PINCP' in pums.columns:
        pums['has_income'] = (pums['PINCP'] > 0).astype(int)
        pums['income_log'] = np.log1p(pums['PINCP'].clip(lower=0))
        
        # Income brackets
        pums['income_bracket'] = pd.cut(
            pums['PINCP'],
            bins=[-100000, 0, 15000, 30000, 50000, 75000, 100000, 150000, 1000000],
            labels=['negative', 'poverty', 'low', 'lower_mid', 'middle', 'upper_mid', 'high', 'very_high'],
            include_lowest=True
        )
    else:
        pums['has_income'] = 0
        pums['income_log'] = 0
        pums['income_bracket'] = 'middle'
    
    # ATUS income
    if 'income_category' in atus.columns:
        atus['has_income'] = (atus['income_category'] != 'very_low').astype(int)
        
        # Map categories to brackets
        income_map = {
            'very_low': 'poverty',
            'low': 'low',
            'medium': 'middle',
            'high': 'high',
            'very_high': 'very_high'
        }
        atus['income_bracket'] = atus['income_category'].map(income_map).fillna('middle')
        
        # Approximate log income
        income_log_map = {
            'very_low': np.log1p(10000),
            'low': np.log1p(25000),
            'medium': np.log1p(50000),
            'high': np.log1p(85000),
            'very_high': np.log1p(125000)
        }
        atus['income_log'] = atus['income_category'].map(income_log_map).fillna(np.log1p(50000))
    else:
        atus['has_income'] = 0
        atus['income_log'] = np.log1p(50000)
        atus['income_bracket'] = 'middle'
    
    # Poverty indicator
    pums['in_poverty'] = (pums['income_bracket'] == 'poverty').astype(int)
    atus['in_poverty'] = (atus['income_bracket'] == 'poverty').astype(int)
    
    # High income indicator
    pums['high_income'] = pums['income_bracket'].isin(['high', 'very_high']).astype(int)
    atus['high_income'] = atus['income_bracket'].isin(['high', 'very_high']).astype(int)
    
    return pums, atus


def create_time_use_predictors(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create features that predict time use patterns."""
    
    # Work schedule flexibility
    pums['schedule_flexibility'] = (
        pums.get('works_from_home', 0) * 0.5 +
        pums.get('part_time', 0) * 0.3 +
        (1 - pums.get('employed', 0)) * 0.2
    )
    
    atus['schedule_flexibility'] = (
        atus.get('works_from_home', 0) * 0.5 +
        atus.get('part_time', 0) * 0.3 +
        (1 - atus.get('employed', 0)) * 0.2
    )
    
    # Childcare responsibility score
    pums['childcare_responsibility'] = (
        pums.get('has_young_children', 0) * 0.5 +
        pums.get('has_school_children', 0) * 0.3 +
        pums.get('is_female', 0) * 0.1 +
        pums.get('married', 0) * 0.1
    )
    
    atus['childcare_responsibility'] = (
        atus.get('has_young_children', 0) * 0.5 +
        atus.get('has_school_children', 0) * 0.3 +
        atus.get('is_female', 0) * 0.1 +
        atus.get('married', 0) * 0.1
    )
    
    # Leisure time availability
    pums['leisure_availability'] = (
        (1 - pums.get('overtime', 0)) * 0.3 +
        (1 - pums.get('has_young_children', 0)) * 0.3 +
        pums.get('is_retirement_age', 0) * 0.2 +
        pums.get('single_person_hh', 0) * 0.2
    )
    
    atus['leisure_availability'] = (
        (1 - atus.get('overtime', 0)) * 0.3 +
        (1 - atus.get('has_young_children', 0)) * 0.3 +
        atus.get('is_retirement_age', 0) * 0.2 +
        atus.get('single_person_hh', 0) * 0.2
    )
    
    # Commute impact
    if 'JWMNP' in pums.columns:
        pums['commute_impact'] = pums['JWMNP'].fillna(0) / 60  # Convert to hours
    else:
        pums['commute_impact'] = 0
    
    if 'has_long_commute' in atus.columns:
        atus['commute_impact'] = atus['has_long_commute'] * 1.5  # Approximate hours
    else:
        atus['commute_impact'] = 0
    
    # Household role (breadwinner likelihood)
    pums['primary_earner_likely'] = (
        pums.get('full_time', 0) * 0.4 +
        pums.get('is_male', 0) * 0.2 +
        pums.get('high_income', 0) * 0.2 +
        pums.get('age_over_30', 0) * 0.2
    )
    
    atus['primary_earner_likely'] = (
        atus.get('full_time', 0) * 0.4 +
        atus.get('is_male', 0) * 0.2 +
        atus.get('high_income', 0) * 0.2 +
        atus.get('age_over_30', 0) * 0.2
    )
    
    # Time poverty index (likelihood of being time-constrained)
    pums['time_poverty'] = (
        pums.get('overtime', 0) * 0.3 +
        pums.get('has_young_children', 0) * 0.25 +
        pums.get('commute_impact', 0) / 2 * 0.2 +  # Normalize commute to 0-1
        pums.get('single_person_hh', 0) * 0.15 +
        pums.get('in_poverty', 0) * 0.1
    )
    
    atus['time_poverty'] = (
        atus.get('overtime', 0) * 0.3 +
        atus.get('has_young_children', 0) * 0.25 +
        atus.get('commute_impact', 0) / 2 * 0.2 +
        atus.get('single_person_hh', 0) * 0.15 +
        atus.get('in_poverty', 0) * 0.1
    )
    
    return pums, atus


def create_activity_likelihood_scores(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create likelihood scores for different activity types."""
    
    # Work activity likelihood
    pums['work_activity_likely'] = (
        pums.get('employed', 0) * 0.5 +
        pums.get('full_time', 0) * 0.3 +
        pums.get('is_prime_working', 0) * 0.2
    )
    
    atus['work_activity_likely'] = (
        atus.get('employed', 0) * 0.5 +
        atus.get('full_time', 0) * 0.3 +
        atus.get('is_prime_working', 0) * 0.2
    )
    
    # Childcare activity likelihood
    pums['childcare_likely'] = (
        pums.get('has_young_children', 0) * 0.4 +
        pums.get('has_school_children', 0) * 0.2 +
        pums.get('is_female', 0) * 0.2 +
        pums.get('married', 0) * 0.1 +
        (1 - pums.get('employed', 0)) * 0.1
    )
    
    atus['childcare_likely'] = (
        atus.get('has_young_children', 0) * 0.4 +
        atus.get('has_school_children', 0) * 0.2 +
        atus.get('is_female', 0) * 0.2 +
        atus.get('married', 0) * 0.1 +
        (1 - atus.get('employed', 0)) * 0.1
    )
    
    # Household work likelihood
    pums['housework_likely'] = (
        pums.get('is_female', 0) * 0.3 +
        (1 - pums.get('single_person_hh', 0)) * 0.2 +
        pums.get('not_working', 0) * 0.2 +
        pums.get('married', 0) * 0.15 +
        pums.get('age_over_30', 0) * 0.15
    )
    
    atus['housework_likely'] = (
        atus.get('is_female', 0) * 0.3 +
        (1 - atus.get('single_person_hh', 0)) * 0.2 +
        atus.get('not_working', 0) * 0.2 +
        atus.get('married', 0) * 0.15 +
        atus.get('age_over_30', 0) * 0.15
    )
    
    # Leisure activity likelihood
    pums['leisure_likely'] = (
        pums.get('is_retirement_age', 0) * 0.3 +
        pums.get('not_working', 0) * 0.2 +
        (1 - pums.get('has_young_children', 0)) * 0.2 +
        pums.get('single_person_hh', 0) * 0.15 +
        pums.get('is_young_adult', 0) * 0.15
    )
    
    atus['leisure_likely'] = (
        atus.get('is_retirement_age', 0) * 0.3 +
        atus.get('not_working', 0) * 0.2 +
        (1 - atus.get('has_young_children', 0)) * 0.2 +
        atus.get('single_person_hh', 0) * 0.15 +
        atus.get('age_over_18', 0) * 0.15
    )
    
    # Education activity likelihood
    pums['education_likely'] = (
        pums.get('is_college_age', 0) * 0.4 +
        pums.get('is_school_age', 0) * 0.3 +
        pums.get('edu_some_college', 0) * 0.15 +
        pums.get('part_time', 0) * 0.15
    )
    
    atus['education_likely'] = (
        atus.get('is_college_age', 0) * 0.4 +
        atus.get('is_school_age', 0) * 0.3 +
        atus.get('edu_some_college', 0) * 0.15 +
        atus.get('part_time', 0) * 0.15
    )
    
    return pums, atus


def create_composite_indices(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create composite indices combining multiple features."""
    
    # Socioeconomic status index
    pums['ses_index'] = (
        pums.get('has_college', 0) * 0.3 +
        pums.get('high_income', 0) * 0.3 +
        pums.get('white_collar', 0) * 0.2 +
        pums.get('occ_professional', 0) * 0.2
    )
    
    atus['ses_index'] = (
        atus.get('has_college', 0) * 0.3 +
        atus.get('high_income', 0) * 0.3 +
        atus.get('white_collar', 0) * 0.2 +
        atus.get('occ_professional', 0) * 0.2
    )
    
    # Life complexity index
    pums['life_complexity'] = (
        pums.get('has_young_children', 0) * 0.25 +
        pums.get('employed', 0) * 0.2 +
        pums.get('married', 0) * 0.15 +
        pums.get('large_family', 0) * 0.15 +
        pums.get('commute_impact', 0) / 2 * 0.15 +
        pums.get('has_school_children', 0) * 0.1
    )
    
    atus['life_complexity'] = (
        atus.get('has_young_children', 0) * 0.25 +
        atus.get('employed', 0) * 0.2 +
        atus.get('married', 0) * 0.15 +
        atus.get('large_family', 0) * 0.15 +
        atus.get('commute_impact', 0) / 2 * 0.15 +
        atus.get('has_school_children', 0) * 0.1
    )
    
    # Work-life balance index
    pums['work_life_balance'] = (
        (1 - pums.get('overtime', 0)) * 0.3 +
        pums.get('works_from_home', 0) * 0.2 +
        (1 - pums.get('commute_impact', 0) / 2) * 0.2 +
        pums.get('part_time', 0) * 0.15 +
        (1 - pums.get('time_poverty', 0)) * 0.15
    )
    
    atus['work_life_balance'] = (
        (1 - atus.get('overtime', 0)) * 0.3 +
        atus.get('works_from_home', 0) * 0.2 +
        (1 - atus.get('commute_impact', 0) / 2) * 0.2 +
        atus.get('part_time', 0) * 0.15 +
        (1 - atus.get('time_poverty', 0)) * 0.15
    )
    
    # Traditional gender role index
    pums['traditional_role'] = (
        pums.get('married', 0) * 0.25 +
        pums.get('has_children', 0) * 0.25 +
        (pums.get('is_male', 0) * pums.get('full_time', 0)) * 0.25 +
        (pums.get('is_female', 0) * (1 - pums.get('employed', 0))) * 0.25
    )
    
    atus['traditional_role'] = (
        atus.get('married', 0) * 0.25 +
        atus.get('has_children', 0) * 0.25 +
        (atus.get('is_male', 0) * atus.get('full_time', 0)) * 0.25 +
        (atus.get('is_female', 0) * (1 - atus.get('employed', 0))) * 0.25
    )
    
    return pums, atus


def create_interaction_features(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create interaction features between key variables."""
    
    # Age × Employment interactions
    pums['young_employed'] = (pums.get('age_under_30', 0) * pums.get('employed', 0)).astype(int)
    pums['senior_employed'] = (pums.get('is_retirement_age', 0) * pums.get('employed', 0)).astype(int)
    pums['prime_age_not_working'] = (pums.get('is_prime_working', 0) * (1 - pums.get('employed', 0))).astype(int)
    
    atus['young_employed'] = (atus.get('age_under_30', 0) * atus.get('employed', 0)).astype(int)
    atus['senior_employed'] = (atus.get('is_retirement_age', 0) * atus.get('employed', 0)).astype(int)
    atus['prime_age_not_working'] = (atus.get('is_prime_working', 0) * (1 - atus.get('employed', 0))).astype(int)
    
    # Parent × Work interactions
    pums['working_parent'] = (pums.get('has_children', 0) * pums.get('employed', 0)).astype(int)
    pums['stay_home_parent'] = (pums.get('has_children', 0) * (1 - pums.get('employed', 0))).astype(int)
    pums['single_working_parent'] = (
        pums.get('has_children', 0) * 
        pums.get('currently_single', 0) * 
        pums.get('employed', 0)
    ).astype(int)
    
    atus['working_parent'] = (atus.get('has_children', 0) * atus.get('employed', 0)).astype(int)
    atus['stay_home_parent'] = (atus.get('has_children', 0) * (1 - atus.get('employed', 0))).astype(int)
    atus['single_working_parent'] = (
        atus.get('has_children', 0) * 
        atus.get('currently_single', 0) * 
        atus.get('employed', 0)
    ).astype(int)
    
    # Education × Age interactions
    pums['young_college'] = (pums.get('age_under_30', 0) * pums.get('has_college', 0)).astype(int)
    pums['older_no_college'] = (pums.get('age_over_40', 0) * (1 - pums.get('has_college', 0))).astype(int)
    
    atus['young_college'] = (atus.get('age_under_30', 0) * atus.get('has_college', 0)).astype(int)
    atus['older_no_college'] = (atus.get('age_over_40', 0) * (1 - atus.get('has_college', 0))).astype(int)
    
    # Gender × Household interactions
    pums['female_breadwinner'] = (pums.get('is_female', 0) * pums.get('primary_earner_likely', 0))
    pums['male_caregiver'] = (pums.get('is_male', 0) * pums.get('childcare_responsibility', 0))
    
    atus['female_breadwinner'] = (atus.get('is_female', 0) * atus.get('primary_earner_likely', 0))
    atus['male_caregiver'] = (atus.get('is_male', 0) * atus.get('childcare_responsibility', 0))
    
    # Normalize continuous interaction features
    for col in ['female_breadwinner', 'male_caregiver']:
        if col in pums.columns:
            pums[col] = pums[col].clip(0, 1)
        if col in atus.columns:
            atus[col] = atus[col].clip(0, 1)
    
    return pums, atus


def ensure_feature_completeness(pums: pd.DataFrame, atus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both datasets have exactly the same features."""
    
    # Get all feature columns (exclude ID columns)
    pums_features = set(pums.columns) - {'person_id', 'building_id', 'person_idx', 'SERIALNO', 'SPORDER'}
    atus_features = set(atus.columns) - {'case_id', 'template_id', 'activities', 'TUCASEID'}
    
    # Find missing features in each dataset
    missing_in_pums = atus_features - pums_features
    missing_in_atus = pums_features - atus_features
    
    # Add missing features with default values - batch operation to avoid fragmentation
    # Collect all new columns for PUMS
    new_pums_cols = {}
    for col in missing_in_pums:
        if col in atus.columns:
            # Use mode for categorical, median for numeric
            if atus[col].dtype == 'object' or atus[col].dtype.name == 'category':
                default_value = atus[col].mode()[0] if len(atus[col].mode()) > 0 else 'unknown'
            elif atus[col].dtype in ['bool', 'bool_']:
                default_value = False
            else:
                # Convert to numeric first to handle categorical
                numeric_col = pd.to_numeric(atus[col], errors='coerce')
                default_value = numeric_col.median() if numeric_col.notna().any() else 0
            new_pums_cols[col] = default_value
    
    # Add all new columns to PUMS at once
    if new_pums_cols:
        new_pums_df = pd.DataFrame(new_pums_cols, index=pums.index)
        pums = pd.concat([pums, new_pums_df], axis=1)
    
    # Collect all new columns for ATUS
    new_atus_cols = {}
    for col in missing_in_atus:
        if col in pums.columns:
            if pums[col].dtype == 'object' or pums[col].dtype.name == 'category':
                default_value = pums[col].mode()[0] if len(pums[col].mode()) > 0 else 'unknown'
            elif pums[col].dtype in ['bool', 'bool_']:
                default_value = False
            else:
                # Convert to numeric first to handle categorical
                numeric_col = pd.to_numeric(pums[col], errors='coerce')
                default_value = numeric_col.median() if numeric_col.notna().any() else 0
            new_atus_cols[col] = default_value
    
    # Add all new columns to ATUS at once
    if new_atus_cols:
        new_atus_df = pd.DataFrame(new_atus_cols, index=atus.index)
        atus = pd.concat([atus, new_atus_df], axis=1)
    
    # Ensure consistent data types
    common_features = set(pums.columns) & set(atus.columns)
    for col in common_features:
        if pums[col].dtype != atus[col].dtype:
            # Convert to most general type
            if 'int' in str(pums[col].dtype) or 'int' in str(atus[col].dtype):
                pums[col] = pd.to_numeric(pums[col], errors='coerce').fillna(0).astype(float)
                atus[col] = pd.to_numeric(atus[col], errors='coerce').fillna(0).astype(float)
            elif 'float' in str(pums[col].dtype) or 'float' in str(atus[col].dtype):
                pums[col] = pd.to_numeric(pums[col], errors='coerce').fillna(0)
                atus[col] = pd.to_numeric(atus[col], errors='coerce').fillna(0)
            else:
                pums[col] = pums[col].astype(str)
                atus[col] = atus[col].astype(str)
    
    # Create unified blocking keys - batch operation
    # Create blocking keys for both datasets efficiently
    for df, name in [(pums, 'pums'), (atus, 'atus')]:
        block_keys = {}
        
        # Primary blocking key: age group + employment + children
        age_col = df['age_group_simple'] if 'age_group_simple' in df.columns else pd.Series(['adult'] * len(df), index=df.index)
        emp_col = df['employed'] if 'employed' in df.columns else pd.Series([0] * len(df), index=df.index)
        child_col = df['has_children'] if 'has_children' in df.columns else pd.Series([0] * len(df), index=df.index)
        
        block_keys['block_key_primary'] = (
            age_col.astype(str) + '_' +
            emp_col.astype(str) + '_' +
            child_col.astype(str)
        )
        
        # Secondary blocking key: household size + marital status  
        hh_col = df['hh_size_group'] if 'hh_size_group' in df.columns else pd.Series(['small'] * len(df), index=df.index)
        mar_col = df['married'] if 'married' in df.columns else pd.Series([0] * len(df), index=df.index)
        
        block_keys['block_key_secondary'] = (
            hh_col.astype(str) + '_' +
            mar_col.astype(str)
        )
        
        # Tertiary blocking key: SES + work intensity
        ses_col = df['ses_index'] if 'ses_index' in df.columns else pd.Series([0.5] * len(df), index=df.index)
        work_col = df['work_intensity'] if 'work_intensity' in df.columns else pd.Series(['none'] * len(df), index=df.index)
        
        block_keys['block_key_tertiary'] = (
            pd.cut(ses_col, bins=[0, 0.33, 0.67, 1], 
                   labels=['low', 'med', 'high'], include_lowest=True).astype(str) + '_' +
            work_col.astype(str)
        )
        
        # Add all blocking keys at once
        block_df = pd.DataFrame(block_keys, index=df.index)
        if name == 'pums':
            pums = pd.concat([pums, block_df], axis=1)
        else:
            atus = pd.concat([atus, block_df], axis=1)
    
    # Defragment DataFrames after all operations
    pums = pums.copy()
    atus = atus.copy()
    
    logger.info(f"Feature alignment complete. Total features: {len(common_features)}")
    
    return pums, atus


def create_feature_importance_weights() -> Dict[str, float]:
    """
    Define feature importance weights for matching.
    
    Returns:
        Dictionary mapping feature names to importance weights
    """
    weights = {
        # Demographics (high importance)
        'age_group_simple': 2.0,
        'age_group_detailed': 1.8,
        'sex_code': 0.8,
        'is_female': 0.8,
        
        # Employment (very high importance)
        'employed': 2.5,
        'work_intensity': 2.0,
        'full_time': 1.8,
        'part_time': 1.5,
        'not_working': 1.5,
        'work_hours_weekly': 1.5,
        
        # Household (high importance)
        'has_children': 2.0,
        'has_young_children': 2.2,
        'hh_size_group': 1.5,
        'married': 1.5,
        
        # Education (medium importance)
        'has_college': 1.2,
        'education_years': 1.0,
        
        # Time use predictors (very high importance)
        'schedule_flexibility': 2.0,
        'childcare_responsibility': 2.2,
        'time_poverty': 1.8,
        'work_life_balance': 1.8,
        
        # Activity likelihood (high importance)
        'work_activity_likely': 2.0,
        'childcare_likely': 2.0,
        'leisure_likely': 1.5,
        
        # Composite indices (medium-high importance)
        'life_complexity': 1.5,
        'ses_index': 1.2,
        
        # Interactions (medium importance)
        'working_parent': 1.5,
        'young_employed': 1.2,
        
        # Default weight for unlisted features
        '_default': 1.0
    }
    
    return weights