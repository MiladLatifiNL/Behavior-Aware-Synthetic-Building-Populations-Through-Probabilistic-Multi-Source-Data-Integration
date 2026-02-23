"""
ATUS (American Time Use Survey) data loader for Phase 3.

This module loads and processes REAL ATUS activity diary data for probabilistic
matching with PUMS persons to assign realistic daily activity patterns.

NO SYNTHETIC DATA - Uses only real ATUS 2023 survey data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_atus_respondent_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS respondent data for Phase 3 matching.
    
    Args:
        atus_path: Path to ATUS data directory (default: data/raw/atus/2023)
        
    Returns:
        DataFrame with ATUS respondent data
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    # Validate base path exists
    if not atus_path.exists():
        raise FileNotFoundError(f"ATUS base directory not found: {atus_path}")
    
    # Check for subdirectory structure with more flexible logic
    respondent_subdir = None
    
    # First check if there's a subdirectory with 'resp' in the name
    possible_subdirs = [
        atus_path / "atusresp-2023",
        atus_path / "atusresp_2023", 
        atus_path / "respondent",
        atus_path / "resp"
    ]
    
    # Also check for any directory containing 'resp' in its name
    if respondent_subdir is None:
        for item in atus_path.iterdir():
            if item.is_dir() and 'resp' in item.name.lower():
                respondent_subdir = item
                logger.info(f"Found respondent directory: {respondent_subdir}")
                break
    
    # Try the explicit list if dynamic search failed
    if respondent_subdir is None:
        for alt_dir in possible_subdirs:
            if alt_dir.exists():
                respondent_subdir = alt_dir
                break
    
    # If still not found, check if files are directly in atus_path
    if respondent_subdir is None:
        # Check if respondent files are directly in the base directory
        direct_files = list(atus_path.glob("*resp*.dat")) + list(atus_path.glob("*resp*.csv"))
        if direct_files:
            respondent_subdir = atus_path
            logger.info(f"Found respondent files directly in: {atus_path}")
        else:
            raise FileNotFoundError(f"ATUS respondent subdirectory not found in {atus_path}")
    
    # Look for respondent file with multiple possible extensions
    possible_files = [
        respondent_subdir / "atusresp_2023.dat",
        respondent_subdir / "atusresp_2023.csv",
        respondent_subdir / "atusresp-2023.dat",
        respondent_subdir / "atusresp-2023.csv"
    ]
    
    respondent_file = None
    for file_path in possible_files:
        if file_path.exists():
            respondent_file = file_path
            break
    
    if respondent_file is None:
        raise FileNotFoundError(f"ATUS respondent file not found in {respondent_subdir}. Looked for: {[f.name for f in possible_files]}")
    
    logger.info(f"Loading ATUS respondent data from {respondent_file}")
    
    # Read the CSV file (though named .dat, it's CSV format)
    df = pd.read_csv(respondent_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} ATUS respondents with {len(df.columns)} variables")
    
    # Rename key variables for consistency
    column_mapping = {
        'TUCASEID': 'case_id',
        'TUYEAR': 'year',
        'TUMONTH': 'month',
        'TEAGE': 'age',  # Age is not in respondent file, it's in summary
        'TESEX': 'sex',  # Sex is not in respondent file, it's in summary
        'TELFS': 'labor_force_status',
        'TEMJOT': 'multiple_jobs',
        'TRDPFTPT': 'full_part_time',
        'TESCHENR': 'school_enrollment',
        'TESCHLVL': 'school_level',
        'TRSPPRES': 'spouse_present',
        'TESPEMPNOT': 'spouse_employed',
        'TRCHILDNUM': 'num_children',
        'TEHRUSLT': 'usual_hours_worked',
        'TRHOLIDAY': 'holiday',
        'TRDTOCC1': 'occupation_code',
        'TRDTIND1': 'industry_code',
        'TRERNWA': 'weekly_earnings'
    }
    
    # Only rename columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    return df


def load_atus_summary_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS summary data with time use aggregations.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with ATUS summary data including time use variables
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    # Validate base path exists
    if not atus_path.exists():
        raise FileNotFoundError(f"ATUS base directory not found: {atus_path}")
    
    # Check for subdirectory structure
    summary_subdir = atus_path / "atussum-2023"
    if not summary_subdir.exists():
        # Try alternate naming conventions
        alt_subdirs = [
            atus_path / "atussum_2023",
            atus_path / "summary",
            atus_path / "sum"
        ]
        for alt_dir in alt_subdirs:
            if alt_dir.exists():
                summary_subdir = alt_dir
                break
        else:
            raise FileNotFoundError(f"ATUS summary subdirectory not found. Expected: {summary_subdir}")
    
    # Look for summary file with multiple possible extensions
    possible_files = [
        summary_subdir / "atussum_2023.dat",
        summary_subdir / "atussum_2023.csv",
        summary_subdir / "atussum-2023.dat",
        summary_subdir / "atussum-2023.csv"
    ]
    
    summary_file = None
    for file_path in possible_files:
        if file_path.exists():
            summary_file = file_path
            break
    
    if summary_file is None:
        raise FileNotFoundError(f"ATUS summary file not found in {summary_subdir}. Looked for: {[f.name for f in possible_files]}")
    
    logger.info(f"Loading ATUS summary data from {summary_file}")
    
    df = pd.read_csv(summary_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} ATUS summary records with {len(df.columns)} variables")
    
    # The summary file has demographics AND time use variables
    # Time variables are like t010101 (sleeping), t010201 (personal care), etc.
    
    # Rename demographic variables
    demo_mapping = {
        'TUCASEID': 'case_id',
        'TUFINLWGT': 'final_weight',
        'TRYHHCHILD': 'hh_child_status',
        'TEAGE': 'age',
        'TESEX': 'sex',
        'PEEDUCA': 'education',
        'PTDTRACE': 'race',
        'PEHSPNON': 'hispanic',
        'GTMETSTA': 'metro_status',
        'TELFS': 'labor_force_status',
        'TEMJOT': 'multiple_jobs',
        'TRDPFTPT': 'full_part_time',
        'TESCHENR': 'school_enrollment',
        'TESCHLVL': 'school_level',
        'TRSPPRES': 'spouse_present',
        'TESPEMPNOT': 'spouse_employed',
        'TRERNWA': 'weekly_earnings',
        'TRCHILDNUM': 'num_children',
        'TEHRUSLT': 'usual_hours_worked',
        'TUDIARYDAY': 'diary_day',
        'TRHOLIDAY': 'holiday'
    }
    
    existing_cols = {k: v for k, v in demo_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    # Calculate aggregated time use categories (in minutes)
    # Based on ATUS activity codes:
    time_categories = {
        'sleep_time': ['t010101', 't010102'],  # Sleeping
        'personal_care': ['t010201', 't010299', 't010301', 't010399'],  # Grooming, health care
        'eating': ['t110101', 't110199', 't110201', 't110299'],  # Eating and drinking
        'household_work': [col for col in df.columns if col.startswith('t02')],  # Household activities
        'caring_household': [col for col in df.columns if col.startswith('t03')],  # Caring for HH members
        'caring_nonhh': [col for col in df.columns if col.startswith('t04')],  # Caring for non-HH
        'work': [col for col in df.columns if col.startswith('t05')],  # Work and work-related
        'education': [col for col in df.columns if col.startswith('t06')],  # Education
        'shopping': [col for col in df.columns if col.startswith('t07') or col.startswith('t08')],  # Consumer purchases, services
        'professional_services': [col for col in df.columns if col.startswith('t09')],  # Professional services
        'leisure': [col for col in df.columns if col.startswith('t12') or col.startswith('t13')],  # Socializing, leisure, sports
        'civic': [col for col in df.columns if col.startswith('t14') or col.startswith('t15')],  # Volunteer, religious
        'travel': [col for col in df.columns if col.startswith('t18')],  # Travel
    }
    
    for category, cols in time_categories.items():
        valid_cols = [col for col in cols if col in df.columns]
        if valid_cols:
            df[category] = df[valid_cols].sum(axis=1)
        else:
            df[category] = 0
    
    return df


def load_atus_activity_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS activity-level data.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with detailed activity records
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    activity_file = atus_path / "atusact-2023" / "atusact_2023.dat"
    
    if not activity_file.exists():
        raise FileNotFoundError(f"ATUS activity file not found: {activity_file}")
    
    logger.info(f"Loading ATUS activity data from {activity_file}")
    
    df = pd.read_csv(activity_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} activity records")
    
    # Rename key columns
    column_mapping = {
        'TUCASEID': 'case_id',
        'TUACTIVITY_N': 'activity_number',
        'TUACTDUR24': 'duration_minutes',
        'TUSTARTTIM': 'start_time',
        'TUSTOPTIME': 'stop_time',
        'TRCODEP': 'activity_code',
        'TRTIER1P': 'tier1_code',
        'TRTIER2P': 'tier2_code',
        'TUWHERE': 'location',
        'TUTIER1CODE': 'activity_tier1',
        'TUTIER2CODE': 'activity_tier2',
        'TUTIER3CODE': 'activity_tier3'
    }
    
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    return df


def load_atus_roster_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS household roster data.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with household member information
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    roster_file = atus_path / "atusrost-2023" / "atusrost_2023.dat"
    
    if not roster_file.exists():
        raise FileNotFoundError(f"ATUS roster file not found: {roster_file}")
    
    logger.info(f"Loading ATUS roster data from {roster_file}")
    
    df = pd.read_csv(roster_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} roster records")
    
    # This contains household composition information
    column_mapping = {
        'TUCASEID': 'case_id',
        'TULINENO': 'line_number',
        'TERRP': 'relationship',
        'TEAGE': 'age',
        'TESEX': 'sex'
    }
    
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    return df


def load_atus_cps_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS-CPS data with household characteristics.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with CPS household data
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    cps_file = atus_path / "atuscps-2023" / "atuscps_2023.dat"
    
    if not cps_file.exists():
        raise FileNotFoundError(f"ATUS CPS file not found: {cps_file}")
    
    logger.info(f"Loading ATUS-CPS data from {cps_file}")
    
    df = pd.read_csv(cps_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} CPS records")
    
    return df


def load_atus_weights_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load ATUS replicate weights data.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with survey weights
        
    Raises:
        FileNotFoundError: If ATUS data files not found
    """
    if atus_path is None:
        atus_path = Path("data/raw/atus/2023")
    
    weights_file = atus_path / "atuswgts-2023" / "atuswgts_2023.dat"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"ATUS weights file not found: {weights_file}")
    
    logger.info(f"Loading ATUS weights data from {weights_file}")
    
    df = pd.read_csv(weights_file, low_memory=False)
    
    logger.info(f"Loaded {len(df)} weight records")
    
    return df


def load_all_atus_data(atus_path: Path = None) -> pd.DataFrame:
    """
    Load and merge all ATUS data sources into a comprehensive dataset.
    
    This function loads all 10 ATUS data files and merges them appropriately
    to create a complete picture of each respondent's demographics, household,
    and time use patterns.
    
    Args:
        atus_path: Path to ATUS data directory
        
    Returns:
        DataFrame with complete ATUS data for matching
        
    Raises:
        FileNotFoundError: If any required ATUS data files not found
    """
    logger.info("Loading all ATUS data sources (10 files)")
    
    # Load primary datasets
    respondents = load_atus_respondent_data(atus_path)
    summary = load_atus_summary_data(atus_path)
    activities = load_atus_activity_data(atus_path)
    roster = load_atus_roster_data(atus_path)
    weights = load_atus_weights_data(atus_path)
    
    logger.info("Merging ATUS datasets...")
    
    # Start with summary data (has both demographics and time use)
    atus_data = summary.copy()
    
    # Add respondent-specific data
    resp_cols = [col for col in respondents.columns if col not in atus_data.columns or col == 'case_id']
    atus_data = atus_data.merge(respondents[resp_cols], on='case_id', how='left')
    
    # Add household composition from roster
    # Calculate household size and composition metrics
    household_comp = roster.groupby('case_id').agg({
        'line_number': 'count',  # household size
        'age': ['min', 'max', 'mean'],  # age distribution
        'sex': lambda x: (x == 2).sum()  # number of females
    }).reset_index()
    household_comp.columns = ['case_id', 'hh_size', 'min_age', 'max_age', 'mean_age', 'num_females']
    household_comp['has_children'] = household_comp['min_age'] < 18
    household_comp['has_elderly'] = household_comp['max_age'] >= 65
    
    atus_data = atus_data.merge(household_comp, on='case_id', how='left')
    
    # Add activity diversity metrics
    agg_dict = {
        'activity_number': 'count',  # number of activities
        'duration_minutes': ['sum', 'mean', 'std']  # duration statistics
    }
    
    # Only add location if it exists
    if 'location' in activities.columns:
        agg_dict['location'] = lambda x: x.nunique()  # location diversity
    
    activity_diversity = activities.groupby('case_id').agg(agg_dict).reset_index()
    
    # Flatten column names
    activity_diversity.columns = ['case_id', 'num_activities', 'total_time', 'mean_duration', 'std_duration'] + \
                                 (['num_locations'] if 'location' in activities.columns else [])
    
    atus_data = atus_data.merge(activity_diversity, on='case_id', how='left')
    
    # Add final weights
    weight_cols = ['case_id', 'TUFINLWGT'] if 'TUFINLWGT' in weights.columns else ['case_id']
    if len(weight_cols) > 1:
        atus_data = atus_data.merge(weights[weight_cols], on='case_id', how='left', suffixes=('', '_wgt'))
    
    # Create standardized features for matching
    atus_data = standardize_atus_features(atus_data)
    
    logger.info(f"Created comprehensive ATUS dataset with {len(atus_data)} respondents and {len(atus_data.columns)} features")
    
    return atus_data


def standardize_atus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize ATUS features for matching with PUMS data.
    
    Args:
        df: Raw ATUS data
        
    Returns:
        DataFrame with standardized features
    """
    logger.info("Standardizing ATUS features for matching")
    
    # Add age groups matching PUMS categories
    from ..utils.dtype_utils import safe_cut
    df['age_group'] = safe_cut(df['age'].fillna(35), 
                             bins=[0, 18, 25, 35, 45, 55, 65, 100],
                             labels=['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    
    # Employment status standardization
    df['employed'] = df['labor_force_status'].isin([1, 2]) if 'labor_force_status' in df.columns else False
    
    # Education level standardization (PEEDUCA to standard categories)
    if 'education' in df.columns:
        edu_map = {
            31: 'less_than_hs',
            32: 'less_than_hs',
            33: 'less_than_hs',
            34: 'less_than_hs',
            35: 'less_than_hs',
            36: 'less_than_hs',
            37: 'less_than_hs',
            38: 'less_than_hs',
            39: 'high_school',
            40: 'some_college',
            41: 'some_college',
            42: 'some_college',
            43: 'bachelors',
            44: 'graduate',
            45: 'graduate',
            46: 'graduate'
        }
        df['education_level'] = df['education'].map(edu_map).fillna('high_school')
    
    # Household type
    df['household_type'] = 'other'
    df.loc[df['hh_size'] == 1, 'household_type'] = 'single'
    df.loc[(df['hh_size'] == 2) & (df.get('spouse_present', 0) == 1), 'household_type'] = 'couple'
    df.loc[df.get('has_children', False) == True, 'household_type'] = 'family_with_children'
    
    # Work intensity
    if 'usual_hours_worked' in df.columns:
        df['work_intensity'] = safe_cut(df['usual_hours_worked'].fillna(0),
                                      bins=[-1, 0, 20, 35, 45, 100],
                                      labels=['not_working', 'part_time_low', 'part_time_high', 'full_time', 'overtime'])
    
    # Create unique template ID
    df['template_id'] = 'atus_' + df['case_id'].astype(str)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['age', 'sex', 'hh_size', 'num_children', 'usual_hours_worked',
                    'sleep_time', 'work', 'household_work', 'leisure', 'caring_household']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def create_activity_templates(atus_data: pd.DataFrame, num_templates: Optional[int] = None) -> pd.DataFrame:
    """
    Create activity templates from real ATUS data for matching.
    
    Args:
        atus_data: Complete ATUS dataset from load_all_atus_data()
        num_templates: Optional limit on number of templates (default: all)
        
    Returns:
        DataFrame with activity templates ready for matching
    """
    logger.info("Creating activity templates from real ATUS data")
    
    templates = atus_data.copy()
    
    # Select key features for templates
    template_features = [
        'template_id', 'case_id', 'age', 'sex', 'education_level',
        'employed', 'work_intensity', 'household_type', 'hh_size',
        'has_children', 'num_children', 'usual_hours_worked',
        'sleep_time', 'personal_care', 'eating', 'household_work',
        'caring_household', 'work', 'education', 'shopping',
        'leisure', 'civic', 'travel', 'num_activities',
        'final_weight'
    ]
    
    # Keep only available features
    available_features = [f for f in template_features if f in templates.columns]
    templates = templates[available_features]
    
    # Apply limit if specified
    if num_templates and num_templates < len(templates):
        # Sample weighted by survey weights if available
        if 'final_weight' in templates.columns:
            templates = templates.sample(n=num_templates, weights='final_weight', random_state=42)
        else:
            templates = templates.sample(n=num_templates, random_state=42)
    
    logger.info(f"Created {len(templates)} activity templates from real ATUS respondents")
    
    return templates


# Remove all synthetic data generation functions - NO LONGER NEEDED
# We only use real ATUS data now