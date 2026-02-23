"""
PUMS data loader for PUMS Enrichment Pipeline.

This module provides functions to load and clean PUMS household and person data
with support for sampling and data standardization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from tqdm import tqdm

from ..utils.config_loader import get_config
from ..utils.data_standardization import standardize_names, standardize_field
from ..utils.logging_setup import log_execution_time, log_memory_usage

logger = logging.getLogger(__name__)


class PUMSDataError(Exception):
    """Raised when there's an issue with PUMS data loading."""
    pass


class PUMSDataLoader:
    """
    Wrapper class for PUMS data loading (for optimization compatibility).
    """
    
    def __init__(self, config: Dict):
        """Initialize PUMS data loader with config."""
        self.config = config
    
    def load_households(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load PUMS household data."""
        return load_pums_households(sample_size)
    
    def load_persons_for_serialnos(self, serialnos: List[str]) -> pd.DataFrame:
        """Load persons for specific SERIALNOs."""
        return load_pums_persons(serialnos)


@log_execution_time(logger)
@log_memory_usage(logger)
def load_pums_households(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and clean PUMS household data.
    
    Args:
        sample_size: Number of households to load (None for all)
        
    Returns:
        DataFrame with cleaned household data
        
    Raises:
        PUMSDataError: If data cannot be loaded or required columns are missing
        ValueError: If invalid parameters provided
    """
    # Input validation
    if sample_size is not None:
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError(f"Sample size must be a positive integer, got: {sample_size}")
    
    config = get_config()
    household_file = config.get_data_path('pums_household')
    household_columns = config.get_phase1_columns('household')
    
    # IMPORTANT: Load both A and B sample files for complete data
    # The config only specifies the A file, so we need to construct the B file path
    household_file_a = Path(household_file)
    household_file_b = household_file_a.parent / household_file_a.name.replace('husa', 'husb')
    
    logger.info(f"Loading PUMS household data from both A and B samples:")
    logger.info(f"  A sample: {household_file_a}")
    logger.info(f"  B sample: {household_file_b}")
    
    # Check if files exist
    if not household_file_a.exists():
        raise PUMSDataError(f"Household data file A not found: {household_file_a}")
    if not household_file_b.exists():
        logger.warning(f"Household data file B not found: {household_file_b} - will use A sample only")
        household_files = [household_file_a]
    else:
        household_files = [household_file_a, household_file_b]
    
    try:
        # Determine sample size
        if sample_size is None:
            sample_size = config.get_sample_size()
        
        # Define data types for memory efficiency
        dtype_dict = {
            'RT': 'category',
            'SERIALNO': 'object',  # Keep as string to preserve leading zeros
            'DIVISION': 'int8',
            'PUMA': 'object',  # Keep as string
            'REGION': 'int8',
            'STATE': 'object',  # Keep as string for FIPS codes
            'NP': 'int8',
            'TYPEHUGQ': 'int8',
            'WGTP': 'int16',
            'HINCP': 'float32',
            'YRBLT': 'float32',
            'BDSP': 'float32',
            'RMSP': 'float32',
            'VEH': 'float32',
            'HFL': 'float32',
            'ELEP': 'float32',
            'GASP': 'float32',
            'FULP': 'float32',
            'BLD': 'float32',  # Building type (units in structure) - float to handle NaN
            'TEN': 'float32',  # Tenure (owned/rented) - float to handle NaN
            # Additional columns that may be needed
            'ACR': 'float32',  # Lot size
            'BATH': 'float32',  # Complete bathrooms - float to handle NaN
            'KIT': 'float32',  # Complete kitchen - float to handle NaN
            'HHT': 'float32',  # Household type - float to handle NaN
            'GRPIP': 'float32',  # Gross rent as percentage of income
            'OCPIP': 'float32',  # Owner costs as percentage of income
            'R18': 'float32',  # Presence of persons under 18 - float to handle NaN
            'R60': 'float32',  # Presence of persons 60+ - float to handle NaN
            'R65': 'float32'  # Presence of persons 65+ - float to handle NaN
        }
        
        # Filter to only columns we need
        usecols = [col for col in household_columns if col in dtype_dict.keys() or col.startswith('WGTP')]
        
        # Load data from all files
        if sample_size:
            logger.info(f"Loading sample of {sample_size} households")
            # Read in chunks to handle sampling
            chunk_size = min(10000, sample_size)
            chunks = []
            total_read = 0
            
            for file_path in household_files:
                if total_read >= sample_size:
                    break
                    
                for chunk in pd.read_csv(file_path, 
                                       usecols=usecols,
                                       dtype=dtype_dict,
                                       chunksize=chunk_size,
                                       low_memory=False):
                    # Filter to housing units only (RT='H' and TYPEHUGQ=1)
                    chunk = chunk[(chunk['RT'] == 'H') & (chunk['TYPEHUGQ'] == 1)]
                    
                    if len(chunk) > 0:
                        remaining = sample_size - total_read
                        if remaining <= 0:
                            break
                        
                        if len(chunk) > remaining:
                            chunk = chunk.head(remaining)
                        
                        chunks.append(chunk)
                        total_read += len(chunk)
                        
                        if total_read >= sample_size:
                            break
            
            if not chunks:
                raise PUMSDataError("No valid housing units found in data")
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            logger.info("Loading all household data from all samples")
            dfs = []
            for file_path in household_files:
                logger.info(f"  Loading: {file_path.name}")
                df_temp = pd.read_csv(file_path,
                               usecols=usecols,
                               dtype=dtype_dict,
                               low_memory=False)
                # Filter to housing units only
                df_temp = df_temp[(df_temp['RT'] == 'H') & (df_temp['TYPEHUGQ'] == 1)]
                dfs.append(df_temp)
                logger.info(f"    Loaded {len(df_temp)} households from {file_path.name}")
            
            df = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Loaded {len(df)} household records")
        
        # Add any missing columns with appropriate defaults
        for col in household_columns:
            if col not in df.columns:
                if col.startswith('WGTP'):
                    df[col] = df['WGTP']  # Use base weight as default
                else:
                    df[col] = np.nan
        
        # Ensure SERIALNO is string type
        df['SERIALNO'] = df['SERIALNO'].astype(str)
        
        # Create building_id (will be unique identifier going forward)
        df['building_id'] = 'BLDG_' + df['SERIALNO']
        
        # Data quality checks
        required_cols = ['SERIALNO', 'STATE', 'PUMA', 'NP', 'WGTP']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise PUMSDataError(f"Required columns missing: {missing_cols}")
        
        # Check for duplicates
        duplicate_count = df['SERIALNO'].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate SERIALNO values, keeping first occurrence")
            df = df.drop_duplicates(subset=['SERIALNO'], keep='first')
        
        # Log summary statistics
        logger.info(f"Household data summary:")
        logger.info(f"  - Total households: {len(df)}")
        logger.info(f"  - Average household size: {df['NP'].mean():.2f}")
        logger.info(f"  - States represented: {df['STATE'].nunique()}")
        logger.info(f"  - Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        raise PUMSDataError(f"Error loading household data: {str(e)}")


@log_execution_time(logger)
@log_memory_usage(logger)
def load_persons_for_serialnos(household_serials: List[str]) -> pd.DataFrame:
    """
    Load persons for specific SERIALNOs (for parallel processing).
    
    This is an alias for load_pums_persons to support the new parallel API.
    """
    return load_pums_persons(household_serials)


def load_pums_persons(household_serials: List[str]) -> pd.DataFrame:
    """
    Load and clean PUMS person data for specified households.
    
    Args:
        household_serials: List of SERIALNO values to load persons for
        
    Returns:
        DataFrame with cleaned person data
        
    Raises:
        PUMSDataError: If data cannot be loaded
        ValueError: If invalid parameters provided
    """
    # Input validation
    if not isinstance(household_serials, list):
        raise ValueError(f"household_serials must be a list, got {type(household_serials)}")
    
    if not household_serials:
        raise ValueError("household_serials list is empty")
    
    # Check that all serials are strings
    non_string_serials = [s for s in household_serials if not isinstance(s, str)]
    if non_string_serials:
        raise ValueError(f"All household serials must be strings, found non-string values: {non_string_serials[:5]}")
    
    config = get_config()
    person_file = config.get_data_path('pums_person')
    person_columns = config.get_phase1_columns('person')
    
    # IMPORTANT: Load both A and B sample files for complete data
    # The config only specifies the A file, so we need to construct the B file path
    person_file_a = Path(person_file)
    person_file_b = person_file_a.parent / person_file_a.name.replace('pusa', 'pusb')
    
    logger.info(f"Loading PUMS person data from both A and B samples:")
    logger.info(f"  A sample: {person_file_a}")
    logger.info(f"  B sample: {person_file_b}")
    
    # Check if files exist
    if not person_file_a.exists():
        raise PUMSDataError(f"Person data file A not found: {person_file_a}")
    if not person_file_b.exists():
        logger.warning(f"Person data file B not found: {person_file_b} - will use A sample only")
        person_files = [person_file_a]
    else:
        person_files = [person_file_a, person_file_b]
    
    # Convert to set for faster lookup
    serial_set = set(household_serials)
    
    try:
        # Define data types - optimize for memory usage
        dtype_dict = {
            'RT': 'category',
            'SERIALNO': 'object',
            'SPORDER': 'int8',
            'DIVISION': 'int8',
            'PUMA': 'object',
            'REGION': 'int8',
            'STATE': 'object',
            'PWGTP': 'int16',
            'AGEP': 'int8',
            'SEX': 'int8',
            'MAR': 'int8',
            'SCHL': 'float32',
            'ESR': 'float32',
            'PINCP': 'float32',
            'POVPIP': 'float32'
        }
        
        # Add all weight columns to dtype dict for efficiency
        for i in range(1, 81):
            dtype_dict[f'PWGTP{i}'] = 'int16'
        
        # Filter to columns we need
        usecols = [col for col in person_columns if col in dtype_dict.keys()]
        
        # Load persons in chunks and filter
        chunks = []
        total_persons = 0
        households_found = set()

        logger.info(f"Loading persons for {len(serial_set)} households")

        # Adjust chunk size based on sample size for efficiency (more conservative to avoid OOM)
        if len(serial_set) < 100:
            chunk_size = 8000   # Smaller chunks for tiny sets
        elif len(serial_set) < 1000:
            chunk_size = 16000  # Medium chunks
        else:
            chunk_size = 32000  # Conservative large for full data
        min_chunk_size = 4000

        # For small samples, implement early stopping
        early_stop = len(serial_set) < 100
        expected_persons_per_household = 3  # Typical household size
        expected_total_persons = len(serial_set) * expected_persons_per_household

        with tqdm(desc="Loading person records", unit="chunks") as pbar:
            for file_path in person_files:
                logger.info(f"  Processing: {file_path.name}")
                # Robust chunked reading with adaptive fallback on memory/tokenizer errors
                engine = 'c'
                while True:
                    try:
                        for chunk in pd.read_csv(
                            file_path,
                            usecols=usecols,
                            dtype=dtype_dict,
                            chunksize=chunk_size,
                            low_memory=False,
                            engine=engine
                        ):
                            # Filter to person records first to reduce memory
                            chunk = chunk[chunk['RT'] == 'P']
                            if len(chunk) == 0:
                                pbar.update(1)
                                continue
                            # Convert SERIALNO to str once and filter to target households
                            chunk['SERIALNO'] = chunk['SERIALNO'].astype(str)
                            chunk = chunk[chunk['SERIALNO'].isin(serial_set)]
                            if len(chunk) > 0:
                                chunks.append(chunk)
                                total_persons += len(chunk)
                                households_found.update(chunk['SERIALNO'].unique())
                            pbar.update(1)
                            # Early stopping for small samples
                            if early_stop and len(households_found) >= len(serial_set):
                                if households_found == serial_set or total_persons >= expected_total_persons * 1.5:
                                    logger.info(f"Early stopping: found all {len(households_found)} households")
                                    break
                        break  # finished this file successfully
                    except MemoryError as me:
                        # Reduce chunk size and retry, switch engine if needed
                        if chunk_size > min_chunk_size:
                            old = chunk_size
                            chunk_size = max(min_chunk_size, chunk_size // 2)
                            logger.warning(f"MemoryError while reading {file_path.name}: reducing chunk size {old} -> {chunk_size} and retrying")
                            continue
                        if engine == 'c':
                            engine = 'python'
                            logger.warning(f"Switching CSV engine to 'python' for {file_path.name} due to memory issues")
                            continue
                        raise
                    except pd.errors.ParserError as pe:
                        # Tokenizing OOM or parse error: decrease chunk size or switch engine
                        if 'out of memory' in str(pe).lower() and chunk_size > min_chunk_size:
                            old = chunk_size
                            chunk_size = max(min_chunk_size, chunk_size // 2)
                            logger.warning(f"ParserError OOM while reading {file_path.name}: reducing chunk size {old} -> {chunk_size} and retrying")
                            continue
                        if engine == 'c':
                            engine = 'python'
                            logger.warning(f"ParserError for {file_path.name}: switching engine to 'python' and retrying")
                            continue
                        raise
                
                # Break outer loop if early stopping triggered
                if early_stop and len(households_found) >= len(serial_set):
                    if households_found == serial_set or total_persons >= expected_total_persons * 1.5:
                        break
        
        if not chunks:
            raise PUMSDataError("No persons found for specified households")
        
        df = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"Loaded {len(df)} person records")
        
        # Add any missing columns
        for col in person_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Create person_id
        df['person_id'] = df['SERIALNO'].astype(str) + '_P' + df['SPORDER'].astype(str).str.zfill(2)
        
        # Standardize names if name columns exist (for future phases)
        # Note: PUMS doesn't have actual names, but we prepare the structure
        # These fields are created empty to maintain compatibility with Phase 2-3
        # matching infrastructure which expects standardized name fields
        df['name_standardized'] = ''  # Will be used in Phase 2-3 for RECS/ATUS matching
        df['name_first'] = ''
        df['name_last'] = ''
        df['name_soundex'] = ''
        
        # Data quality checks
        persons_per_household = df.groupby('SERIALNO').size()
        logger.info(f"Person data summary:")
        logger.info(f"  - Total persons: {len(df)}")
        logger.info(f"  - Households with persons: {len(persons_per_household)}")
        logger.info(f"  - Average persons per household: {persons_per_household.mean():.2f}")
        logger.info(f"  - Age range: {df['AGEP'].min()} - {df['AGEP'].max()}")
        
        # Check for missing households
        missing_households = serial_set - set(df['SERIALNO'].unique())
        if missing_households:
            logger.warning(f"No persons found for {len(missing_households)} households")
        
        return df
        
    except Exception as e:
        raise PUMSDataError(f"Error loading person data: {str(e)}")


def validate_pums_data(households: pd.DataFrame, persons: pd.DataFrame) -> Dict[str, any]:
    """
    Validate PUMS household and person data.
    
    Args:
        households: Household DataFrame
        persons: Person DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check household data
    if len(households) == 0:
        validation_results['valid'] = False
        validation_results['errors'].append("No household data loaded")
    
    # Check person data
    if len(persons) == 0:
        validation_results['valid'] = False
        validation_results['errors'].append("No person data loaded")
    
    # Check for required columns
    required_household_cols = ['SERIALNO', 'building_id', 'STATE', 'PUMA', 'NP', 'WGTP']
    missing_household_cols = [col for col in required_household_cols if col not in households.columns]
    if missing_household_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing household columns: {missing_household_cols}")
    
    required_person_cols = ['SERIALNO', 'person_id', 'SPORDER', 'AGEP', 'PWGTP']
    missing_person_cols = [col for col in required_person_cols if col not in persons.columns]
    if missing_person_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing person columns: {missing_person_cols}")
    
    if validation_results['valid']:
        # Check data consistency
        household_serials = set(households['SERIALNO'].unique())
        person_serials = set(persons['SERIALNO'].unique())
        
        # Households without persons
        households_without_persons = household_serials - person_serials
        if households_without_persons:
            validation_results['warnings'].append(
                f"{len(households_without_persons)} households have no person records"
            )
        
        # Check NP (number of persons) consistency
        actual_persons = persons.groupby('SERIALNO').size()
        household_np = households.set_index('SERIALNO')['NP']
        
        mismatched = []
        for serial in actual_persons.index:
            if serial in household_np.index:
                expected = household_np[serial]
                actual = actual_persons[serial]
                if expected != actual:
                    mismatched.append((serial, expected, actual))
        
        if mismatched:
            validation_results['warnings'].append(
                f"{len(mismatched)} households have mismatched person counts"
            )
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_households': len(households),
            'total_persons': len(persons),
            'households_with_persons': len(person_serials),
            'avg_household_size': persons.groupby('SERIALNO').size().mean(),
            'avg_age': persons['AGEP'].mean(),
            'total_population_weight': persons['PWGTP'].sum()
        }
    
    return validation_results


def create_household_roster(households: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    """
    Create a household roster with person counts by demographic.
    
    Args:
        households: Household DataFrame
        persons: Person DataFrame
        
    Returns:
        DataFrame with household roster information
        
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
    
    # Check required columns
    if 'SERIALNO' not in households.columns:
        raise ValueError("Households DataFrame missing required column: SERIALNO")
    
    if 'SERIALNO' not in persons.columns or 'AGEP' not in persons.columns:
        raise ValueError("Persons DataFrame missing required columns: SERIALNO or AGEP")
    
    # Import safe_cut at the top of the function if not already imported
    from ..utils.dtype_utils import safe_cut
    
    # Create age categories using safe_cut to ensure string dtype
    persons['age_cat'] = safe_cut(persons['AGEP'], 
                               bins=[0, 5, 17, 64, 120],
                               labels=['under_5', 'child_5_17', 'adult_18_64', 'senior_65_plus'])
    
    # Create summary by household
    roster = persons.groupby(['SERIALNO', 'age_cat'], observed=True).size().unstack(fill_value=0)
    roster['total_persons'] = roster.sum(axis=1)
    
    # Add sex breakdown
    sex_counts = persons.groupby(['SERIALNO', 'SEX'], observed=True).size().unstack(fill_value=0)
    sex_counts.columns = ['male_count', 'female_count']
    
    roster = roster.join(sex_counts)
    
    # Add employment status for adults
    employed = persons[persons['ESR'].isin([1, 2])].groupby('SERIALNO').size()
    roster['employed_count'] = employed.reindex(roster.index, fill_value=0)
    
    # Reset index to make SERIALNO a column for easier merging
    roster = roster.reset_index()
    
    return roster


if __name__ == "__main__":
    # Test data loading with small sample
    try:
        logger.info("Testing PUMS data loader with 10 households")
        
        # Load 10 households
        households = load_pums_households(sample_size=10)
        print(f"\nLoaded {len(households)} households")
        print(f"Columns: {list(households.columns[:10])}...")  # First 10 columns
        
        # Load persons for these households
        household_serials = households['SERIALNO'].tolist()
        persons = load_pums_persons(household_serials)
        print(f"\nLoaded {len(persons)} persons")
        print(f"Columns: {list(persons.columns[:10])}...")
        
        # Validate data
        validation_results = validate_pums_data(households, persons)
        print(f"\nValidation results:")
        print(f"Valid: {validation_results['valid']}")
        print(f"Errors: {validation_results['errors']}")
        print(f"Warnings: {validation_results['warnings']}")
        print(f"Statistics: {validation_results['statistics']}")
        
        # Create roster
        roster = create_household_roster(households, persons)
        print(f"\nHousehold roster shape: {roster.shape}")
        print(f"Roster columns: {list(roster.columns)}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise