"""
Phase 1 PUMS Integration - Merge households with persons.

This module implements the core logic for Phase 1 of the PUMS Enrichment Pipeline,
creating a foundation dataset of buildings with their occupants.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

from ..utils.config_loader import get_config
from ..utils.logging_setup import setup_logging, log_execution_time, log_memory_usage, create_performance_summary
from ..data_loading.pums_loader import load_pums_households, load_pums_persons, validate_pums_data, create_household_roster
from ..utils.feature_engineering import create_household_features, create_person_features, create_matching_features, create_energy_profile_features
from ..utils.enhanced_feature_engineering import create_comprehensive_matching_features
from ..validation.phase_validators import validate_phase1_output
from ..validation.report_generator import generate_phase1_report

# Set up phase-specific logger
logger = setup_logging('phase1')


class Phase1ProcessingError(Exception):
    """Raised when Phase 1 processing fails."""
    pass


@log_execution_time(logger)
@log_memory_usage(logger)
def process_phase1(sample_size: Optional[int] = None, validate: bool = True,
                  input_data: Optional[pd.DataFrame] = None,
                  save_output: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute Phase 1 processing: merge PUMS households with persons.
    
    Args:
        sample_size: Number of households to process (None for config default)
    validate: Whether to run validation and generate report
    input_data: Optional pre-loaded household data (for streaming mode)
    save_output: When True (default), writes phase outputs to disk. In streaming
             orchestrations, set False to avoid overwriting partial outputs
             per batch; the caller should persist the aggregated result once.
        
    Returns:
        Tuple of (merged_data, processing_metadata)
        
    Raises:
        Phase1ProcessingError: If processing fails
        ValueError: If invalid parameters provided
    """
    # Input validation
    if sample_size is not None:
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError(f"Sample size must be a positive integer, got: {sample_size}")
    
    if not isinstance(validate, bool):
        raise ValueError(f"validate must be a boolean, got: {type(validate)}")
    
    logger.info("=" * 80)
    logger.info("STARTING PHASE 1: PUMS HOUSEHOLD-PERSON INTEGRATION")
    logger.info("=" * 80)
    
    config = get_config()
    start_time = datetime.now()
    
    # Initialize metadata
    metadata = {
        'phase': 'phase1',
        'start_time': start_time.isoformat(),
        'sample_size': sample_size or config.get_sample_size(),
        'config': {
            'random_seed': config.get_random_seed(),
            'household_columns': len(config.get_phase1_columns('household')),
            'person_columns': len(config.get_phase1_columns('person'))
        },
        'processing_steps': [],
        'validation_results': {},
        'performance_metrics': {}
    }
    
    try:
        # Step 1: Load household data
        logger.info("Step 1: Loading household data")
        if input_data is not None:
            # Use pre-loaded data for streaming mode
            households = input_data
            logger.info("Using pre-loaded household data (streaming mode)")
        else:
            households = load_pums_households(sample_size=metadata['sample_size'])
        metadata['processing_steps'].append({
            'step': 'load_households',
            'records': len(households),
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Loaded {len(households)} households")
        
        # Step 2: Load person data
        logger.info("Step 2: Loading person data")
        household_serials = households['SERIALNO'].tolist()
        persons = load_pums_persons(household_serials)
        metadata['processing_steps'].append({
            'step': 'load_persons',
            'records': len(persons),
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Loaded {len(persons)} persons")
        
        # Step 3: Validate data consistency
        logger.info("Step 3: Validating data consistency")
        validation_results = validate_pums_data(households, persons)
        if not validation_results['valid']:
            raise Phase1ProcessingError(f"Data validation failed: {validation_results['errors']}")
        
        for warning in validation_results['warnings']:
            logger.warning(warning)
        
        metadata['validation_results']['data_consistency'] = validation_results
        
        # Step 4: Create household features
        logger.info("Step 4: Creating household features")
        try:
            households = create_household_features(households)
            metadata['processing_steps'].append({
                'step': 'household_features',
                'features_created': len([col for col in households.columns if col not in ['SERIALNO', 'building_id']]),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create household features: {str(e)}")
        
        # Step 5: Create person features
        logger.info("Step 5: Creating person features")
        try:
            persons = create_person_features(persons)
            metadata['processing_steps'].append({
                'step': 'person_features',
                'features_created': len([col for col in persons.columns if col not in ['SERIALNO', 'person_id']]),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create person features: {str(e)}")
        
        # Step 6: Create household roster
        logger.info("Step 6: Creating household roster")
        try:
            roster = create_household_roster(households, persons)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create household roster: {str(e)}")
        
        # Step 7: Create matching features
        logger.info("Step 7: Creating matching features")
        try:
            households, persons = create_matching_features(households, persons)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create matching features: {str(e)}")
        
        # Step 8: Create energy profile features
        logger.info("Step 8: Creating energy profile features")
        try:
            households = create_energy_profile_features(households)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create energy profile features: {str(e)}")
        
        # Step 8b: Create comprehensive matching features
        logger.info("Step 8b: Creating comprehensive matching features")
        try:
            households = create_comprehensive_matching_features(households, dataset_type='pums')
            metadata['processing_steps'].append({
                'step': 'comprehensive_features',
                'features_created': len([col for col in households.columns if col not in ['SERIALNO', 'building_id']]),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to create comprehensive matching features: {str(e)}")
        
        # Step 9: Merge persons into households
        logger.info("Step 9: Merging persons into household structure")
        try:
            buildings = merge_persons_to_buildings(households, persons, roster)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to merge persons to buildings: {str(e)}")
        metadata['processing_steps'].append({
            'step': 'merge_data',
            'buildings_created': len(buildings),
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 10: Final data preparation
        logger.info("Step 10: Final data preparation")
        buildings = prepare_final_output(buildings)
        
        # Calculate performance metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate average persons per building only for occupied buildings
        occupied_serials = buildings['SERIALNO'].unique()
        persons_in_occupied = persons[persons['SERIALNO'].isin(occupied_serials)]
        avg_persons_per_building = persons_in_occupied.groupby('SERIALNO').size().mean()
        
        metadata['end_time'] = end_time.isoformat()
        metadata['performance_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'households_per_second': len(households) / processing_time,
            'total_records_processed': len(households) + len(persons),
            'final_building_count': len(buildings),
            'average_persons_per_building': avg_persons_per_building,
            'memory_usage_mb': buildings.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Log performance summary
        create_performance_summary('phase1', metadata['performance_metrics'], logger)
        
        # Step 11: Save output (optional in streaming orchestrations)
        output_path = None
        if save_output:
            logger.info("Step 11: Saving output files")
            output_path = save_phase1_output(buildings, metadata)
            metadata['output_path'] = str(output_path)
        
        # Step 12: Validation and reporting
        if validate:
            logger.info("Step 12: Running validation and generating report")
            validation_results = validate_phase1_output(buildings)
            metadata['validation_results']['output_validation'] = validation_results
            
            report_path = generate_phase1_report(buildings, metadata, validation_results)
            metadata['report_path'] = str(report_path)
            logger.info(f"Validation report saved to: {report_path}")
        
        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
        if output_path is not None:
            logger.info(f"Output saved to: {output_path}")
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info("=" * 80)
        
        return buildings, metadata
        
    except Exception as e:
        logger.error(f"Phase 1 processing failed: {str(e)}")
        metadata['error'] = str(e)
        metadata['end_time'] = datetime.now().isoformat()
        
        # Save error metadata
        error_path = Path(config.get_data_path('phase1_output')).parent / 'phase1_error_metadata.json'
        serializable_metadata = convert_to_serializable(metadata)
        with open(error_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, default=str)
        
        raise Phase1ProcessingError(f"Phase 1 failed: {str(e)}")


def merge_persons_to_buildings(households: pd.DataFrame, persons: pd.DataFrame, 
                              roster: pd.DataFrame) -> pd.DataFrame:
    """
    Merge person data into household structure to create buildings.
    
    Args:
        households: Household DataFrame with features
        persons: Person DataFrame with features
        roster: Household roster with person counts
        
    Returns:
        DataFrame with building-level data including nested person information
    """
    logger.info("Merging persons into building structure")
    
    # Start with households as base - use copy() without deep to avoid fragmentation
    # Then ensure we're working with a fresh DataFrame
    buildings = pd.DataFrame(households)
    
    # Add roster information efficiently
    # First identify columns that need to be added from roster
    roster_cols_to_add = []
    roster_renamed = {}
    
    for col in roster.columns:
        if col != 'SERIALNO':  # Skip the join key
            if col in buildings.columns:
                # Rename conflicting columns
                new_col_name = f'roster_{col}'
                roster_renamed[col] = new_col_name
                roster_cols_to_add.append(new_col_name)
            else:
                roster_cols_to_add.append(col)
    
    # Rename columns in roster if needed
    if roster_renamed:
        roster = roster.rename(columns=roster_renamed)
    
    # Merge roster information in one operation
    if roster_cols_to_add and 'SERIALNO' in roster.columns:
        buildings = buildings.merge(
            roster[['SERIALNO'] + roster_cols_to_add],
            on='SERIALNO',
            how='left',
            suffixes=('', '_roster')
        )
    
    # Group persons by household
    persons_grouped = persons.groupby('SERIALNO')
    
    # Create person list for each building efficiently using dict comprehension
    person_lists = {
        serial: person_group.to_dict('records') 
        for serial, person_group in persons_grouped
    }
    
    # Add person lists to buildings using safer assignment method
    # Create a new Series first to avoid length mismatch issues
    person_series = pd.Series(
        [person_lists.get(serial, []) for serial in buildings['SERIALNO']],
        index=buildings.index
    )
    buildings['persons'] = person_series
    
    # Add summary statistics
    buildings['actual_person_count'] = buildings['persons'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    buildings['has_complete_persons'] = buildings['actual_person_count'] == buildings['NP']
    
    # Calculate household-level person statistics
    person_stats = persons.groupby('SERIALNO').agg({
        'AGEP': ['mean', 'min', 'max', 'std'],
        'is_employed': ['sum', 'mean'],
        'works_from_home': 'sum',
        'PINCP': ['sum', 'mean', 'max']
    })
    
    person_stats.columns = [
        'person_age_mean', 'person_age_min', 'person_age_max', 'person_age_std',
        'person_employed_count', 'employment_rate',
        'person_work_from_home_count',
        'household_income_sum', 'person_income_mean', 'person_income_max'
    ]
    
    # Check for any columns that already exist in buildings
    existing_cols = [col for col in person_stats.columns if col in buildings.columns]
    if existing_cols:
        logger.warning(f"Dropping existing columns before join: {existing_cols}")
        buildings = buildings.drop(columns=existing_cols, errors='ignore')
    
    buildings = buildings.join(person_stats, on='SERIALNO', how='left')
    
    # Fill missing values
    for col in person_stats.columns:
        if col in buildings.columns:
            buildings[col] = buildings[col].fillna(0)
    
    logger.info(f"Created {len(buildings)} buildings with embedded person data")
    
    return buildings


def prepare_final_output(buildings: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final output with proper column ordering and data types.
    
    Args:
        buildings: Building DataFrame
        
    Returns:
        Cleaned and ordered DataFrame with only occupied buildings
    """
    logger.info("Preparing final output")
    
    # Filter out buildings without persons
    initial_count = len(buildings)
    buildings = buildings[buildings['actual_person_count'] > 0].copy()
    removed_count = initial_count - len(buildings)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} buildings without persons (empty buildings)")
        logger.info(f"Keeping {len(buildings)} occupied buildings")
    
    # Define column order (key columns first)
    key_columns = [
        'building_id', 'SERIALNO', 'STATE', 'PUMA', 'DIVISION', 'REGION',
        'NP', 'actual_person_count', 'has_complete_persons'
    ]
    
    # Household characteristic columns
    household_columns = [
        'household_size_cat', 'income_quintile', 'building_type', 'building_type_simple',
        'building_age_cat', 'household_composition', 'tenure_type', 'heating_fuel',
        'urban_rural', 'has_children', 'has_seniors', 'multigenerational'
    ]
    
    # Energy-related columns
    energy_columns = [
        'energy_burden', 'high_energy_burden', 'total_energy_cost',
        'base_load_score', 'hvac_load_score', 'peak_load_score',
        'energy_intensity', 'energy_intensity_cat'
    ]
    
    # Person summary columns
    person_columns = [
        'person_age_mean', 'person_age_min', 'person_age_max',
        'person_employed_count', 'employment_rate', 'person_work_from_home_count'
    ]
    
    # Weight columns
    weight_columns = ['WGTP'] + [f'WGTP{i}' for i in range(1, 81)]
    
    # Collect all ordered columns that exist
    ordered_columns = []
    for col_group in [key_columns, household_columns, energy_columns, person_columns]:
        ordered_columns.extend([col for col in col_group if col in buildings.columns])
    
    # Add weight columns
    ordered_columns.extend([col for col in weight_columns if col in buildings.columns])
    
    # Add persons column
    if 'persons' in buildings.columns:
        ordered_columns.append('persons')
    
    # Add any remaining columns
    remaining_columns = [col for col in buildings.columns if col not in ordered_columns]
    ordered_columns.extend(sorted(remaining_columns))
    
    # Reorder columns
    buildings = buildings[ordered_columns]
    
    # Optimize data types
    # Convert float columns that are actually integers
    int_columns = ['NP', 'actual_person_count', 'person_employed_count', 'person_work_from_home_count']
    for col in int_columns:
        if col in buildings.columns:
            buildings[col] = buildings[col].fillna(0).astype(int)
    
    # Convert boolean columns
    bool_columns = ['has_complete_persons', 'has_children', 'has_seniors', 
                   'multigenerational', 'high_energy_burden']
    for col in bool_columns:
        if col in buildings.columns:
            buildings[col] = buildings[col].astype(bool)
    
    # Ensure building_id is string
    buildings['building_id'] = buildings['building_id'].astype(str)
    
    logger.info(f"Final output prepared with {len(buildings.columns)} columns")
    
    return buildings


def process_phase1_chunk(households_chunk: pd.DataFrame, loader, config: Dict) -> pd.DataFrame:
    """
    Process a chunk of households for parallel execution.
    
    Args:
        households_chunk: Chunk of household data
        loader: PUMSDataLoader instance
        config: Configuration dictionary
        
    Returns:
        DataFrame of buildings with persons
    """
    # Get unique SERIALNOs from this chunk
    serialnos = households_chunk['SERIALNO'].unique()
    
    # Convert numpy array to list for the loader
    serialnos_list = serialnos.tolist() if hasattr(serialnos, 'tolist') else list(serialnos)
    
    # Load persons for these households
    persons_df = loader.load_persons_for_serialnos(serialnos_list)
    
    # Group persons by household
    persons_by_household = persons_df.groupby('SERIALNO').apply(
        lambda x: x.to_dict('records')
    ).to_dict()
    
    # Add persons to households
    buildings = households_chunk.copy()
    buildings['persons'] = buildings['SERIALNO'].map(persons_by_household)
    
    # Filter out empty households
    buildings = buildings[buildings['persons'].notna()]
    buildings = buildings[buildings['persons'].apply(lambda x: len(x) > 0 if x else False)]
    
    # Add enhanced features using the same approach as main process_phase1
    from ..utils.feature_engineering import create_household_features, create_person_features, create_matching_features, create_energy_profile_features
    from ..utils.enhanced_feature_engineering import create_comprehensive_matching_features
    
    # Apply feature engineering
    buildings = create_household_features(buildings)
    
    # Create person features (this creates is_employed, works_from_home, etc.)
    persons_df = create_person_features(persons_df)
    
    # Update persons in buildings with the new features
    persons_by_household_updated = persons_df.groupby('SERIALNO').apply(
        lambda x: x.to_dict('records')
    ).to_dict()
    buildings['persons'] = buildings['SERIALNO'].map(persons_by_household_updated)
    
    # Extract persons for matching features
    # Note: persons_df already contains the persons we loaded with features
    buildings, persons_df = create_matching_features(buildings, persons_df)
    
    buildings = create_energy_profile_features(buildings)
    buildings = create_comprehensive_matching_features(buildings, dataset_type='pums')
    
    return buildings


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    # Handle numpy integer types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    # Handle numpy unsigned integer types
    elif isinstance(obj, (np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    # Handle numpy float types
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    # Handle numpy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle pandas Series
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # Handle lists recursively
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    # Handle any other numpy scalar with .item() method
    elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        return obj.item()
    # Return as-is for regular Python types
    return obj


@log_execution_time(logger)
def save_phase1_output(buildings: pd.DataFrame, metadata: Dict) -> Path:
    """
    Save Phase 1 output and metadata.
    
    Args:
        buildings: Building DataFrame to save
        metadata: Processing metadata
        
    Returns:
        Path to saved file
        
    Raises:
        Phase1ProcessingError: If saving fails
    """
    try:
        config = get_config()
        output_path = Path(config.get_data_path('phase1_output'))
        
        # Improved path handling - check if it's a file or directory
        if output_path.exists() and output_path.is_dir():
            # If it's a directory, append the filename
            output_path = output_path / 'phase1_pums_buildings.pkl'
        elif not output_path.suffix:
            # If no extension, assume it's a directory path
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / 'phase1_pums_buildings.pkl'
        else:
            # It's a file path, ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main data as pickle
        logger.info(f"Saving buildings data to: {output_path}")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(buildings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to save buildings data: {str(e)}")
        
        # Save metadata
        metadata_path = output_path.parent / 'phase1_metadata.json'
        logger.info(f"Saving metadata to: {metadata_path}")
        
        try:
            serializable_metadata = convert_to_serializable(metadata)
            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2, default=str)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to save metadata: {str(e)}")
        
        # Save sample for inspection (first 10 buildings without person lists)
        try:
            sample_path = output_path.parent / 'phase1_sample.csv'
            sample_df = buildings.head(10).copy()
            if 'persons' in sample_df.columns:
                sample_df = sample_df.drop('persons', axis=1)
            sample_df.to_csv(sample_path, index=False)
            logger.info(f"Saved sample to: {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save sample CSV: {str(e)}")
            # Non-critical, so just log warning
        
        # Log file sizes
        try:
            file_size_mb = output_path.stat().st_size / 1024**2
            logger.info(f"Output file size: {file_size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get file size: {str(e)}")
        
        return output_path
        
    except Exception as e:
        if isinstance(e, Phase1ProcessingError):
            raise
        else:
            raise Phase1ProcessingError(f"Failed to save Phase 1 output: {str(e)}")


def load_phase1_output() -> Tuple[pd.DataFrame, Dict]:
    """
    Load Phase 1 output and metadata.
    
    Returns:
        Tuple of (buildings, metadata)
        
    Raises:
        Phase1ProcessingError: If files cannot be loaded
    """
    try:
        config = get_config()
        output_path = Path(config.get_data_path('phase1_output'))
        metadata_path = output_path.parent / 'phase1_metadata.json'
        
        # Check if files exist
        if not output_path.exists():
            raise Phase1ProcessingError(f"Phase 1 output file not found: {output_path}")
        if not metadata_path.exists():
            raise Phase1ProcessingError(f"Phase 1 metadata file not found: {metadata_path}")
        
        # Load buildings
        try:
            with open(output_path, 'rb') as f:
                buildings = pickle.load(f)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to load buildings data: {str(e)}")
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            raise Phase1ProcessingError(f"Failed to load metadata: {str(e)}")
        
        # Validate loaded data
        if not isinstance(buildings, pd.DataFrame):
            raise Phase1ProcessingError("Loaded buildings data is not a DataFrame")
        if not isinstance(metadata, dict):
            raise Phase1ProcessingError("Loaded metadata is not a dictionary")
        
        logger.info(f"Successfully loaded {len(buildings)} buildings from Phase 1")
        
        return buildings, metadata
        
    except Exception as e:
        if isinstance(e, Phase1ProcessingError):
            raise
        else:
            raise Phase1ProcessingError(f"Failed to load Phase 1 output: {str(e)}")


if __name__ == "__main__":
    # Test Phase 1 processing with small sample
    try:
        logger.info("Running Phase 1 test with 10 buildings")
        buildings, metadata = process_phase1(sample_size=10, validate=True)
        
        print(f"\nPhase 1 test completed successfully!")
        print(f"Buildings created: {len(buildings)}")
        print(f"Columns: {len(buildings.columns)}")
        print(f"Processing time: {metadata['performance_metrics']['total_processing_time_seconds']:.2f} seconds")
        
        # Show sample building
        if len(buildings) > 0:
            sample = buildings.iloc[0]
            print(f"\nSample building:")
            print(f"  Building ID: {sample['building_id']}")
            print(f"  State: {sample['STATE']}")
            print(f"  Household size: {sample['NP']}")
            print(f"  Person count: {sample['actual_person_count']}")
            print(f"  Building type: {sample.get('building_type', 'unknown')}")
            print(f"  Income quintile: {sample.get('income_quintile', 'unknown')}")
            
    except Exception as e:
        logger.error(f"Phase 1 test failed: {str(e)}")
        raise