"""
Phase-specific validators for PUMS Enrichment Pipeline.

This module provides validation functions for each phase of the pipeline,
ensuring data quality and consistency throughout processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


def validate_phase1_output(buildings: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate Phase 1 output (PUMS household-person integration).
    
    Args:
        buildings: Building DataFrame from Phase 1
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating Phase 1 output")
    
    results = {
        'phase': 'phase1',
        'valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {},
        'checks_passed': [],
        'checks_failed': []
    }
    
    # Check 1: Required columns exist
    required_columns = [
        'building_id', 'SERIALNO', 'STATE', 'PUMA', 'NP', 
        'actual_person_count', 'has_complete_persons', 'persons'
    ]
    missing_columns = [col for col in required_columns if col not in buildings.columns]
    
    if missing_columns:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
        results['checks_failed'].append('required_columns')
    else:
        results['checks_passed'].append('required_columns')
    
    # Check 2: Unique building IDs
    duplicate_ids = buildings['building_id'].duplicated().sum()
    if duplicate_ids > 0:
        results['valid'] = False
        results['errors'].append(f"Found {duplicate_ids} duplicate building IDs")
        results['checks_failed'].append('unique_ids')
    else:
        results['checks_passed'].append('unique_ids')
    
    # Check 3: Person count consistency
    if 'persons' in buildings.columns:
        actual_counts = buildings['persons'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mismatched = (actual_counts != buildings['actual_person_count']).sum()
        if mismatched > 0:
            results['warnings'].append(f"{mismatched} buildings have mismatched person counts")
            results['checks_failed'].append('person_count_consistency')
        else:
            results['checks_passed'].append('person_count_consistency')
    
    # Check 4: NP vs actual person count
    if all(col in buildings.columns for col in ['NP', 'actual_person_count']):
        incomplete = (buildings['NP'] != buildings['actual_person_count']).sum()
        incomplete_pct = (incomplete / len(buildings)) * 100
        
        results['metrics']['incomplete_households'] = incomplete
        results['metrics']['incomplete_households_pct'] = incomplete_pct
        
        config = get_config()
        max_missing = config.get_validation_config().get('max_missing_percentage', 5.0)
        
        if incomplete_pct > max_missing:
            results['warnings'].append(
                f"{incomplete_pct:.1f}% of households have incomplete person records "
                f"(exceeds threshold of {max_missing}%)"
            )
            results['checks_failed'].append('completeness_threshold')
        else:
            results['checks_passed'].append('completeness_threshold')
    
    # Check 5: Geographic coverage
    results['metrics']['states_covered'] = buildings['STATE'].nunique()
    results['metrics']['pumas_covered'] = buildings['PUMA'].nunique()
    
    # Check 6: Data types
    expected_types = {
        'building_id': 'object',
        'SERIALNO': 'object',
        'STATE': 'object',
        'PUMA': 'object',
        'NP': 'int',
        'actual_person_count': 'int',
        'has_complete_persons': 'bool'
    }
    
    type_issues = []
    for col, expected_type in expected_types.items():
        if col in buildings.columns:
            actual_type = str(buildings[col].dtype)
            if expected_type == 'int' and 'int' not in actual_type:
                type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
            elif expected_type == 'object' and actual_type not in ['object', 'string']:
                type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
            elif expected_type == 'bool' and actual_type != 'bool':
                type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
    
    if type_issues:
        results['warnings'].extend(type_issues)
        results['checks_failed'].append('data_types')
    else:
        results['checks_passed'].append('data_types')
    
    # Check 7: Feature completeness
    feature_columns = [
        'household_size_cat', 'income_quintile', 'building_type',
        'energy_intensity_cat', 'household_composition'
    ]
    existing_features = [col for col in feature_columns if col in buildings.columns]
    
    results['metrics']['feature_columns'] = len(existing_features)
    results['metrics']['feature_coverage'] = len(existing_features) / len(feature_columns)
    
    if len(existing_features) < len(feature_columns) * 0.8:
        results['warnings'].append(
            f"Only {len(existing_features)}/{len(feature_columns)} expected features found"
        )
        results['checks_failed'].append('feature_completeness')
    else:
        results['checks_passed'].append('feature_completeness')
    
    # Check 8: Weight columns
    weight_columns = ['WGTP'] + [f'WGTP{i}' for i in range(1, 81)]
    existing_weights = [col for col in weight_columns if col in buildings.columns]
    
    results['metrics']['weight_columns'] = len(existing_weights)
    
    if len(existing_weights) < 81:
        results['warnings'].append(
            f"Only {len(existing_weights)}/81 weight columns found"
        )
    
    # Check 9: Value ranges
    if 'NP' in buildings.columns:
        invalid_np = (buildings['NP'] < 1) | (buildings['NP'] > 20)
        if invalid_np.any():
            results['warnings'].append(
                f"{invalid_np.sum()} buildings have invalid NP values"
            )
            results['checks_failed'].append('value_ranges')
        else:
            results['checks_passed'].append('value_ranges')
    
    # Check 10: Person data structure
    if 'persons' in buildings.columns:
        sample_size = min(10, len(buildings))
        sample_buildings = buildings.head(sample_size)
        
        person_structure_valid = True
        for idx, row in sample_buildings.iterrows():
            if isinstance(row['persons'], list):
                for person in row['persons']:
                    if not isinstance(person, dict):
                        person_structure_valid = False
                        break
                    required_person_fields = ['person_id', 'SERIALNO', 'SPORDER', 'AGEP']
                    if not all(field in person for field in required_person_fields):
                        person_structure_valid = False
                        break
            else:
                person_structure_valid = False
                break
        
        if person_structure_valid:
            results['checks_passed'].append('person_data_structure')
        else:
            results['warnings'].append("Person data structure validation failed in sample")
            results['checks_failed'].append('person_data_structure')
    
    # Calculate summary metrics
    results['metrics']['total_buildings'] = len(buildings)
    results['metrics']['total_persons'] = buildings['actual_person_count'].sum()
    results['metrics']['avg_persons_per_building'] = (
        results['metrics']['total_persons'] / results['metrics']['total_buildings']
    )
    
    if 'household_income_sum' in buildings.columns:
        results['metrics']['avg_household_income'] = buildings['household_income_sum'].mean()
    
    if 'energy_intensity' in buildings.columns:
        results['metrics']['avg_energy_intensity'] = buildings['energy_intensity'].mean()
    
    # Overall validation status
    if len(results['errors']) > 0:
        results['valid'] = False
        results['summary'] = f"Phase 1 validation FAILED with {len(results['errors'])} errors"
    elif len(results['warnings']) > 0:
        results['summary'] = f"Phase 1 validation PASSED with {len(results['warnings'])} warnings"
    else:
        results['summary'] = "Phase 1 validation PASSED - all checks successful"
    
    logger.info(f"Validation complete: {results['summary']}")
    logger.info(f"Checks passed: {len(results['checks_passed'])}/{len(results['checks_passed']) + len(results['checks_failed'])}")
    
    return results


def validate_phase2_output(buildings: pd.DataFrame, matching_stats: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Validate Phase 2 output (RECS matching).
    
    Args:
        buildings: Building DataFrame from Phase 2
        matching_stats: Optional matching statistics
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating Phase 2 output")
    
    results = {
        'phase': 'phase2',
        'valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {},
        'checks_passed': [],
        'checks_failed': []
    }
    
    # Check all Phase 1 requirements still met
    phase1_results = validate_phase1_output(buildings)
    if not phase1_results['valid']:
        results['valid'] = False
        results['errors'].append("Phase 1 validation failed on Phase 2 output")
        results['checks_failed'].append('phase1_requirements')
    else:
        results['checks_passed'].append('phase1_requirements')
    
    # Phase 2 specific checks
    # Check 1: RECS template assignment
    if 'recs_template_id' not in buildings.columns:
        results['valid'] = False
        results['errors'].append("Missing recs_template_id column")
        results['checks_failed'].append('recs_assignment')
    else:
        unmatched = buildings['recs_template_id'].isna().sum()
        match_rate = (len(buildings) - unmatched) / len(buildings)
        
        results['metrics']['match_rate'] = match_rate
        results['metrics']['unmatched_buildings'] = unmatched
        
        config = get_config()
        min_match_rate = config.get_validation_config().get('min_match_rate', 0.95)
        
        if match_rate < min_match_rate:
            results['valid'] = False
            results['errors'].append(
                f"Match rate {match_rate:.2%} below minimum {min_match_rate:.2%}"
            )
            results['checks_failed'].append('match_rate_threshold')
        else:
            results['checks_passed'].append('match_rate_threshold')
            results['checks_passed'].append('recs_assignment')
    
    # Check 2: Match quality scores
    if 'match_weight' in buildings.columns:
        results['metrics']['avg_match_weight'] = buildings['match_weight'].mean()
        results['metrics']['min_match_weight'] = buildings['match_weight'].min()
        results['metrics']['max_match_weight'] = buildings['match_weight'].max()
        
        low_quality = (buildings['match_weight'] < 5).sum()
        if low_quality > len(buildings) * 0.1:
            results['warnings'].append(
                f"{low_quality} buildings ({low_quality/len(buildings):.1%}) have low match weights"
            )
            results['checks_failed'].append('match_quality')
        else:
            results['checks_passed'].append('match_quality')
    
    # Check 3: RECS features added
    expected_recs_features = [
        'square_footage', 'year_built_recs', 'housing_unit_type',
        'heating_fuel_recs', 'cooling_type', 'water_heater_fuel'
    ]
    
    recs_features_found = [col for col in expected_recs_features if col in buildings.columns]
    results['metrics']['recs_features_added'] = len(recs_features_found)
    
    if len(recs_features_found) < len(expected_recs_features) * 0.5:
        results['warnings'].append(
            f"Only {len(recs_features_found)}/{len(expected_recs_features)} RECS features found"
        )
        results['checks_failed'].append('recs_features')
    else:
        results['checks_passed'].append('recs_features')
    
    # Check 4: Template usage balance
    if 'recs_template_id' in buildings.columns:
        template_usage = buildings['recs_template_id'].value_counts()
        usage_cv = template_usage.std() / template_usage.mean()  # Coefficient of variation
        
        results['metrics']['unique_templates_used'] = len(template_usage)
        results['metrics']['template_usage_cv'] = usage_cv
        
        if usage_cv > 1.0:  # High variability in usage
            results['warnings'].append(
                f"Template usage is imbalanced (CV={usage_cv:.2f})"
            )
            results['checks_failed'].append('template_balance')
        else:
            results['checks_passed'].append('template_balance')
    
    # Check matching statistics if provided
    if matching_stats:
        results['metrics']['matching_stats'] = matching_stats
        
        # Check EM convergence
        if 'em_converged' in matching_stats and not matching_stats['em_converged']:
            results['warnings'].append("EM algorithm did not converge")
            results['checks_failed'].append('em_convergence')
        else:
            results['checks_passed'].append('em_convergence')
    
    # Overall status
    if len(results['errors']) > 0:
        results['valid'] = False
        results['summary'] = f"Phase 2 validation FAILED with {len(results['errors'])} errors"
    elif len(results['warnings']) > 0:
        results['summary'] = f"Phase 2 validation PASSED with {len(results['warnings'])} warnings"
    else:
        results['summary'] = "Phase 2 validation PASSED - all checks successful"
    
    return results


def validate_phase3_output(buildings: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate Phase 3 output (ATUS activity assignment).
    
    Args:
        buildings: Building DataFrame from Phase 3
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating Phase 3 output")
    
    results = {
        'phase': 'phase3',
        'valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {},
        'checks_passed': [],
        'checks_failed': []
    }
    
    # Check Phase 2 requirements
    phase2_results = validate_phase2_output(buildings)
    if not phase2_results['valid']:
        results['valid'] = False
        results['errors'].append("Phase 2 validation failed on Phase 3 output")
        results['checks_failed'].append('phase2_requirements')
    else:
        results['checks_passed'].append('phase2_requirements')
    
    # Phase 3 specific checks
    # Check 1: Activity assignment
    if 'persons' not in buildings.columns:
        results['valid'] = False
        results['errors'].append("Missing persons column")
        return results
    
    # Sample buildings to check person activities
    sample_size = min(100, len(buildings))
    sample_buildings = buildings.sample(n=sample_size, random_state=42)
    
    persons_with_activities = 0
    total_persons_checked = 0
    activity_issues = []
    
    for idx, building in sample_buildings.iterrows():
        if isinstance(building['persons'], list):
            for person in building['persons']:
                total_persons_checked += 1
                # Check for ATUS template ID (fixed from activity_pattern)
                if isinstance(person, dict) and ('atus_template_id' in person or 'activity_pattern' in person):
                    persons_with_activities += 1
                    
                    # Check activity pattern structure (if it exists)
                    if 'activity_pattern' in person:
                        pattern = person['activity_pattern']
                        if not isinstance(pattern, (list, dict)):
                            activity_issues.append("Invalid activity pattern structure")
    
    if total_persons_checked > 0:
        activity_rate = persons_with_activities / total_persons_checked
        results['metrics']['activity_assignment_rate'] = activity_rate
        
        if activity_rate < 0.9:
            results['warnings'].append(
                f"Only {activity_rate:.1%} of persons have activity patterns"
            )
            results['checks_failed'].append('activity_assignment')
        else:
            results['checks_passed'].append('activity_assignment')
    
    # Check 2: Household coordination
    coordination_issues = 0
    households_checked = 0
    
    for idx, building in sample_buildings.iterrows():
        if isinstance(building['persons'], list) and len(building['persons']) > 1:
            households_checked += 1
            
            # Check for basic coordination (simplified check)
            has_children = any(p.get('AGEP', 100) < 18 for p in building['persons'])
            has_adult_home = any(
                p.get('activity_pattern', {}).get('daytime_at_home', False) 
                for p in building['persons'] 
                if p.get('AGEP', 0) >= 18
            )
            
            if has_children and not has_adult_home:
                coordination_issues += 1
    
    if households_checked > 0:
        coordination_rate = 1 - (coordination_issues / households_checked)
        results['metrics']['household_coordination_rate'] = coordination_rate
        
        if coordination_rate < 0.9:
            results['warnings'].append(
                f"Household coordination issues in {coordination_issues}/{households_checked} families"
            )
            results['checks_failed'].append('household_coordination')
        else:
            results['checks_passed'].append('household_coordination')
    
    # Check 3: Activity diversity
    activity_types = set()
    for idx, building in sample_buildings.iterrows():
        if isinstance(building['persons'], list):
            for person in building['persons']:
                if isinstance(person, dict) and 'person_type' in person:
                    activity_types.add(person['person_type'])
    
    results['metrics']['activity_diversity'] = len(activity_types)
    
    if len(activity_types) < 5:
        results['warnings'].append(
            f"Low activity diversity: only {len(activity_types)} person types found"
        )
        results['checks_failed'].append('activity_diversity')
    else:
        results['checks_passed'].append('activity_diversity')
    
    # Overall status
    if len(results['errors']) > 0:
        results['valid'] = False
        results['summary'] = f"Phase 3 validation FAILED with {len(results['errors'])} errors"
    elif len(results['warnings']) > 0:
        results['summary'] = f"Phase 3 validation PASSED with {len(results['warnings'])} warnings"
    else:
        results['summary'] = "Phase 3 validation PASSED - all checks successful"
    
    return results


def validate_phase4_output(buildings: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate Phase 4 output (weather integration with activity alignment).
    
    Args:
        buildings: Building DataFrame from Phase 4
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating Phase 4 output")
    
    results = {
        'phase': 'phase4',
        'valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {},
        'checks_passed': [],
        'checks_failed': []
    }
    
    # Check Phase 3 requirements
    phase3_results = validate_phase3_output(buildings)
    if not phase3_results['valid']:
        results['valid'] = False
        results['errors'].append("Phase 3 validation failed on Phase 4 output")
        results['checks_failed'].append('phase3_requirements')
    else:
        results['checks_passed'].append('phase3_requirements')
    
    # Phase 4 specific checks
    # Check 1: Weather summary data
    buildings_with_weather = 0
    buildings_with_weather_summary = 0
    
    for idx, building in buildings.iterrows():
        if 'has_weather' in building and building['has_weather']:
            buildings_with_weather += 1
        if 'weather_summary' in building and building['weather_summary']:
            buildings_with_weather_summary += 1
    
    weather_coverage = buildings_with_weather / len(buildings) if len(buildings) > 0 else 0
    results['metrics']['weather_coverage'] = weather_coverage
    results['metrics']['buildings_with_weather_summary'] = buildings_with_weather_summary
    
    if weather_coverage < 0.95:
        results['warnings'].append(
            f"Weather data missing for {len(buildings) - buildings_with_weather} buildings ({1-weather_coverage:.1%})"
        )
        results['checks_failed'].append('weather_coverage')
    else:
        results['checks_passed'].append('weather_coverage')
    
    # Check 2: Activity-level weather alignment
    persons_with_weather = 0
    activities_with_weather = 0
    total_persons = 0
    total_activities = 0
    
    sample_buildings = buildings.head(min(100, len(buildings)))
    
    for idx, building in sample_buildings.iterrows():
        if 'persons' in building and isinstance(building['persons'], list):
            for person in building['persons']:
                if isinstance(person, dict):
                    total_persons += 1
                    if person.get('weather_aligned', False):
                        persons_with_weather += 1
                    
                    # Check activity sequences
                    if 'activity_sequence' in person:
                        for activity in person['activity_sequence']:
                            total_activities += 1
                            if 'weather' in activity:
                                activities_with_weather += 1
    
    if total_persons > 0:
        person_weather_rate = persons_with_weather / total_persons
        results['metrics']['person_weather_alignment_rate'] = person_weather_rate
        
        if person_weather_rate < 0.9:
            results['warnings'].append(
                f"Only {person_weather_rate:.1%} of persons have weather-aligned activities"
            )
            results['checks_failed'].append('person_weather_alignment')
        else:
            results['checks_passed'].append('person_weather_alignment')
    
    if total_activities > 0:
        activity_weather_rate = activities_with_weather / total_activities
        results['metrics']['activity_weather_alignment_rate'] = activity_weather_rate
        
        if activity_weather_rate < 0.9:
            results['warnings'].append(
                f"Only {activity_weather_rate:.1%} of activities have weather data"
            )
            results['checks_failed'].append('activity_weather_alignment')
        else:
            results['checks_passed'].append('activity_weather_alignment')
    
    # Check 3: Weather data reasonableness
    sample_weather = []
    for idx, building in sample_buildings.iterrows():
        if 'weather_summary' in building and isinstance(building['weather_summary'], dict):
            if 'temp_mean' in building['weather_summary']:
                sample_weather.append(building['weather_summary']['temp_mean'])
    
    if sample_weather:
        temp_mean = np.mean(sample_weather)
        temp_min = np.min(sample_weather)
        temp_max = np.max(sample_weather)
        
        results['metrics']['temp_range'] = f"{temp_min:.1f} to {temp_max:.1f}"
        
        # Check for unreasonable temperatures (Celsius)
        if temp_min < -40 or temp_max > 50:
            results['warnings'].append(
                f"Potentially unreasonable temperature range: {temp_min:.1f} to {temp_max:.1f}°C"
            )
            results['checks_failed'].append('temperature_range')
        else:
            results['checks_passed'].append('temperature_range')
    
    # Check 4: HDD/CDD presence
    hdd_cdd_found = 0
    for idx, building in sample_buildings.iterrows():
        if 'weather_summary' in building and isinstance(building['weather_summary'], dict):
            if 'HDD' in building['weather_summary'] and 'CDD' in building['weather_summary']:
                hdd_cdd_found += 1
    
    if hdd_cdd_found > 0:
        results['checks_passed'].append('degree_days_calculation')
    else:
        results['warnings'].append("No HDD/CDD calculations found")
        results['checks_failed'].append('degree_days_calculation')
    
    # Final completeness check
    results['metrics']['final_building_count'] = len(buildings)
    results['metrics']['final_person_count'] = buildings['actual_person_count'].sum()
    results['metrics']['data_completeness'] = 1 - (buildings.isna().sum().sum() / buildings.size)
    
    # Overall status
    if len(results['errors']) > 0:
        results['valid'] = False
        results['summary'] = f"Phase 4 validation FAILED with {len(results['errors'])} errors"
    elif len(results['warnings']) > 0:
        results['summary'] = f"Phase 4 validation PASSED with {len(results['warnings'])} warnings"
    else:
        results['summary'] = "Phase 4 validation PASSED - all checks successful"
    
    return results


def run_all_validations(buildings: pd.DataFrame, current_phase: int = 4) -> Dict[str, Any]:
    """
    Run all applicable validations up to the current phase.
    
    Args:
        buildings: Building DataFrame
        current_phase: Current phase number (1-4)
        
    Returns:
        Dictionary with all validation results
    """
    all_results = {
        'current_phase': current_phase,
        'overall_valid': True,
        'phase_results': {}
    }
    
    validators = [
        (1, validate_phase1_output),
        (2, validate_phase2_output),
        (3, validate_phase3_output),
        (4, validate_phase4_output)
    ]
    
    for phase_num, validator in validators:
        if phase_num <= current_phase:
            results = validator(buildings)
            all_results['phase_results'][f'phase{phase_num}'] = results
            
            if not results['valid']:
                all_results['overall_valid'] = False
    
    # Summary
    total_errors = sum(
        len(r['errors']) for r in all_results['phase_results'].values()
    )
    total_warnings = sum(
        len(r['warnings']) for r in all_results['phase_results'].values()
    )
    
    all_results['summary'] = {
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'phases_validated': len(all_results['phase_results']),
        'overall_status': 'PASSED' if all_results['overall_valid'] else 'FAILED'
    }
    
    return all_results