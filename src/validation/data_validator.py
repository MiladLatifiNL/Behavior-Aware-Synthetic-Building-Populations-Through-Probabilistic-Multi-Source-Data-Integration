"""
General data validation utilities for PUMS Enrichment Pipeline.

This module provides common validation functions used across all phases
to ensure data quality and integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validate that input is a non-empty DataFrame.
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
        
    Raises:
        DataValidationError: If validation fails
    """
    if df is None:
        raise DataValidationError(f"{name} is None")
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{name} is not a pandas DataFrame")
    
    if len(df) == 0:
        raise DataValidationError(f"{name} is empty")


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], 
                            name: str = "DataFrame") -> List[str]:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages
        
    Returns:
        List of missing columns (empty if all present)
        
    Raises:
        DataValidationError: If any required columns are missing
    """
    validate_dataframe(df, name)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataValidationError(
            f"{name} missing required columns: {missing_columns}"
        )
    
    return missing_columns


def validate_column_types(df: pd.DataFrame, expected_types: Dict[str, Union[str, type]], 
                         strict: bool = False) -> Dict[str, str]:
    """
    Validate column data types.
    
    Args:
        df: DataFrame to validate
        expected_types: Dict mapping column names to expected types
        strict: If True, raise error on type mismatch; if False, log warning
        
    Returns:
        Dict of column type mismatches
        
    Raises:
        DataValidationError: If strict=True and types don't match
    """
    type_issues = {}
    
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            continue
            
        actual_type = str(df[col].dtype)
        
        # Handle flexible type matching
        type_matches = False
        
        if isinstance(expected_type, str):
            if expected_type == 'numeric':
                type_matches = pd.api.types.is_numeric_dtype(df[col])
            elif expected_type == 'string':
                type_matches = pd.api.types.is_string_dtype(df[col]) or actual_type == 'object'
            elif expected_type == 'integer':
                type_matches = pd.api.types.is_integer_dtype(df[col])
            elif expected_type == 'float':
                type_matches = pd.api.types.is_float_dtype(df[col])
            elif expected_type == 'boolean':
                type_matches = pd.api.types.is_bool_dtype(df[col])
            elif expected_type == 'datetime':
                type_matches = pd.api.types.is_datetime64_any_dtype(df[col])
            elif expected_type == 'category':
                type_matches = pd.api.types.is_categorical_dtype(df[col])
            else:
                type_matches = actual_type == expected_type
        else:
            type_matches = df[col].dtype == expected_type
        
        if not type_matches:
            type_issues[col] = f"Expected {expected_type}, got {actual_type}"
            
            if strict:
                raise DataValidationError(
                    f"Column '{col}' has wrong type: {type_issues[col]}"
                )
            else:
                logger.warning(f"Column '{col}' type mismatch: {type_issues[col]}")
    
    return type_issues


def validate_numeric_range(df: pd.DataFrame, column: str, 
                          min_value: Optional[float] = None, 
                          max_value: Optional[float] = None,
                          allow_null: bool = True) -> Dict[str, Any]:
    """
    Validate numeric column values are within expected range.
    
    Args:
        df: DataFrame containing the column
        column: Column name to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allow_null: Whether null values are allowed
        
    Returns:
        Dict with validation statistics
        
    Raises:
        DataValidationError: If values are out of range
    """
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' not found")
    
    col_data = df[column]
    
    # Check nulls
    null_count = col_data.isna().sum()
    if null_count > 0 and not allow_null:
        raise DataValidationError(f"Column '{column}' contains {null_count} null values")
    
    # Get non-null values for range check
    non_null_data = col_data.dropna()
    
    results = {
        'column': column,
        'null_count': null_count,
        'null_percentage': (null_count / len(df)) * 100,
        'min_value': non_null_data.min() if len(non_null_data) > 0 else None,
        'max_value': non_null_data.max() if len(non_null_data) > 0 else None,
        'mean_value': non_null_data.mean() if len(non_null_data) > 0 else None,
        'out_of_range_count': 0
    }
    
    if len(non_null_data) > 0:
        # Check range
        out_of_range = 0
        
        if min_value is not None:
            below_min = (non_null_data < min_value).sum()
            if below_min > 0:
                out_of_range += below_min
                logger.warning(f"{below_min} values in '{column}' below minimum {min_value}")
        
        if max_value is not None:
            above_max = (non_null_data > max_value).sum()
            if above_max > 0:
                out_of_range += above_max
                logger.warning(f"{above_max} values in '{column}' above maximum {max_value}")
        
        results['out_of_range_count'] = out_of_range
        
        if out_of_range > 0:
            raise DataValidationError(
                f"Column '{column}' has {out_of_range} values outside range [{min_value}, {max_value}]"
            )
    
    return results


def validate_categorical_values(df: pd.DataFrame, column: str, 
                               valid_values: List[Any],
                               allow_null: bool = True) -> Dict[str, Any]:
    """
    Validate categorical column contains only expected values.
    
    Args:
        df: DataFrame containing the column
        column: Column name to validate
        valid_values: List of valid values
        allow_null: Whether null values are allowed
        
    Returns:
        Dict with validation statistics
        
    Raises:
        DataValidationError: If invalid values found
    """
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' not found")
    
    col_data = df[column]
    
    # Check nulls
    null_count = col_data.isna().sum()
    if null_count > 0 and not allow_null:
        raise DataValidationError(f"Column '{column}' contains {null_count} null values")
    
    # Check values
    unique_values = col_data.dropna().unique()
    invalid_values = [val for val in unique_values if val not in valid_values]
    
    results = {
        'column': column,
        'null_count': null_count,
        'unique_count': len(unique_values),
        'invalid_count': len(invalid_values),
        'invalid_values': invalid_values[:10]  # First 10 for brevity
    }
    
    if invalid_values:
        logger.warning(f"Column '{column}' contains {len(invalid_values)} invalid values: {invalid_values[:5]}")
        raise DataValidationError(
            f"Column '{column}' contains invalid values: {invalid_values[:5]}"
        )
    
    return results


def validate_uniqueness(df: pd.DataFrame, column: str, 
                       expected_unique: bool = True) -> Dict[str, Any]:
    """
    Validate column uniqueness constraints.
    
    Args:
        df: DataFrame containing the column
        column: Column name to validate
        expected_unique: Whether values should be unique
        
    Returns:
        Dict with validation statistics
        
    Raises:
        DataValidationError: If uniqueness constraint violated
    """
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' not found")
    
    duplicate_count = df[column].duplicated().sum()
    
    results = {
        'column': column,
        'total_count': len(df),
        'unique_count': df[column].nunique(),
        'duplicate_count': duplicate_count,
        'is_unique': duplicate_count == 0
    }
    
    if expected_unique and duplicate_count > 0:
        # Find example duplicates
        duplicated_values = df[df[column].duplicated(keep=False)][column].value_counts().head()
        results['example_duplicates'] = duplicated_values.to_dict()
        
        raise DataValidationError(
            f"Column '{column}' expected to be unique but has {duplicate_count} duplicates"
        )
    
    return results


def validate_relationships(df_parent: pd.DataFrame, df_child: pd.DataFrame,
                          parent_key: str, child_key: str) -> Dict[str, Any]:
    """
    Validate foreign key relationships between DataFrames.
    
    Args:
        df_parent: Parent DataFrame
        df_child: Child DataFrame
        parent_key: Key column in parent
        child_key: Key column in child
        
    Returns:
        Dict with relationship validation results
    """
    # Get unique keys
    parent_keys = set(df_parent[parent_key].dropna().unique())
    child_keys = set(df_child[child_key].dropna().unique())
    
    # Find orphans (child records without parent)
    orphan_keys = child_keys - parent_keys
    orphan_count = df_child[df_child[child_key].isin(orphan_keys)].shape[0]
    
    # Find childless parents
    childless_keys = parent_keys - child_keys
    childless_count = df_parent[df_parent[parent_key].isin(childless_keys)].shape[0]
    
    results = {
        'parent_key_count': len(parent_keys),
        'child_key_count': len(child_keys),
        'orphan_count': orphan_count,
        'orphan_percentage': (orphan_count / len(df_child)) * 100,
        'childless_count': childless_count,
        'childless_percentage': (childless_count / len(df_parent)) * 100,
        'orphan_keys_sample': list(orphan_keys)[:5],
        'childless_keys_sample': list(childless_keys)[:5]
    }
    
    if orphan_count > 0:
        logger.warning(f"Found {orphan_count} orphan records in child table")
    
    return results


def validate_data_consistency(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate data consistency based on custom rules.
    
    Args:
        df: DataFrame to validate
        rules: List of rule dictionaries with keys:
            - name: Rule name
            - condition: Lambda function that returns boolean Series
            - error_message: Message if rule fails
            
    Returns:
        List of rule violations
    """
    violations = []
    
    for rule in rules:
        try:
            # Apply rule condition
            mask = rule['condition'](df)
            violation_count = (~mask).sum()
            
            if violation_count > 0:
                violation = {
                    'rule_name': rule['name'],
                    'violation_count': violation_count,
                    'violation_percentage': (violation_count / len(df)) * 100,
                    'error_message': rule['error_message']
                }
                
                # Get sample of violating records
                if violation_count > 0:
                    sample_indices = df[~mask].index[:5].tolist()
                    violation['sample_indices'] = sample_indices
                
                violations.append(violation)
                logger.warning(f"Rule '{rule['name']}' violated by {violation_count} records")
                
        except Exception as e:
            logger.error(f"Error evaluating rule '{rule.get('name', 'unknown')}': {str(e)}")
            violations.append({
                'rule_name': rule.get('name', 'unknown'),
                'error': str(e)
            })
    
    return violations


def generate_data_quality_report(df: pd.DataFrame, 
                               column_specs: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        column_specs: Optional dict with column specifications
        
    Returns:
        Dict with quality metrics
    """
    report = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'column_count': len(df.columns),
        'row_count': len(df),
        'duplicate_rows': df.duplicated().sum(),
        'column_analysis': {}
    }
    
    # Analyze each column
    for col in df.columns:
        col_analysis = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isna().sum(),
            'null_percentage': (df[col].isna().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        # Additional analysis for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_analysis.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'negative': (df[col] < 0).sum()
            })
        
        # Additional analysis for string columns
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            col_analysis.update({
                'empty_strings': (df[col] == '').sum(),
                'avg_length': df[col].dropna().astype(str).str.len().mean(),
                'max_length': df[col].dropna().astype(str).str.len().max()
            })
        
        report['column_analysis'][col] = col_analysis
    
    # Apply column specifications if provided
    if column_specs:
        report['validation_results'] = {}
        for col, spec in column_specs.items():
            if col not in df.columns:
                continue
            
            col_results = {}
            
            # Check type
            if 'type' in spec:
                try:
                    validate_column_types(df, {col: spec['type']}, strict=False)
                    col_results['type_valid'] = True
                except:
                    col_results['type_valid'] = False
            
            # Check range
            if 'min' in spec or 'max' in spec:
                try:
                    range_results = validate_numeric_range(
                        df, col, 
                        min_value=spec.get('min'),
                        max_value=spec.get('max'),
                        allow_null=spec.get('allow_null', True)
                    )
                    col_results['range_valid'] = range_results['out_of_range_count'] == 0
                except:
                    col_results['range_valid'] = False
            
            # Check valid values
            if 'valid_values' in spec:
                try:
                    cat_results = validate_categorical_values(
                        df, col, 
                        valid_values=spec['valid_values'],
                        allow_null=spec.get('allow_null', True)
                    )
                    col_results['values_valid'] = cat_results['invalid_count'] == 0
                except:
                    col_results['values_valid'] = False
            
            report['validation_results'][col] = col_results
    
    return report


if __name__ == "__main__":
    # Test the validator functions
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Create sample data
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, -5, 150],  # Invalid ages
        'category': ['A', 'B', 'C', 'D', 'X'],  # X is invalid
        'score': [0.5, 0.8, 0.9, 1.2, -0.1]  # Out of range
    })
    
    print("Testing data validator functions...")
    
    # Test DataFrame validation
    try:
        validate_dataframe(test_df, "test_df")
        print(" DataFrame validation passed")
    except DataValidationError as e:
        print(f" DataFrame validation failed: {e}")
    
    # Test required columns
    try:
        validate_required_columns(test_df, ['id', 'name', 'age'], "test_df")
        print(" Required columns validation passed")
    except DataValidationError as e:
        print(f" Required columns validation failed: {e}")
    
    # Test numeric range
    try:
        results = validate_numeric_range(test_df, 'age', min_value=0, max_value=120)
        print(f" Age range validation should have failed but didn't")
    except DataValidationError as e:
        print(f" Age range validation correctly failed: {e}")
    
    # Test categorical values
    try:
        results = validate_categorical_values(test_df, 'category', ['A', 'B', 'C', 'D'])
        print(f" Category validation should have failed but didn't")
    except DataValidationError as e:
        print(f" Category validation correctly failed: {e}")
    
    # Generate quality report
    print("\nGenerating data quality report...")
    report = generate_data_quality_report(test_df)
    print(f"Shape: {report['shape']}")
    print(f"Memory usage: {report['memory_usage_mb']:.2f} MB")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    
    print("\nData validator tests completed!")