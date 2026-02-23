"""
Data type utilities for ensuring consistent types across the pipeline.

This module provides functions to handle categorical data type conversions
and ensure consistency, particularly for pd.cut() and pd.qcut() operations.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def safe_cut(series: pd.Series, bins, labels=None, **kwargs) -> pd.Series:
    """
    Wrapper for pd.cut that always returns string dtype.
    
    Args:
        series: Series to bin
        bins: Number of bins or bin edges
        labels: Labels for the bins
        **kwargs: Additional arguments for pd.cut
        
    Returns:
        Series with string dtype
    """
    result = pd.cut(series, bins=bins, labels=labels, **kwargs)
    
    # Convert categorical to string
    if isinstance(result.dtype, pd.CategoricalDtype):
        return result.astype(str)
    
    return result


def safe_qcut(series: pd.Series, q, labels=None, **kwargs) -> pd.Series:
    """
    Wrapper for pd.qcut that always returns string dtype.
    
    Args:
        series: Series to bin
        q: Number of quantiles or quantile edges
        labels: Labels for the bins
        **kwargs: Additional arguments for pd.qcut
        
    Returns:
        Series with string dtype
    """
    result = pd.qcut(series, q=q, labels=labels, **kwargs)
    
    # Convert categorical to string
    if isinstance(result.dtype, pd.CategoricalDtype):
        return result.astype(str)
    
    return result


def ensure_string_dtype(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Ensure specified columns (or all categorical columns) are string dtype.
    
    Args:
        df: DataFrame to process
        columns: List of columns to convert, or None for all categorical columns
        
    Returns:
        DataFrame with converted columns
    """
    if columns is None:
        # Find all categorical columns
        columns = [col for col in df.columns 
                  if isinstance(df[col].dtype, pd.CategoricalDtype)]
    
    for col in columns:
        if col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)
    
    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types across a DataFrame for consistency.
    
    Conversions:
    - Categorical -> string
    - Object with numeric values -> appropriate numeric type
    - Boolean-like values -> bool
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        DataFrame with standardized types
    """
    df = df.copy()
    
    for col in df.columns:
        # Convert categorical to string
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
        
        # Try to infer better types for object columns
        elif df[col].dtype == 'object':
            # Try numeric conversion
            try:
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                if numeric_values.notna().sum() > len(df) * 0.9:  # 90% non-null
                    df[col] = numeric_values
            except:
                pass
    
    return df


def fix_mixed_types(series: pd.Series) -> pd.Series:
    """
    Fix series with mixed types by converting to most appropriate type.
    
    Args:
        series: Series with potentially mixed types
        
    Returns:
        Series with consistent type
    """
    # Check if all non-null values are numeric
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    
    try:
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().equals(series.notna()):
            return numeric_series
    except:
        pass
    
    # Default to string for mixed types
    return series.astype(str)