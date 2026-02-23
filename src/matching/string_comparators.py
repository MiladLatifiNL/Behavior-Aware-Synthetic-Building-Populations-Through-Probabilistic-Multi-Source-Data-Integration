"""
String comparison utilities for probabilistic record linkage.

This module provides various string similarity metrics optimized
for record linkage applications, based on the Fellegi-Sunter framework.
"""

import jellyfish
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _manual_jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Manual implementation of Jaro-Winkler similarity as fallback."""
    # Jaro similarity calculation
    s1_len = len(s1)
    s2_len = len(s2)
    
    if s1_len == 0 and s2_len == 0:
        return 1.0
    if s1_len == 0 or s2_len == 0:
        return 0.0
    
    # Calculate match window
    match_window = max(s1_len, s2_len) // 2 - 1
    if match_window < 1:
        match_window = 1
    
    s1_matches = [False] * s1_len
    s2_matches = [False] * s2_len
    
    matches = 0
    transpositions = 0
    
    # Identify matches
    for i in range(s1_len):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, s2_len)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(s1_len):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    # Calculate Jaro similarity
    jaro = (matches/s1_len + matches/s2_len + (matches - transpositions/2)/matches) / 3
    
    # Calculate common prefix for Jaro-Winkler
    prefix_len = 0
    for i in range(min(s1_len, s2_len, 4)):  # Max prefix of 4
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    # Calculate Jaro-Winkler
    return jaro + prefix_len * prefix_weight * (1 - jaro)


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.
    
    The Jaro-Winkler similarity is a variant of Jaro similarity that
    gives additional weight to strings with matching prefixes.
    
    Args:
        s1: First string to compare
        s2: Second string to compare
        prefix_weight: Weight given to matching prefix (max 0.25)
        
    Returns:
        Similarity score between 0 (no match) and 1 (identical)
        
    Examples:
        >>> jaro_winkler_similarity("MARTHA", "MARHTA")
        0.961
        >>> jaro_winkler_similarity("DIXON", "DICKSONX")
        0.813
    """
    # Handle empty strings - both empty strings are identical
    if not s1 and not s2:
        return 1.0
    
    # Handle None values or one empty string
    if s1 is None or s2 is None or not s1 or not s2:
        return 0.0
    
    # Convert to strings
    s1, s2 = str(s1).upper(), str(s2).upper()
    
    # Identical strings
    if s1 == s2:
        return 1.0
    
    try:
        # Use jellyfish implementation which is optimized
        # Note: jellyfish uses jaro_winkler_similarity not jaro_winkler
        if hasattr(jellyfish, 'jaro_winkler_similarity'):
            return jellyfish.jaro_winkler_similarity(s1, s2)
        elif hasattr(jellyfish, 'jaro_similarity'):
            # Fallback to regular Jaro if Jaro-Winkler not available
            return jellyfish.jaro_similarity(s1, s2)
        else:
            # Manual implementation fallback
            return _manual_jaro_winkler(s1, s2, prefix_weight)
    except Exception as e:
        logger.warning(f"Jaro-Winkler calculation failed: {e}")
        return 0.0


def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein distance between two strings.
    
    Args:
        s1: First string to compare
        s2: Second string to compare
        
    Returns:
        Similarity score between 0 (completely different) and 1 (identical)
    """
    # Handle empty strings - both empty strings are identical
    if not s1 and not s2:
        return 1.0
    
    # Handle None values or one empty string  
    if s1 is None or s2 is None or not s1 or not s2:
        return 0.0
    
    s1, s2 = str(s1).upper(), str(s2).upper()
    
    if s1 == s2:
        return 1.0
    
    try:
        # Calculate Levenshtein distance
        distance = jellyfish.levenshtein_distance(s1, s2)
        
        # Normalize by maximum possible distance
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    except Exception as e:
        logger.warning(f"Edit distance calculation failed: {e}")
        return 0.0


def soundex_match(s1: str, s2: str) -> float:
    """
    Compare phonetic encodings of strings using Soundex algorithm.
    
    Args:
        s1: First string to compare
        s2: Second string to compare
        
    Returns:
        1.0 if Soundex codes match, 0.0 otherwise
    """
    if not s1 or not s2:
        return 0.0
    
    if s1 is None or s2 is None:
        return 0.0
    
    try:
        soundex1 = jellyfish.soundex(str(s1))
        soundex2 = jellyfish.soundex(str(s2))
        
        return 1.0 if soundex1 == soundex2 else 0.0
    except Exception as e:
        logger.warning(f"Soundex calculation failed: {e}")
        return 0.0


def nysiis_match(s1: str, s2: str) -> float:
    """
    Compare phonetic encodings using NYSIIS algorithm.
    
    NYSIIS (New York State Identification and Intelligence System)
    is more accurate than Soundex for diverse names.
    
    Args:
        s1: First string to compare
        s2: Second string to compare
        
    Returns:
        1.0 if NYSIIS codes match, 0.0 otherwise
    """
    if not s1 or not s2:
        return 0.0
    
    if s1 is None or s2 is None:
        return 0.0
    
    try:
        nysiis1 = jellyfish.nysiis(str(s1))
        nysiis2 = jellyfish.nysiis(str(s2))
        
        return 1.0 if nysiis1 == nysiis2 else 0.0
    except Exception as e:
        logger.warning(f"NYSIIS calculation failed: {e}")
        return 0.0


def qgram_similarity(s1: str, s2: str, q: int = 2) -> float:
    """
    Calculate q-gram (n-gram) similarity between strings.
    
    Breaks strings into overlapping subsequences of length q
    and calculates the overlap coefficient.
    
    Args:
        s1: First string to compare
        s2: Second string to compare
        q: Length of q-grams (default 2 for bigrams)
        
    Returns:
        Similarity score between 0 and 1
    """
    if not s1 or not s2:
        return 0.0
    
    if s1 is None or s2 is None:
        return 0.0
    
    s1, s2 = str(s1).upper(), str(s2).upper()
    
    if s1 == s2:
        return 1.0
    
    # Pad strings for q-grams
    s1_padded = ' ' * (q - 1) + s1 + ' ' * (q - 1)
    s2_padded = ' ' * (q - 1) + s2 + ' ' * (q - 1)
    
    # Generate q-grams
    qgrams1 = [s1_padded[i:i+q] for i in range(len(s1_padded) - q + 1)]
    qgrams2 = [s2_padded[i:i+q] for i in range(len(s2_padded) - q + 1)]
    
    if not qgrams1 or not qgrams2:
        return 0.0
    
    # Calculate overlap coefficient
    common = len(set(qgrams1) & set(qgrams2))
    min_len = min(len(set(qgrams1)), len(set(qgrams2)))
    
    return common / min_len if min_len > 0 else 0.0


def get_similarity_level(score: float) -> str:
    """
    Convert continuous similarity score to discrete levels for EM algorithm.
    
    Levels based on empirical analysis from Fellegi-Sunter papers:
    - [0, 0.66): Disagree
    - [0.66, 0.88): Partial agree
    - [0.88, 0.94): Mostly agree
    - [0.94, 1.0]: Agree
    
    Args:
        score: Similarity score between 0 and 1
        
    Returns:
        Similarity level as string
    """
    if score < 0.66:
        return "disagree"
    elif score < 0.88:
        return "partial_agree"
    elif score < 0.94:
        return "mostly_agree"
    else:
        return "agree"


def numeric_similarity(val1: Union[int, float], val2: Union[int, float], 
                      tolerance: float = 0.2) -> float:
    """
    Calculate similarity between numeric values.
    
    Args:
        val1: First numeric value
        val2: Second numeric value
        tolerance: Relative tolerance for considering values similar
        
    Returns:
        Similarity score between 0 and 1
    """
    # Handle None/NaN values
    if val1 is None or val2 is None:
        return 0.0
    
    try:
        val1 = float(val1)
        val2 = float(val2)
    except (ValueError, TypeError):
        return 0.0
    
    if np.isnan(val1) or np.isnan(val2):
        return 0.0
    
    # Exact match
    if val1 == val2:
        return 1.0
    
    # Both zero
    if val1 == 0 and val2 == 0:
        return 1.0
    
    # Calculate relative difference
    avg_val = (abs(val1) + abs(val2)) / 2
    if avg_val == 0:
        return 1.0
    
    rel_diff = abs(val1 - val2) / avg_val
    
    # Convert to similarity score with smoother decay
    if rel_diff <= tolerance:
        # Linear decay within tolerance
        return 1.0 - (rel_diff / tolerance) * 0.3
    elif rel_diff <= tolerance * 2:
        # Slower decay in medium range
        return 0.7 - ((rel_diff - tolerance) / tolerance) * 0.4
    else:
        # Exponential decay beyond 2x tolerance
        return 0.3 * np.exp(-(rel_diff - tolerance * 2))


def categorical_similarity(cat1: str, cat2: str, 
                         similarity_matrix: Optional[dict] = None) -> float:
    """
    Calculate similarity between categorical values.
    
    Args:
        cat1: First category
        cat2: Second category
        similarity_matrix: Optional custom similarity matrix
        
    Returns:
        Similarity score between 0 and 1
    """
    if cat1 is None or cat2 is None:
        return 0.0
    
    cat1_str, cat2_str = str(cat1), str(cat2)
    
    # Exact match (case-insensitive)
    if cat1_str.upper() == cat2_str.upper():
        return 1.0
    
    # Use custom similarity matrix if provided
    if similarity_matrix:
        # Try to find match in similarity matrix with various case combinations
        for (k1, k2), score in similarity_matrix.items():
            k1_str, k2_str = str(k1), str(k2)
            # Check all combinations case-insensitively
            if (k1_str.upper(), k2_str.upper()) in [(cat1_str.upper(), cat2_str.upper()), 
                                                      (cat2_str.upper(), cat1_str.upper())]:
                return score
    
    # Default: no similarity for different categories
    return 0.0


def compare_field(val1: any, val2: any, field_type: str = 'string',
                 comparison_method: str = 'jaro_winkler') -> float:
    """
    Generic field comparison function that selects appropriate method.
    
    Args:
        val1: First value
        val2: Second value
        field_type: Type of field ('string', 'numeric', 'categorical')
        comparison_method: Method to use for string comparison
        
    Returns:
        Similarity score between 0 and 1
    """
    import pandas as pd
    import numpy as np
    
    # Enhanced NaN handling - check for various types of missing values
    if val1 is None or val2 is None:
        return 0.0
    
    # Check for pandas/numpy NaN
    try:
        if pd.isna(val1) or pd.isna(val2):
            return 0.0
    except:
        pass
    
    # Check for numpy NaN specifically
    try:
        if np.isnan(val1) or np.isnan(val2):
            return 0.0
    except (TypeError, ValueError):
        # Not a numeric type, continue
        pass
    
    # Handle empty strings
    if field_type == 'string':
        if (isinstance(val1, str) and val1.strip() == '') or (isinstance(val2, str) and val2.strip() == ''):
            if val1 == val2:  # Both empty
                return 1.0
            else:
                return 0.0
    
    if field_type == 'numeric':
        return numeric_similarity(val1, val2)
    elif field_type == 'categorical':
        return categorical_similarity(val1, val2)
    else:  # string
        if comparison_method == 'jaro_winkler':
            return jaro_winkler_similarity(val1, val2)
        elif comparison_method == 'edit_distance':
            return normalized_edit_distance(val1, val2)
        elif comparison_method == 'soundex':
            return soundex_match(val1, val2)
        elif comparison_method == 'nysiis':
            return nysiis_match(val1, val2)
        elif comparison_method == 'qgram':
            return qgram_similarity(val1, val2)
        else:
            return jaro_winkler_similarity(val1, val2)  # Default


def calculate_agreement_vector(record1: dict, record2: dict,
                             comparison_fields: list) -> list:
    """
    Calculate agreement vector for a pair of records.
    
    Args:
        record1: First record as dictionary
        record2: Second record as dictionary
        comparison_fields: List of field specifications
        
    Returns:
        List of similarity scores for each field
    """
    agreement_vector = []
    
    for field_spec in comparison_fields:
        field_name = field_spec['name']
        field_type = field_spec.get('type', 'string')
        method = field_spec.get('method', 'jaro_winkler')
        
        val1 = record1.get(field_name)
        val2 = record2.get(field_name)
        
        similarity = compare_field(val1, val2, field_type, method)
        agreement_vector.append(similarity)
    
    return agreement_vector


if __name__ == "__main__":
    # Test string comparators
    print("String Comparator Tests:")
    print("-" * 50)
    
    # Test cases
    test_pairs = [
        ("SMITH", "SMYTH"),
        ("JOHN", "JON"),
        ("MARTHA", "MARHTA"),
        ("DIXON", "DICKSONX"),
        ("APARTMENT 2B", "APT 2B"),
        ("123 MAIN ST", "123 MAIN STREET")
    ]
    
    for s1, s2 in test_pairs:
        print(f"\nComparing: '{s1}' vs '{s2}'")
        print(f"  Jaro-Winkler: {jaro_winkler_similarity(s1, s2):.3f}")
        print(f"  Edit Distance: {normalized_edit_distance(s1, s2):.3f}")
        print(f"  Soundex Match: {soundex_match(s1, s2):.3f}")
        print(f"  NYSIIS Match: {nysiis_match(s1, s2):.3f}")
        print(f"  Q-gram: {qgram_similarity(s1, s2):.3f}")
    
    # Test numeric similarity
    print("\n" + "-" * 50)
    print("Numeric Similarity Tests:")
    numeric_pairs = [(100, 100), (100, 105), (100, 110), (100, 150)]
    for n1, n2 in numeric_pairs:
        print(f"{n1} vs {n2}: {numeric_similarity(n1, n2):.3f}")
    
    # Test similarity levels
    print("\n" + "-" * 50)
    print("Similarity Level Tests:")
    scores = [0.0, 0.5, 0.7, 0.9, 0.95, 1.0]
    for score in scores:
        print(f"Score {score:.2f} → {get_similarity_level(score)}")