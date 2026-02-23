"""
Fellegi-Sunter probabilistic record linkage framework.

This module implements the core Fellegi-Sunter model for probabilistic
record linkage, including likelihood ratio calculation and pair classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from .string_comparators import compare_field, get_similarity_level

logger = logging.getLogger(__name__)


@dataclass
class ComparisonField:
    """Configuration for a field comparison."""
    name: str
    field_type: str = 'string'  # 'string', 'numeric', 'categorical'
    comparison_method: str = 'jaro_winkler'
    weight: float = 1.0
    m_probability: Optional[float] = None  # P(agree|match)
    u_probability: Optional[float] = None  # P(agree|non-match)


@dataclass
class MatchWeights:
    """Container for match weights and classification."""
    idx1: int
    idx2: int
    agreement_vector: List[float]
    weight: float
    probability: float
    classification: str = 'possible'  # 'match', 'non-match', 'possible'


class FellegiSunterMatcher:
    """
    Fellegi-Sunter probabilistic record linkage model.
    
    This implements the classic Fellegi-Sunter framework for record linkage,
    calculating likelihood ratios based on agreement patterns.
    """
    
    def __init__(self, comparison_fields: List[Union[ComparisonField, dict]]):
        """
        Initialize matcher with comparison field specifications.
        
        Args:
            comparison_fields: List of ComparisonField objects or dicts
        """
        # Convert dicts to ComparisonField objects
        self.fields = []
        for field in comparison_fields:
            if isinstance(field, dict):
                self.fields.append(ComparisonField(**field))
            else:
                self.fields.append(field)
        
        # Initialize probabilities
        self.m_probs = {}  # Match probabilities
        self.u_probs = {}  # Non-match probabilities
        
        # Set default probabilities if provided
        for field in self.fields:
            if field.m_probability is not None:
                self.m_probs[field.name] = field.m_probability
            if field.u_probability is not None:
                self.u_probs[field.name] = field.u_probability
        
        # Classification thresholds
        self.upper_threshold = None
        self.lower_threshold = None
        
        # Storage for comparison results
        self.comparison_results = []
    
    def set_probabilities(self, m_probs: Dict[str, float], u_probs: Dict[str, float]):
        """
        Set m and u probabilities for fields.
        
        Args:
            m_probs: Dictionary of match probabilities by field name
            u_probs: Dictionary of non-match probabilities by field name
        """
        self.m_probs.update(m_probs)
        self.u_probs.update(u_probs)
    
    def set_thresholds(self, upper: float, lower: float):
        """
        Set classification thresholds.
        
        Args:
            upper: Upper threshold (above = match)
            lower: Lower threshold (below = non-match)
        """
        if lower >= upper:
            raise ValueError("Lower threshold must be less than upper threshold")
        
        self.upper_threshold = upper
        self.lower_threshold = lower
    
    def _get_field_value(self, record: Union[dict, pd.Series], field_name: str) -> Any:
        """
        Safely extract field value from record regardless of type.
        
        Args:
            record: Record as dict or pandas Series
            field_name: Name of field to extract
            
        Returns:
            Field value or None if not found
        """
        if record is None:
            return None
            
        try:
            # Handle pandas Series specifically
            if isinstance(record, pd.Series):
                # Use .get() method which returns None if key doesn't exist
                value = record.get(field_name)
                # Check for pandas NaN and convert to None
                if pd.isna(value):
                    return None
                return value
            
            # Handle regular dictionaries
            elif isinstance(record, dict):
                value = record.get(field_name)
                # Check for numpy/pandas NaN and convert to None
                if pd.isna(value):
                    return None
                return value
            
            # Handle DataFrame row (which might be a Series)
            elif hasattr(record, 'loc') and hasattr(record, 'index'):
                # This is likely a DataFrame row accessed via iterrows()
                if field_name in record.index:
                    value = record[field_name]
                    if pd.isna(value):
                        return None
                    return value
                return None
            
            # Handle other dict-like objects
            elif hasattr(record, '__getitem__'):
                try:
                    value = record[field_name]
                    if pd.isna(value):
                        return None
                    return value
                except (KeyError, IndexError, TypeError):
                    return None
            
            # Try attribute access as last resort
            else:
                value = getattr(record, field_name, None)
                if pd.isna(value):
                    return None
                return value
                
        except Exception as e:
            logger.debug(f"Error accessing field {field_name} from {type(record)}: {e}")
            return None
    
    def calculate_agreement_patterns(self, record1: Union[dict, pd.Series], 
                                   record2: Union[dict, pd.Series]) -> List[float]:
        """
        Compare two records field by field.
        
        Args:
            record1: First record
            record2: Second record
            
        Returns:
            Agreement vector with similarity scores for each field
        """
        agreement_vector = []
        
        for field in self.fields:
            # Use unified field extraction method
            val1 = self._get_field_value(record1, field.name)
            val2 = self._get_field_value(record2, field.name)
            
            similarity = compare_field(
                val1, val2,
                field_type=field.field_type,
                comparison_method=field.comparison_method
            )
            
            agreement_vector.append(similarity)
        
        return agreement_vector
    
    def compute_match_weight(self, agreement_pattern: List[float]) -> float:
        """
        Calculate log-likelihood ratio for an agreement pattern.
        
        Uses the Fellegi-Sunter formula:
        weight = Σ log₂(m_i/u_i) for agreements + Σ log₂((1-m_i)/(1-u_i)) for disagreements
        
        Args:
            agreement_pattern: List of similarity scores
            
        Returns:
            Match weight (log-likelihood ratio)
        """
        weight = 0.0
        
        for i, similarity in enumerate(agreement_pattern):
            field_name = self.fields[i].name
            
            # Get probabilities (use defaults if not set)
            m_prob = self.m_probs.get(field_name, 0.9)
            u_prob = self.u_probs.get(field_name, 0.1)
            
            # Prevent log(0) errors
            m_prob = max(min(m_prob, 0.9999), 0.0001)
            u_prob = max(min(u_prob, 0.9999), 0.0001)
            
            # Get similarity level
            level = get_similarity_level(similarity)
            
            if level == "agree":
                # Full agreement
                weight += np.log2(m_prob / u_prob) * self.fields[i].weight
            elif level == "disagree":
                # Full disagreement
                weight += np.log2((1 - m_prob) / (1 - u_prob)) * self.fields[i].weight
            else:
                # Partial agreement - interpolate
                weight += self._partial_agreement_weight(
                    similarity, m_prob, u_prob
                ) * self.fields[i].weight
        
        return weight
    
    def _partial_agreement_weight(self, similarity: float, m_prob: float, u_prob: float) -> float:
        """
        Calculate weight for partial agreements.
        
        Uses linear interpolation between full agreement and disagreement weights
        based on the similarity score.
        """
        agree_weight = np.log2(m_prob / u_prob)
        disagree_weight = np.log2((1 - m_prob) / (1 - u_prob))
        
        # Linear interpolation based on similarity
        return agree_weight * similarity + disagree_weight * (1 - similarity)
    
    def compute_match_probability(self, weight: float, prior_match: float = 0.1) -> float:
        """
        Convert weight to match probability using Bayes' theorem.
        
        Args:
            weight: Log-likelihood ratio
            prior_match: Prior probability of a match
            
        Returns:
            Posterior probability of match
        """
        # Convert log-weight to likelihood ratio
        likelihood_ratio = 2 ** weight
        
        # Apply Bayes' theorem
        odds_match = likelihood_ratio * (prior_match / (1 - prior_match))
        probability = odds_match / (1 + odds_match)
        
        return probability
    
    def classify_pair(self, weight: float) -> str:
        """
        Classify a record pair based on weight and thresholds.
        
        Args:
            weight: Match weight
            
        Returns:
            Classification: 'match', 'non-match', or 'possible'
        """
        if self.upper_threshold is None or self.lower_threshold is None:
            return 'possible'
        
        if weight >= self.upper_threshold:
            return 'match'
        elif weight <= self.lower_threshold:
            return 'non-match'
        else:
            return 'possible'
    
    def compare_records(self, record1: Union[dict, pd.Series], 
                       record2: Union[dict, pd.Series],
                       idx1: Optional[int] = None,
                       idx2: Optional[int] = None) -> MatchWeights:
        """
        Complete comparison of two records.
        
        Args:
            record1: First record
            record2: Second record
            idx1: Optional index of first record
            idx2: Optional index of second record
            
        Returns:
            MatchWeights object with results
        """
        # Calculate agreement pattern
        agreement_vector = self.calculate_agreement_patterns(record1, record2)
        
        # Calculate weight
        weight = self.compute_match_weight(agreement_vector)
        
        # Calculate probability
        probability = self.compute_match_probability(weight)
        
        # Classify
        classification = self.classify_pair(weight)
        
        return MatchWeights(
            idx1=idx1,
            idx2=idx2,
            agreement_vector=agreement_vector,
            weight=weight,
            probability=probability,
            classification=classification
        )
    
    def compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        candidate_pairs: List[Tuple[int, int]],
                        include_levels: bool = False) -> pd.DataFrame:
        """
        Compare all candidate pairs between two datasets.
        
        Args:
            df1: First dataset
            df2: Second dataset
            candidate_pairs: List of (idx1, idx2) pairs to compare
            include_levels: When True, include per-field categorical level strings
                (e.g., '<field>_level'). Default False to reduce memory (avoids
                object dtype columns) when only numeric similarities are needed.
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for idx1, idx2 in candidate_pairs:
            record1 = df1.loc[idx1]
            record2 = df2.loc[idx2]
            
            match_result = self.compare_records(record1, record2, idx1, idx2)
            
            # Convert to dict for DataFrame
            result_dict = {
                'idx1': match_result.idx1,
                'idx2': match_result.idx2,
                'weight': match_result.weight,
                'probability': match_result.probability,
                'classification': match_result.classification
            }
            
            # Add individual field agreements
            for i, field in enumerate(self.fields):
                result_dict[f'{field.name}_similarity'] = match_result.agreement_vector[i]
                if include_levels:
                    result_dict[f'{field.name}_level'] = get_similarity_level(
                        match_result.agreement_vector[i]
                    )
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def get_optimal_thresholds(self, comparison_results: pd.DataFrame,
                              target_precision: float = 0.95) -> Tuple[float, float]:
        """
        Calculate optimal thresholds based on weight distribution.
        
        Args:
            comparison_results: DataFrame with comparison results
            target_precision: Desired precision for matches
            
        Returns:
            Tuple of (upper_threshold, lower_threshold)
        """
        weights = comparison_results['weight'].values
        
        # Handle empty weights
        if len(weights) == 0:
            logger.warning("No comparison weights found - using default thresholds")
            return 0.0, -5.0  # Default thresholds
        
        # Sort weights
        sorted_weights = np.sort(weights)[::-1]  # Descending
        
        # Find upper threshold for target precision
        # This is simplified - in practice you'd use labeled data or EM results
        n_target_matches = int(len(weights) * 0.1)  # Assume 10% are true matches
        upper_threshold = sorted_weights[n_target_matches] if n_target_matches < len(weights) else 0
        
        # Set lower threshold
        # Common practice: set it where we're confident of non-matches
        lower_threshold = np.percentile(weights, 5)
        
        # Ensure valid thresholds
        if lower_threshold >= upper_threshold:
            gap = (np.max(weights) - np.min(weights)) * 0.2
            lower_threshold = upper_threshold - gap
        
        return upper_threshold, lower_threshold
    
    def calculate_field_statistics(self, comparison_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each comparison field.
        
        Args:
            comparison_results: DataFrame with comparison results
            
        Returns:
            DataFrame with field statistics
        """
        stats = []
        
        for field in self.fields:
            sim_col = f'{field.name}_similarity'
            level_col = f'{field.name}_level'
            
            if sim_col in comparison_results.columns:
                field_stats = {
                    'field': field.name,
                    'mean_similarity': comparison_results[sim_col].mean(),
                    'std_similarity': comparison_results[sim_col].std(),
                    'agree_rate': (comparison_results[level_col] == 'agree').mean(),
                    'partial_rate': (comparison_results[level_col].isin(['partial_agree', 'mostly_agree'])).mean(),
                    'disagree_rate': (comparison_results[level_col] == 'disagree').mean(),
                    'm_probability': self.m_probs.get(field.name, None),
                    'u_probability': self.u_probs.get(field.name, None)
                }
                stats.append(field_stats)
        
        return pd.DataFrame(stats)


def create_comparison_summary(results: pd.DataFrame) -> str:
    """
    Create a summary report of comparison results.
    
    Args:
        results: DataFrame with comparison results
        
    Returns:
        Summary report string
    """
    report = []
    report.append("=" * 60)
    report.append("FELLEGI-SUNTER COMPARISON SUMMARY")
    report.append("=" * 60)
    report.append(f"Total comparisons: {len(results):,}")
    report.append("")
    
    # Weight statistics
    report.append("Weight Distribution:")
    report.append(f"  Mean: {results['weight'].mean():.2f}")
    report.append(f"  Std: {results['weight'].std():.2f}")
    report.append(f"  Min: {results['weight'].min():.2f}")
    report.append(f"  Max: {results['weight'].max():.2f}")
    report.append("")
    
    # Classification counts
    if 'classification' in results.columns:
        report.append("Classification:")
        for cls in ['match', 'possible', 'non-match']:
            count = (results['classification'] == cls).sum()
            pct = count / len(results) * 100
            report.append(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test Fellegi-Sunter matcher
    print("Testing Fellegi-Sunter Matcher\n")
    
    # Define comparison fields
    fields = [
        ComparisonField(name='state', field_type='categorical'),
        ComparisonField(name='income', field_type='numeric'),
        ComparisonField(name='household_size', field_type='numeric'),
        ComparisonField(name='building_type', field_type='categorical')
    ]
    
    # Create matcher
    matcher = FellegiSunterMatcher(fields)
    
    # Set example probabilities
    m_probs = {
        'state': 0.95,
        'income': 0.85,
        'household_size': 0.90,
        'building_type': 0.88
    }
    u_probs = {
        'state': 0.05,
        'income': 0.20,
        'household_size': 0.15,
        'building_type': 0.12
    }
    matcher.set_probabilities(m_probs, u_probs)
    
    # Test record comparison
    record1 = {
        'state': 'CA',
        'income': 75000,
        'household_size': 3,
        'building_type': 'single_family'
    }
    
    record2 = {
        'state': 'CA',
        'income': 72000,
        'household_size': 3,
        'building_type': 'single_family'
    }
    
    result = matcher.compare_records(record1, record2)
    
    print("Record Comparison:")
    print(f"Agreement vector: {[f'{x:.2f}' for x in result.agreement_vector]}")
    print(f"Weight: {result.weight:.2f}")
    print(f"Match probability: {result.probability:.2%}")
    print(f"Classification: {result.classification}")
    
    # Test with different records
    record3 = {
        'state': 'NY',
        'income': 45000,
        'household_size': 1,
        'building_type': 'apartment'
    }
    
    result2 = matcher.compare_records(record1, record3)
    print(f"\nDifferent records:")
    print(f"Weight: {result2.weight:.2f}")
    print(f"Match probability: {result2.probability:.2%}")