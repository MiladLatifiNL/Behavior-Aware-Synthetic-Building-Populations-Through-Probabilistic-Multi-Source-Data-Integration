"""
Match quality assessment for probabilistic record linkage.

This module provides tools to evaluate the quality of matches produced
by the Fellegi-Sunter framework, including various quality metrics and
diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MatchQualityAssessor:
    """
    Assess the quality of probabilistic matches.
    
    Provides various metrics and diagnostics to evaluate match quality
    without ground truth labels.
    """
    
    def __init__(self):
        """Initialize quality assessor."""
        self.metrics = {}
        self.diagnostics = {}
    
    def assess_matches(self, matched_data: pd.DataFrame, 
                      match_details: pd.DataFrame) -> Dict[str, float]:
        """
        Comprehensive assessment of match quality.
        
        Args:
            matched_data: Final matched dataset
            match_details: Detailed match information (weights, probabilities)
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Assessing match quality")
        
        # Calculate various quality metrics
        self.metrics['coverage_rate'] = self._calculate_coverage_rate(matched_data)
        self.metrics['avg_match_weight'] = self._calculate_avg_weight(match_details)
        self.metrics['avg_match_probability'] = self._calculate_avg_probability(match_details)
        self.metrics['weight_distribution'] = self._analyze_weight_distribution(match_details)
        self.metrics['uniqueness_score'] = self._calculate_uniqueness(match_details)
        self.metrics['consistency_score'] = self._calculate_consistency(matched_data)
        
        # Field-level metrics
        field_metrics = self._calculate_field_metrics(match_details)
        self.metrics.update(field_metrics)
        
        # Generate diagnostics
        self.diagnostics = self._generate_diagnostics(matched_data, match_details)
        
        return self.metrics
    
    def _calculate_coverage_rate(self, matched_data: pd.DataFrame) -> float:
        """
        Calculate the proportion of records that were matched.
        
        Args:
            matched_data: Matched dataset
            
        Returns:
            Coverage rate between 0 and 1
        """
        if 'has_recs_match' in matched_data.columns:
            return matched_data['has_recs_match'].mean()
        return 0.0
    
    def _calculate_avg_weight(self, match_details: pd.DataFrame) -> float:
        """
        Calculate average match weight.
        
        Args:
            match_details: Match details with weights
            
        Returns:
            Average weight
        """
        if 'weight' in match_details.columns and len(match_details) > 0:
            return match_details['weight'].mean()
        return 0.0
    
    def _calculate_avg_probability(self, match_details: pd.DataFrame) -> float:
        """
        Calculate average match probability.
        
        Args:
            match_details: Match details with probabilities
            
        Returns:
            Average probability
        """
        if 'probability' in match_details.columns and len(match_details) > 0:
            return match_details['probability'].mean()
        return 0.0
    
    def _analyze_weight_distribution(self, match_details: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the distribution of match weights.
        
        Args:
            match_details: Match details with weights
            
        Returns:
            Distribution statistics
        """
        if 'weight' not in match_details.columns or len(match_details) == 0:
            return {'min': 0, 'max': 0, 'std': 0, 'q25': 0, 'q50': 0, 'q75': 0}
        
        weights = match_details['weight']
        return {
            'min': weights.min(),
            'max': weights.max(),
            'std': weights.std(),
            'q25': weights.quantile(0.25),
            'q50': weights.quantile(0.50),
            'q75': weights.quantile(0.75)
        }
    
    def _calculate_uniqueness(self, match_details: pd.DataFrame) -> float:
        """
        Calculate uniqueness score - how unique are the matches.
        
        A score of 1.0 means all matches are one-to-one.
        Lower scores indicate many-to-one or one-to-many matches.
        
        Args:
            match_details: Match details
            
        Returns:
            Uniqueness score between 0 and 1
        """
        if len(match_details) == 0:
            return 0.0
        
        # Count unique indices
        unique_idx1 = match_details['idx1'].nunique()
        unique_idx2 = match_details['idx2'].nunique()
        total_matches = len(match_details)
        
        # Perfect uniqueness: each record matched to exactly one other
        max_possible_unique = min(unique_idx1, unique_idx2)
        
        if max_possible_unique > 0:
            uniqueness = total_matches / max_possible_unique
            return min(1.0 / uniqueness, 1.0)  # Normalize to [0, 1]
        
        return 0.0
    
    def _calculate_consistency(self, matched_data: pd.DataFrame) -> float:
        """
        Calculate consistency of matched records.
        
        Checks if matched records have consistent values in key fields.
        
        Args:
            matched_data: Matched dataset
            
        Returns:
            Consistency score between 0 and 1
        """
        consistency_checks = []
        
        # Only check matched records
        if 'has_recs_match' in matched_data.columns:
            matched_only = matched_data[matched_data['has_recs_match']]
        else:
            return 1.0  # No matches to check
        
        if len(matched_only) == 0:
            return 1.0
        
        # Check household size consistency
        if 'NP' in matched_only.columns and 'recs_total_rooms' in matched_only.columns:
            # Rough check: more people should correlate with more rooms
            valid_mask = matched_only[['NP', 'recs_total_rooms']].notna().all(axis=1)
            if valid_mask.sum() > 0:
                valid_data = matched_only[valid_mask]
                correlation = valid_data['NP'].corr(valid_data['recs_total_rooms'])
                if not np.isnan(correlation):
                    consistency_checks.append(max(0, correlation))  # Positive correlation expected
        
        # Check income vs energy cost consistency
        if 'HINCP' in matched_only.columns and 'recs_total_energy_cost_annual' in matched_only.columns:
            valid_mask = (
                (matched_only['HINCP'] > 0) & 
                matched_only['recs_total_energy_cost_annual'].notna()
            )
            if valid_mask.sum() > 0:
                valid_data = matched_only[valid_mask]
                # Energy burden should be reasonable (1-10% of income)
                energy_burden = valid_data['recs_total_energy_cost_annual'] / valid_data['HINCP']
                reasonable_burden = ((energy_burden >= 0.01) & (energy_burden <= 0.10)).mean()
                consistency_checks.append(reasonable_burden)
        
        # Return average of all consistency checks
        if consistency_checks:
            return np.mean(consistency_checks)
        return 1.0
    
    def _calculate_field_metrics(self, match_details: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metrics for individual matching fields.
        
        Args:
            match_details: Match details with field similarities
            
        Returns:
            Field-level metrics
        """
        field_metrics = {}
        
        # Find similarity columns
        similarity_cols = [col for col in match_details.columns if col.endswith('_similarity')]
        
        for col in similarity_cols:
            field_name = col.replace('_similarity', '')
            
            if len(match_details) > 0:
                # Average similarity for this field
                avg_sim = match_details[col].mean()
                field_metrics[f'{field_name}_avg_similarity'] = avg_sim
                
                # Agreement rate (similarity >= 0.88)
                agree_rate = (match_details[col] >= 0.88).mean()
                field_metrics[f'{field_name}_agreement_rate'] = agree_rate
        
        return field_metrics
    
    def _generate_diagnostics(self, matched_data: pd.DataFrame,
                             match_details: pd.DataFrame) -> Dict:
        """
        Generate detailed diagnostics for match quality.
        
        Args:
            matched_data: Matched dataset
            match_details: Match details
            
        Returns:
            Diagnostic information
        """
        diagnostics = {
            'total_records': len(matched_data),
            'matched_records': matched_data['has_recs_match'].sum() if 'has_recs_match' in matched_data.columns else 0,
            'match_details_count': len(match_details)
        }
        
        # Weight distribution by classification
        if 'classification' in match_details.columns and 'weight' in match_details.columns:
            weight_by_class = match_details.groupby('classification')['weight'].agg([
                'count', 'mean', 'min', 'max'
            ]).to_dict('index')
            diagnostics['weight_by_classification'] = weight_by_class
        
        # Multiple match analysis
        if 'idx1' in match_details.columns:
            idx1_counts = match_details['idx1'].value_counts()
            diagnostics['records_with_multiple_matches'] = (idx1_counts > 1).sum()
            diagnostics['max_matches_per_record'] = idx1_counts.max() if len(idx1_counts) > 0 else 0
        
        return diagnostics
    
    def create_quality_report(self) -> str:
        """
        Create a text report of match quality assessment.
        
        Returns:
            Quality report as string
        """
        report = []
        report.append("=" * 60)
        report.append("MATCH QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\nOverall Quality Metrics:")
        report.append("-" * 40)
        
        metric_order = [
            'coverage_rate',
            'avg_match_weight',
            'avg_match_probability',
            'uniqueness_score',
            'consistency_score'
        ]
        
        for metric in metric_order:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Weight distribution
        if 'weight_distribution' in self.metrics:
            report.append("\nWeight Distribution:")
            report.append("-" * 40)
            dist = self.metrics['weight_distribution']
            report.append(f"Min: {dist['min']:.2f}")
            report.append(f"Q25: {dist['q25']:.2f}")
            report.append(f"Median: {dist['q50']:.2f}")
            report.append(f"Q75: {dist['q75']:.2f}")
            report.append(f"Max: {dist['max']:.2f}")
            report.append(f"Std Dev: {dist['std']:.2f}")
        
        # Field-level metrics
        field_metrics = {k: v for k, v in self.metrics.items() 
                        if k.endswith('_similarity') or k.endswith('_agreement_rate')}
        
        if field_metrics:
            report.append("\nField-Level Quality:")
            report.append("-" * 40)
            
            # Group by field
            fields = set()
            for key in field_metrics:
                field = key.replace('_avg_similarity', '').replace('_agreement_rate', '')
                fields.add(field)
            
            for field in sorted(fields):
                avg_key = f'{field}_avg_similarity'
                agree_key = f'{field}_agreement_rate'
                
                if avg_key in field_metrics or agree_key in field_metrics:
                    report.append(f"\n{field}:")
                    if avg_key in field_metrics:
                        report.append(f"  Average Similarity: {field_metrics[avg_key]:.3f}")
                    if agree_key in field_metrics:
                        report.append(f"  Agreement Rate: {field_metrics[agree_key]:.1%}")
        
        # Diagnostics
        if self.diagnostics:
            report.append("\nDiagnostic Information:")
            report.append("-" * 40)
            
            if 'total_records' in self.diagnostics:
                report.append(f"Total Records: {self.diagnostics['total_records']:,}")
            if 'matched_records' in self.diagnostics:
                report.append(f"Matched Records: {self.diagnostics['matched_records']:,}")
            if 'records_with_multiple_matches' in self.diagnostics:
                report.append(f"Records with Multiple Matches: {self.diagnostics['records_with_multiple_matches']:,}")
            if 'max_matches_per_record' in self.diagnostics:
                report.append(f"Max Matches per Record: {self.diagnostics['max_matches_per_record']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def get_match_distribution_plot_data(self, match_details: pd.DataFrame) -> Dict:
        """
        Get data for plotting match weight distribution.
        
        Args:
            match_details: Match details
            
        Returns:
            Plot data dictionary
        """
        if 'weight' not in match_details.columns or len(match_details) == 0:
            return {}
        
        return {
            'weights': match_details['weight'].tolist(),
            'classifications': match_details['classification'].tolist() if 'classification' in match_details.columns else None,
            'bins': np.linspace(
                match_details['weight'].min(),
                match_details['weight'].max(),
                50
            ).tolist()
        }


def analyze_unmatched_records(unmatched_data: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze characteristics of unmatched records.
    
    Args:
        unmatched_data: Records that couldn't be matched
        
    Returns:
        Analysis results
    """
    analysis = {
        'total_unmatched': len(unmatched_data),
        'characteristics': {}
    }
    
    # Analyze key fields
    fields_to_analyze = [
        'STATE', 'income_quintile', 'household_size_cat',
        'building_type_simple', 'urban_rural'
    ]
    
    for field in fields_to_analyze:
        if field in unmatched_data.columns:
            # Get value counts
            value_counts = unmatched_data[field].value_counts()
            analysis['characteristics'][field] = {
                'top_values': value_counts.head(5).to_dict(),
                'unique_count': unmatched_data[field].nunique(),
                'missing_rate': unmatched_data[field].isna().mean()
            }
    
    return analysis


if __name__ == "__main__":
    # Test quality assessor
    print("Testing Match Quality Assessor\n")
    
    # Create sample matched data
    np.random.seed(42)
    n_records = 1000
    n_matched = 800
    
    matched_data = pd.DataFrame({
        'id': range(n_records),
        'has_recs_match': [True] * n_matched + [False] * (n_records - n_matched),
        'NP': np.random.randint(1, 6, n_records),
        'HINCP': np.random.randint(20000, 150000, n_records),
        'recs_total_rooms': np.random.randint(3, 10, n_records),
        'recs_total_energy_cost_annual': np.random.randint(1000, 5000, n_records)
    })
    
    # Create sample match details
    match_details = pd.DataFrame({
        'idx1': np.random.choice(range(n_matched), n_matched, replace=False),
        'idx2': np.random.choice(range(500), n_matched, replace=True),
        'weight': np.random.normal(5, 2, n_matched),
        'probability': np.random.beta(8, 2, n_matched),
        'classification': np.random.choice(['match', 'possible'], n_matched, p=[0.7, 0.3]),
        'STATE_similarity': np.random.beta(9, 1, n_matched),
        'income_quintile_similarity': np.random.beta(7, 3, n_matched),
        'household_size_cat_similarity': np.random.beta(8, 2, n_matched)
    })
    
    # Test assessment
    assessor = MatchQualityAssessor()
    metrics = assessor.assess_matches(matched_data, match_details)
    
    # Print report
    print(assessor.create_quality_report())
    
    # Test unmatched analysis
    unmatched = matched_data[~matched_data['has_recs_match']]
    unmatched['STATE'] = np.random.choice(['CA', 'TX', 'NY'], len(unmatched))
    unmatched['income_quintile'] = np.random.choice(['q1', 'q2', 'q3', 'q4', 'q5'], len(unmatched))
    
    print("\nUnmatched Records Analysis:")
    unmatched_analysis = analyze_unmatched_records(unmatched)
    print(f"Total unmatched: {unmatched_analysis['total_unmatched']}")
