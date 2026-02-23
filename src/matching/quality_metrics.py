"""
Comprehensive quality metrics and diagnostics for matching evaluation.

This module provides advanced metrics for assessing matching quality including:
- Precision, Recall, F1 at various thresholds
- Mean Reciprocal Rank (MRR) and NDCG
- Match stability and consistency scores
- One-to-one vs many-to-one analysis
- Field-level agreement analysis
- Match explanation and interpretability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from scipy.stats import entropy, chi2_contingency, ks_2samp
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    calculate_all_metrics: bool = True
    top_k_values: List[int] = None
    confidence_thresholds: List[float] = None
    generate_plots: bool = True
    save_reports: bool = True
    report_format: str = 'html'  # 'html', 'json', 'markdown'
    output_dir: str = 'data/validation'
    
    def __post_init__(self):
        if self.top_k_values is None:
            self.top_k_values = [1, 3, 5, 10]
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]


class MatchQualityAssessor:
    """
    Comprehensive quality assessment for matching results.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality assessor.
        
        Args:
            config: Configuration for quality assessment
        """
        self.config = config or QualityConfig()
        self.metrics = {}
        self.field_metrics = {}
        self.diagnostics = {}
        
    def evaluate_matches(self, matches_df: pd.DataFrame, 
                        ground_truth: Optional[pd.DataFrame] = None,
                        comparison_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of matching results.
        
        Args:
            matches_df: DataFrame with match results (idx1, idx2, weight, probability)
            ground_truth: Optional DataFrame with true matches
            comparison_data: Optional DataFrame with detailed field comparisons
            
        Returns:
            Dictionary with comprehensive metrics
        """
        logger.info("Starting comprehensive match quality evaluation")
        
        # Basic statistics
        self.metrics['basic'] = self._calculate_basic_metrics(matches_df)
        
        # Match distribution analysis
        self.metrics['distribution'] = self._analyze_match_distribution(matches_df)
        
        # One-to-one vs many-to-one analysis
        self.metrics['cardinality'] = self._analyze_match_cardinality(matches_df)
        
        # Field-level agreement if comparison data available
        if comparison_data is not None:
            self.field_metrics = self._analyze_field_agreement(comparison_data)
            self.metrics['field_analysis'] = self.field_metrics
        
        # Ground truth metrics if available
        if ground_truth is not None:
            self.metrics['supervised'] = self._calculate_supervised_metrics(matches_df, ground_truth)
        else:
            # Unsupervised quality indicators
            self.metrics['unsupervised'] = self._calculate_unsupervised_metrics(matches_df)
        
        # Stability analysis
        self.metrics['stability'] = self._analyze_match_stability(matches_df)
        
        # Generate diagnostics
        self.diagnostics = self._generate_diagnostics(matches_df, comparison_data)
        
        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_quality_plots(matches_df, comparison_data)
        
        # Save reports if requested
        if self.config.save_reports:
            self._save_quality_report()
        
        logger.info("Match quality evaluation completed")
        return self.metrics
    
    def _calculate_basic_metrics(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic matching metrics."""
        return {
            'total_matches': len(matches_df),
            'unique_left_records': matches_df['idx1'].nunique(),
            'unique_right_records': matches_df['idx2'].nunique(),
            'avg_match_weight': float(matches_df['weight'].mean()),
            'std_match_weight': float(matches_df['weight'].std()),
            'avg_match_probability': float(matches_df['probability'].mean()),
            'std_match_probability': float(matches_df['probability'].std()),
            'min_weight': float(matches_df['weight'].min()),
            'max_weight': float(matches_df['weight'].max()),
            'weight_percentiles': {
                'p25': float(matches_df['weight'].quantile(0.25)),
                'p50': float(matches_df['weight'].quantile(0.50)),
                'p75': float(matches_df['weight'].quantile(0.75)),
                'p90': float(matches_df['weight'].quantile(0.90)),
                'p95': float(matches_df['weight'].quantile(0.95))
            }
        }
    
    def _analyze_match_distribution(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of match scores."""
        weights = matches_df['weight'].values
        probs = matches_df['probability'].values
        
        # Classification distribution
        classification_counts = matches_df['classification'].value_counts().to_dict()
        
        # Score concentration
        gini_weight = self._calculate_gini_coefficient(weights)
        gini_prob = self._calculate_gini_coefficient(probs)
        
        # Bimodality test
        bimodal_weight = self._test_bimodality(weights)
        bimodal_prob = self._test_bimodality(probs)
        
        return {
            'classification_distribution': classification_counts,
            'weight_gini': float(gini_weight),
            'probability_gini': float(gini_prob),
            'weight_bimodal': bimodal_weight,
            'probability_bimodal': bimodal_prob,
            'weight_entropy': float(entropy(np.histogram(weights, bins=20)[0] + 1)),
            'probability_entropy': float(entropy(np.histogram(probs, bins=20)[0] + 1))
        }
    
    def _analyze_match_cardinality(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze one-to-one vs many-to-one matching patterns."""
        # Count matches per left record
        left_counts = matches_df.groupby('idx1').size()
        right_counts = matches_df.groupby('idx2').size()
        
        # Identify different matching patterns
        one_to_one = ((left_counts == 1) & (right_counts == 1)).sum()
        one_to_many = (left_counts > 1).sum()
        many_to_one = (right_counts > 1).sum()
        
        # Calculate uniqueness score
        uniqueness_score = one_to_one / len(matches_df) if len(matches_df) > 0 else 0
        
        # Reuse analysis
        avg_reuse_left = left_counts.mean()
        avg_reuse_right = right_counts.mean()
        max_reuse_right = right_counts.max()
        
        return {
            'one_to_one_matches': int(one_to_one),
            'one_to_many_matches': int(one_to_many),
            'many_to_one_matches': int(many_to_one),
            'uniqueness_score': float(uniqueness_score),
            'avg_matches_per_left': float(avg_reuse_left),
            'avg_matches_per_right': float(avg_reuse_right),
            'max_right_record_reuse': int(max_reuse_right),
            'right_record_usage_distribution': right_counts.value_counts().head(10).to_dict()
        }
    
    def _analyze_field_agreement(self, comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze field-level agreement patterns."""
        field_metrics = {}
        
        # Identify similarity columns
        sim_cols = [col for col in comparison_data.columns if col.endswith('_similarity')]
        
        for col in sim_cols:
            field_name = col.replace('_similarity', '')
            similarities = comparison_data[col].values
            
            # Calculate agreement metrics
            high_agreement = (similarities >= 0.9).mean()
            medium_agreement = ((similarities >= 0.7) & (similarities < 0.9)).mean()
            low_agreement = (similarities < 0.7).mean()
            
            # Weight contribution (if weights available)
            weight_correlation = 0
            if 'weight' in comparison_data.columns:
                weight_correlation = comparison_data[[col, 'weight']].corr().iloc[0, 1]
            
            field_metrics[field_name] = {
                'mean_similarity': float(similarities.mean()),
                'std_similarity': float(similarities.std()),
                'high_agreement_rate': float(high_agreement),
                'medium_agreement_rate': float(medium_agreement),
                'low_agreement_rate': float(low_agreement),
                'null_rate': float(pd.isna(similarities).mean()),
                'weight_correlation': float(weight_correlation) if not pd.isna(weight_correlation) else 0
            }
        
        # Identify most discriminative fields
        if field_metrics:
            discriminative_fields = sorted(
                field_metrics.items(),
                key=lambda x: abs(x[1]['weight_correlation']),
                reverse=True
            )[:5]
            
            field_metrics['most_discriminative'] = [f[0] for f in discriminative_fields]
        
        return field_metrics
    
    def _calculate_supervised_metrics(self, matches_df: pd.DataFrame, 
                                     ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics when ground truth is available."""
        # Create match sets for comparison
        predicted_pairs = set(zip(matches_df['idx1'], matches_df['idx2']))
        true_pairs = set(zip(ground_truth['idx1'], ground_truth['idx2']))
        
        # Calculate precision, recall, F1
        tp = len(predicted_pairs & true_pairs)
        fp = len(predicted_pairs - true_pairs)
        fn = len(true_pairs - predicted_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate metrics at different thresholds
        threshold_metrics = {}
        for threshold in self.config.confidence_thresholds:
            thresh_matches = matches_df[matches_df['probability'] >= threshold]
            thresh_pairs = set(zip(thresh_matches['idx1'], thresh_matches['idx2']))
            
            tp_t = len(thresh_pairs & true_pairs)
            fp_t = len(thresh_pairs - true_pairs)
            fn_t = len(true_pairs - thresh_pairs)
            
            prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
            
            threshold_metrics[f'threshold_{threshold}'] = {
                'precision': float(prec_t),
                'recall': float(rec_t),
                'f1': float(f1_t),
                'n_matches': len(thresh_matches)
            }
        
        # Calculate ranking metrics
        ranking_metrics = self._calculate_ranking_metrics(matches_df, ground_truth)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'threshold_metrics': threshold_metrics,
            **ranking_metrics
        }
    
    def _calculate_unsupervised_metrics(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate unsupervised quality indicators."""
        weights = matches_df['weight'].values
        probs = matches_df['probability'].values
        
        # Separation quality - how well separated are matches from non-matches
        high_conf_ratio = (probs >= 0.8).mean()
        low_conf_ratio = (probs <= 0.2).mean()
        separation_score = high_conf_ratio + low_conf_ratio  # Want most to be clearly match/non-match
        
        # Consistency checks
        weight_prob_correlation = matches_df[['weight', 'probability']].corr().iloc[0, 1]
        
        # Match confidence distribution
        confidence_bands = {
            'very_high': float((probs >= 0.9).mean()),
            'high': float(((probs >= 0.7) & (probs < 0.9)).mean()),
            'medium': float(((probs >= 0.5) & (probs < 0.7)).mean()),
            'low': float(((probs >= 0.3) & (probs < 0.5)).mean()),
            'very_low': float((probs < 0.3).mean())
        }
        
        # Statistical tests for match quality
        # Test if high-probability matches have significantly different characteristics
        if len(matches_df) > 100:
            high_prob_weights = weights[probs >= 0.7]
            low_prob_weights = weights[probs < 0.3]
            if len(high_prob_weights) > 10 and len(low_prob_weights) > 10:
                ks_stat, ks_pvalue = ks_2samp(high_prob_weights, low_prob_weights)
            else:
                ks_stat, ks_pvalue = 0, 1
        else:
            ks_stat, ks_pvalue = 0, 1
        
        return {
            'separation_score': float(separation_score),
            'weight_probability_correlation': float(weight_prob_correlation),
            'confidence_distribution': confidence_bands,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'high_confidence_ratio': float(high_conf_ratio),
            'ambiguous_match_ratio': float(((probs >= 0.4) & (probs <= 0.6)).mean())
        }
    
    def _analyze_match_stability(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stability and robustness of matches."""
        # Group by left record to analyze match alternatives
        stability_metrics = {}
        
        for idx1, group in matches_df.groupby('idx1'):
            if len(group) > 1:
                # Sort by weight/probability
                sorted_group = group.sort_values('weight', ascending=False)
                weights = sorted_group['weight'].values
                
                # Calculate stability as ratio of best to second best
                if len(weights) >= 2:
                    stability_ratio = weights[0] / weights[1] if weights[1] != 0 else np.inf
                else:
                    stability_ratio = np.inf
                
                stability_metrics[idx1] = min(stability_ratio, 100)  # Cap at 100 for inf values
        
        if stability_metrics:
            avg_stability = np.mean(list(stability_metrics.values()))
            min_stability = np.min(list(stability_metrics.values()))
            unstable_matches = sum(1 for v in stability_metrics.values() if v < 1.5)
        else:
            avg_stability = min_stability = 0
            unstable_matches = 0
        
        return {
            'average_stability_ratio': float(avg_stability),
            'min_stability_ratio': float(min_stability),
            'unstable_matches': unstable_matches,
            'matches_with_alternatives': len(stability_metrics),
            'single_candidate_matches': len(matches_df['idx1'].unique()) - len(stability_metrics)
        }
    
    def _calculate_ranking_metrics(self, matches_df: pd.DataFrame, 
                                  ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ranking-based metrics (MRR, NDCG, Precision@k)."""
        true_pairs = set(zip(ground_truth['idx1'], ground_truth['idx2']))
        
        mrr_scores = []
        precision_at_k = {k: [] for k in self.config.top_k_values}
        
        # For each left record with true match
        for idx1 in ground_truth['idx1'].unique():
            # Get ranked predictions for this record
            predictions = matches_df[matches_df['idx1'] == idx1].sort_values('weight', ascending=False)
            
            if len(predictions) == 0:
                continue
            
            # Find rank of true match
            true_idx2 = ground_truth[ground_truth['idx1'] == idx1]['idx2'].iloc[0]
            
            rank = None
            for i, (_, row) in enumerate(predictions.iterrows(), 1):
                if row['idx2'] == true_idx2:
                    rank = i
                    break
            
            if rank:
                mrr_scores.append(1.0 / rank)
                
                # Precision@k
                for k in self.config.top_k_values:
                    top_k = predictions.head(k)
                    correct = sum(1 for _, r in top_k.iterrows() 
                                if (r['idx1'], r['idx2']) in true_pairs)
                    precision_at_k[k].append(correct / min(k, len(top_k)))
            else:
                mrr_scores.append(0)
                for k in self.config.top_k_values:
                    precision_at_k[k].append(0)
        
        # Calculate averages
        mrr = np.mean(mrr_scores) if mrr_scores else 0
        p_at_k = {f'precision_at_{k}': np.mean(scores) if scores else 0 
                 for k, scores in precision_at_k.items()}
        
        return {
            'mean_reciprocal_rank': float(mrr),
            **{k: float(v) for k, v in p_at_k.items()}
        }
    
    def _generate_diagnostics(self, matches_df: pd.DataFrame, 
                             comparison_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Generate detailed diagnostics for debugging and analysis."""
        diagnostics = {
            'data_summary': {
                'n_matches': len(matches_df),
                'n_unique_left': matches_df['idx1'].nunique(),
                'n_unique_right': matches_df['idx2'].nunique(),
                'has_comparison_data': comparison_data is not None
            }
        }
        
        # Identify potential issues
        issues = []
        
        # Check for excessive reuse
        right_reuse = matches_df.groupby('idx2').size()
        if right_reuse.max() > 100:
            issues.append({
                'type': 'excessive_template_reuse',
                'severity': 'warning',
                'details': f"Template {right_reuse.idxmax()} used {right_reuse.max()} times"
            })
        
        # Check for low confidence matches
        low_conf_ratio = (matches_df['probability'] < 0.5).mean()
        if low_conf_ratio > 0.2:
            issues.append({
                'type': 'many_low_confidence_matches',
                'severity': 'warning',
                'details': f"{low_conf_ratio:.1%} of matches have probability < 0.5"
            })
        
        # Check for weight-probability inconsistency
        if 'weight' in matches_df.columns and 'probability' in matches_df.columns:
            correlation = matches_df[['weight', 'probability']].corr().iloc[0, 1]
            if correlation < 0.5:
                issues.append({
                    'type': 'weight_probability_inconsistency',
                    'severity': 'warning',
                    'details': f"Low correlation ({correlation:.3f}) between weight and probability"
                })
        
        diagnostics['issues'] = issues
        diagnostics['n_issues'] = len(issues)
        
        # Sample problematic matches for inspection
        if len(issues) > 0 and len(matches_df) > 0:
            problematic_samples = matches_df.nsmallest(5, 'probability')[
                ['idx1', 'idx2', 'weight', 'probability', 'classification']
            ].to_dict('records')
            diagnostics['problematic_samples'] = problematic_samples
        
        return diagnostics
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(n) + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def _test_bimodality(self, values: np.ndarray) -> bool:
        """Test if distribution is bimodal using Hartigan's dip test approximation."""
        # Simple heuristic: check if there's a clear gap in the middle
        hist, bins = np.histogram(values, bins=30)
        
        # Find local minima in the middle third
        middle_start = len(hist) // 3
        middle_end = 2 * len(hist) // 3
        middle_hist = hist[middle_start:middle_end]
        
        if len(middle_hist) > 2:
            # Check for a valley
            min_val = np.min(middle_hist)
            edge_mean = (hist[:middle_start].mean() + hist[middle_end:].mean()) / 2
            
            # Bimodal if valley is significantly lower than edges
            return min_val < 0.5 * edge_mean
        
        return False
    
    def _generate_quality_plots(self, matches_df: pd.DataFrame, 
                               comparison_data: Optional[pd.DataFrame]):
        """Generate quality assessment plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Weight distribution
        axes[0, 0].hist(matches_df['weight'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Match Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Match Weight Distribution')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Probability distribution by classification
        for cls in matches_df['classification'].unique():
            cls_data = matches_df[matches_df['classification'] == cls]['probability']
            axes[0, 1].hist(cls_data, bins=20, alpha=0.5, label=cls)
        axes[0, 1].set_xlabel('Match Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Probability by Classification')
        axes[0, 1].legend()
        
        # 3. Weight vs Probability scatter
        axes[0, 2].scatter(matches_df['weight'], matches_df['probability'], 
                          alpha=0.3, s=10)
        axes[0, 2].set_xlabel('Match Weight')
        axes[0, 2].set_ylabel('Match Probability')
        axes[0, 2].set_title('Weight vs Probability')
        axes[0, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Template reuse distribution
        right_counts = matches_df.groupby('idx2').size()
        axes[1, 0].hist(right_counts, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Uses')
        axes[1, 0].set_ylabel('Number of Templates')
        axes[1, 0].set_title('Template Reuse Distribution')
        axes[1, 0].set_yscale('log')
        
        # 5. Field agreement heatmap (if comparison data available)
        if comparison_data is not None and self.field_metrics:
            field_names = list(self.field_metrics.keys())[:10]  # Top 10 fields
            agreement_matrix = []
            for field in field_names:
                if field != 'most_discriminative':
                    agreement_matrix.append([
                        self.field_metrics[field]['high_agreement_rate'],
                        self.field_metrics[field]['medium_agreement_rate'],
                        self.field_metrics[field]['low_agreement_rate']
                    ])
            
            if agreement_matrix:
                im = axes[1, 1].imshow(agreement_matrix, aspect='auto', cmap='YlOrRd')
                axes[1, 1].set_xticks([0, 1, 2])
                axes[1, 1].set_xticklabels(['High', 'Medium', 'Low'])
                axes[1, 1].set_yticks(range(len(field_names)))
                axes[1, 1].set_yticklabels(field_names)
                axes[1, 1].set_title('Field Agreement Rates')
                plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'No field data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Field Agreement Rates')
        
        # 6. Confidence distribution
        confidence_bands = self.metrics.get('unsupervised', {}).get('confidence_distribution', {})
        if confidence_bands:
            bands = list(confidence_bands.keys())
            values = list(confidence_bands.values())
            axes[1, 2].bar(bands, values)
            axes[1, 2].set_xlabel('Confidence Band')
            axes[1, 2].set_ylabel('Proportion')
            axes[1, 2].set_title('Match Confidence Distribution')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Match Quality Assessment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = f"{self.config.output_dir}/match_quality_assessment.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Quality plots saved to {output_path}")
        plt.close()
    
    def _save_quality_report(self):
        """Save quality assessment report."""
        report = {
            'metrics': self.metrics,
            'field_metrics': self.field_metrics,
            'diagnostics': self.diagnostics,
            'config': {
                'top_k_values': self.config.top_k_values,
                'confidence_thresholds': self.config.confidence_thresholds
            }
        }
        
        if self.config.report_format == 'json':
            output_path = f"{self.config.output_dir}/quality_report.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif self.config.report_format == 'html':
            output_path = f"{self.config.output_dir}/quality_report.html"
            html_content = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        elif self.config.report_format == 'markdown':
            output_path = f"{self.config.output_dir}/quality_report.md"
            md_content = self._generate_markdown_report(report)
            with open(output_path, 'w') as f:
                f.write(md_content)
        
        logger.info(f"Quality report saved to {output_path}")
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Match Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 1px solid #ccc; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #0066cc; }
                .warning { color: #ff6600; }
                .good { color: #00cc00; }
            </style>
        </head>
        <body>
            <h1>Match Quality Assessment Report</h1>
        """
        
        # Basic metrics
        if 'basic' in report['metrics']:
            html += "<h2>Basic Metrics</h2><table>"
            for key, value in report['metrics']['basic'].items():
                if isinstance(value, dict):
                    continue
                html += f"<tr><td>{key}</td><td class='metric'>{value:.4f if isinstance(value, float) else value}</td></tr>"
            html += "</table>"
        
        # Distribution metrics
        if 'distribution' in report['metrics']:
            html += "<h2>Distribution Analysis</h2><table>"
            for key, value in report['metrics']['distribution'].items():
                if isinstance(value, dict):
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                else:
                    html += f"<tr><td>{key}</td><td class='metric'>{value:.4f if isinstance(value, float) else value}</td></tr>"
            html += "</table>"
        
        # Diagnostics
        if report['diagnostics'].get('issues'):
            html += "<h2>Issues Detected</h2><ul>"
            for issue in report['diagnostics']['issues']:
                severity_class = 'warning' if issue['severity'] == 'warning' else 'error'
                html += f"<li class='{severity_class}'><b>{issue['type']}</b>: {issue['details']}</li>"
            html += "</ul>"
        
        html += "</body></html>"
        return html
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate Markdown report."""
        md = "# Match Quality Assessment Report\n\n"
        
        # Basic metrics
        if 'basic' in report['metrics']:
            md += "## Basic Metrics\n\n"
            for key, value in report['metrics']['basic'].items():
                if isinstance(value, dict):
                    continue
                md += f"- **{key}**: {value:.4f if isinstance(value, float) else value}\n"
            md += "\n"
        
        # Distribution metrics
        if 'distribution' in report['metrics']:
            md += "## Distribution Analysis\n\n"
            for key, value in report['metrics']['distribution'].items():
                if isinstance(value, dict):
                    md += f"- **{key}**: {value}\n"
                else:
                    md += f"- **{key}**: {value:.4f if isinstance(value, float) else value}\n"
            md += "\n"
        
        # Issues
        if report['diagnostics'].get('issues'):
            md += "## Issues Detected\n\n"
            for issue in report['diagnostics']['issues']:
                md += f"- **{issue['type']}** ({issue['severity']}): {issue['details']}\n"
            md += "\n"
        
        return md
    
    def explain_match(self, idx1: int, idx2: int, matches_df: pd.DataFrame,
                     comparison_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Provide detailed explanation for a specific match.
        
        Args:
            idx1: Left record index
            idx2: Right record index
            matches_df: DataFrame with matches
            comparison_data: Optional detailed comparison data
            
        Returns:
            Dictionary with match explanation
        """
        match = matches_df[(matches_df['idx1'] == idx1) & (matches_df['idx2'] == idx2)]
        
        if match.empty:
            return {'error': 'Match not found'}
        
        match = match.iloc[0]
        
        explanation = {
            'idx1': idx1,
            'idx2': idx2,
            'weight': float(match['weight']),
            'probability': float(match['probability']),
            'classification': match['classification']
        }
        
        # Add field-level details if available
        if comparison_data is not None:
            comparison = comparison_data[(comparison_data['idx1'] == idx1) & 
                                        (comparison_data['idx2'] == idx2)]
            if not comparison.empty:
                comparison = comparison.iloc[0]
                field_details = {}
                
                for col in comparison.index:
                    if col.endswith('_similarity'):
                        field_name = col.replace('_similarity', '')
                        field_details[field_name] = {
                            'similarity': float(comparison[col]) if not pd.isna(comparison[col]) else None,
                            'contribution': 'positive' if comparison[col] > 0.5 else 'negative'
                        }
                
                explanation['field_details'] = field_details
        
        # Add interpretation
        if match['probability'] > 0.9:
            explanation['assessment'] = 'Very strong match'
        elif match['probability'] > 0.7:
            explanation['assessment'] = 'Strong match'
        elif match['probability'] > 0.5:
            explanation['assessment'] = 'Probable match'
        else:
            explanation['assessment'] = 'Weak match'
        
        return explanation