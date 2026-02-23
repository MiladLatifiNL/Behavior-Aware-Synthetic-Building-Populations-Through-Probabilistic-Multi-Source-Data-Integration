"""
Probability calibration module for match score calibration.

This module implements various calibration methods to ensure match probabilities
are well-calibrated and meaningful, including:
- Platt scaling (sigmoid calibration)
- Isotonic regression
- Beta calibration
- Temperature scaling
- Ensemble calibration methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, KFold
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""
    method: str = 'platt'  # 'platt', 'isotonic', 'beta', 'temperature', 'ensemble'
    cv_folds: int = 5
    min_samples_per_bin: int = 10
    n_bins: int = 10
    confidence_level: float = 0.95
    plot_diagnostics: bool = True
    ensemble_methods: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['platt', 'isotonic', 'beta']


class PlattScaling:
    """
    Platt scaling for probability calibration.
    
    Fits a sigmoid function to calibrate probabilities:
    P_calibrated = 1 / (1 + exp(A * score + B))
    """
    
    def __init__(self):
        self.model = None
        self.A = None
        self.B = None
        
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters.
        
        Args:
            scores: Raw match scores
            labels: True labels (1 for match, 0 for non-match)
        """
        # Reshape for sklearn
        scores = scores.reshape(-1, 1)
        
        # Fit logistic regression
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.model.fit(scores, labels)
        
        # Extract parameters
        self.A = self.model.coef_[0, 0]
        self.B = self.model.intercept_[0]
        
        logger.info(f"Platt scaling fitted: A={self.A:.4f}, B={self.B:.4f}")
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to scores.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = scores.reshape(-1, 1)
        return self.model.predict_proba(scores)[:, 1]
    
    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(scores, labels)
        return self.transform(scores)


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.
    
    Fits a monotonic function to calibrate probabilities.
    """
    
    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Initialize isotonic calibration.
        
        Args:
            out_of_bounds: How to handle out-of-bounds predictions ('clip', 'nan', or 'raise')
        """
        self.model = None
        self.out_of_bounds = out_of_bounds
        
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit isotonic regression.
        
        Args:
            scores: Raw match scores
            labels: True labels
        """
        self.model = IsotonicRegression(out_of_bounds=self.out_of_bounds)
        self.model.fit(scores, labels)
        
        logger.info("Isotonic regression calibration fitted")
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.transform(scores)


class BetaCalibration:
    """
    Beta calibration for probability calibration.
    
    Maps scores through a beta distribution CDF for calibration.
    """
    
    def __init__(self):
        self.alpha = None
        self.beta_param = None
        self.min_score = None
        self.max_score = None
        
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit beta calibration parameters.
        
        Args:
            scores: Raw match scores
            labels: True labels
        """
        # Normalize scores to [0, 1]
        self.min_score = scores.min()
        self.max_score = scores.max()
        normalized_scores = (scores - self.min_score) / (self.max_score - self.min_score + 1e-10)
        
        # Fit beta distribution parameters using MLE
        def negative_log_likelihood(params):
            alpha, beta_param = params
            if alpha <= 0 or beta_param <= 0:
                return np.inf
            
            # Calculate probabilities using beta CDF
            probs = beta.cdf(normalized_scores, alpha, beta_param)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            
            # Negative log-likelihood
            nll = -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return nll
        
        # Optimize parameters
        result = minimize(negative_log_likelihood, x0=[1.0, 1.0], 
                        bounds=[(0.01, 100), (0.01, 100)], method='L-BFGS-B')
        
        self.alpha = result.x[0]
        self.beta_param = result.x[1]
        
        logger.info(f"Beta calibration fitted: alpha={self.alpha:.4f}, beta={self.beta_param:.4f}")
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if self.alpha is None or self.beta_param is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Normalize scores
        normalized_scores = (scores - self.min_score) / (self.max_score - self.min_score + 1e-10)
        normalized_scores = np.clip(normalized_scores, 0, 1)
        
        # Apply beta CDF
        return beta.cdf(normalized_scores, self.alpha, self.beta_param)


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    
    Applies a temperature parameter to logits:
    P_calibrated = softmax(logits / T)
    """
    
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit temperature parameter.
        
        Args:
            scores: Raw match scores (as probabilities)
            labels: True labels
        """
        # Convert probabilities to logits
        scores_clipped = np.clip(scores, 1e-10, 1 - 1e-10)
        logits = logit(scores_clipped)
        
        # Optimize temperature
        def nll_loss(T):
            if T <= 0:
                return np.inf
            
            # Apply temperature scaling
            scaled_logits = logits / T
            probs = expit(scaled_logits)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            
            # Negative log-likelihood
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        result = minimize(nll_loss, x0=1.0, bounds=[(0.01, 10)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        logger.info(f"Temperature scaling fitted: T={self.temperature:.4f}")
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling.
        
        Args:
            scores: Raw scores (as probabilities)
            
        Returns:
            Calibrated probabilities
        """
        # Convert to logits
        scores_clipped = np.clip(scores, 1e-10, 1 - 1e-10)
        logits = logit(scores_clipped)
        
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Convert back to probabilities
        return expit(scaled_logits)


class ProbabilityCalibrator:
    """
    Main class for probability calibration with multiple methods and diagnostics.
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize probability calibrator.
        
        Args:
            config: Configuration for calibration
        """
        self.config = config or CalibrationConfig()
        self.calibrators = {}
        self.calibration_metrics = {}
        self.is_fitted = False
        
    def fit(self, scores: np.ndarray, labels: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """
        Fit calibration model(s) using cross-validation.
        
        Args:
            scores: Raw match scores
            labels: True labels (1 for match, 0 for non-match)
            sample_weights: Optional sample weights
        """
        logger.info(f"Fitting probability calibration using {self.config.method} method")
        
        if self.config.method == 'ensemble':
            # Fit multiple calibrators
            for method in self.config.ensemble_methods:
                self._fit_single_calibrator(method, scores, labels, sample_weights)
        else:
            # Fit single calibrator
            self._fit_single_calibrator(self.config.method, scores, labels, sample_weights)
        
        # Calculate calibration metrics
        self._calculate_calibration_metrics(scores, labels)
        
        # Plot diagnostics if requested
        if self.config.plot_diagnostics:
            self._plot_calibration_diagnostics(scores, labels)
        
        self.is_fitted = True
        logger.info("Calibration fitting completed")
    
    def transform(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply calibration to scores.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Tuple of (calibrated_probabilities, confidence_intervals)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.config.method == 'ensemble':
            # Ensemble calibration
            calibrated_probs = self._ensemble_transform(scores)
        else:
            # Single method calibration
            calibrator = self.calibrators[self.config.method]
            calibrated_probs = calibrator.transform(scores)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(calibrated_probs)
        
        return calibrated_probs, confidence_intervals
    
    def _fit_single_calibrator(self, method: str, scores: np.ndarray, labels: np.ndarray,
                              sample_weights: Optional[np.ndarray] = None):
        """Fit a single calibration method."""
        if method == 'platt':
            calibrator = PlattScaling()
        elif method == 'isotonic':
            calibrator = IsotonicCalibration()
        elif method == 'beta':
            calibrator = BetaCalibration()
        elif method == 'temperature':
            calibrator = TemperatureScaling()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Use cross-validation to avoid overfitting
        if self.config.cv_folds > 1:
            calibrated_scores = self._cross_val_calibrate(calibrator, scores, labels)
        else:
            calibrator.fit(scores, labels)
            calibrated_scores = calibrator.transform(scores)
        
        self.calibrators[method] = calibrator
        
    def _cross_val_calibrate(self, calibrator, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Perform cross-validated calibration."""
        kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        calibrated_scores = np.zeros_like(scores)
        
        for train_idx, val_idx in kf.split(scores):
            # Fit on training fold
            calibrator_fold = calibrator.__class__()
            calibrator_fold.fit(scores[train_idx], labels[train_idx])
            
            # Predict on validation fold
            calibrated_scores[val_idx] = calibrator_fold.transform(scores[val_idx])
        
        # Fit final model on all data
        calibrator.fit(scores, labels)
        
        return calibrated_scores
    
    def _ensemble_transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply ensemble calibration."""
        calibrated_results = []
        
        for method in self.config.ensemble_methods:
            if method in self.calibrators:
                calibrated = self.calibrators[method].transform(scores)
                calibrated_results.append(calibrated)
        
        # Average ensemble predictions
        return np.mean(calibrated_results, axis=0)
    
    def _calculate_calibration_metrics(self, scores: np.ndarray, labels: np.ndarray):
        """Calculate calibration quality metrics."""
        # Apply calibration
        if self.config.method == 'ensemble':
            calibrated_probs = self._ensemble_transform(scores)
        else:
            calibrated_probs = self.calibrators[self.config.method].transform(scores)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_probs, labels)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(calibrated_probs, labels)
        
        # Brier Score
        brier_score = np.mean((calibrated_probs - labels) ** 2)
        
        # Log loss
        eps = 1e-10
        log_loss = -np.mean(labels * np.log(calibrated_probs + eps) + 
                           (1 - labels) * np.log(1 - calibrated_probs + eps))
        
        self.calibration_metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'log_loss': log_loss
        }
        
        logger.info(f"Calibration metrics - ECE: {ece:.4f}, MCE: {mce:.4f}, "
                   f"Brier: {brier_score:.4f}, Log Loss: {log_loss:.4f}")
    
    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        ece = 0
        
        for i in range(self.config.n_bins):
            bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > self.config.min_samples_per_bin:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probs[bin_mask])
                bin_weight = np.sum(bin_mask) / len(probs)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _calculate_mce(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        mce = 0
        
        for i in range(self.config.n_bins):
            bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > self.config.min_samples_per_bin:
                bin_accuracy = np.mean(labels[bin_mask])
                bin_confidence = np.mean(probs[bin_mask])
                
                mce = max(mce, abs(bin_accuracy - bin_confidence))
        
        return mce
    
    def _calculate_confidence_intervals(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculate confidence intervals for calibrated probabilities.
        
        Args:
            probs: Calibrated probabilities
            
        Returns:
            Array of confidence interval widths
        """
        # Use Wilson score interval for binomial proportions
        z = 1.96  # 95% confidence
        n = 100  # Assumed effective sample size
        
        # Wilson score interval
        center = (probs + z**2 / (2*n)) / (1 + z**2/n)
        half_width = z * np.sqrt((probs * (1 - probs) + z**2/(4*n)) / n) / (1 + z**2/n)
        
        lower = np.maximum(center - half_width, 0)
        upper = np.minimum(center + half_width, 1)
        
        return np.column_stack([lower, upper])
    
    def _plot_calibration_diagnostics(self, scores: np.ndarray, labels: np.ndarray):
        """Plot calibration diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Apply calibration
        if self.config.method == 'ensemble':
            calibrated_probs = self._ensemble_transform(scores)
        else:
            calibrated_probs = self.calibrators[self.config.method].transform(scores)
        
        # 1. Reliability diagram
        self._plot_reliability_diagram(axes[0, 0], scores, calibrated_probs, labels)
        
        # 2. Calibration curve
        self._plot_calibration_curve(axes[0, 1], scores, calibrated_probs)
        
        # 3. Score distribution
        self._plot_score_distribution(axes[1, 0], scores, calibrated_probs, labels)
        
        # 4. ROC curve comparison
        self._plot_roc_comparison(axes[1, 1], scores, calibrated_probs, labels)
        
        plt.suptitle(f'Calibration Diagnostics - {self.config.method.capitalize()} Method')
        plt.tight_layout()
        
        # Save plot
        output_path = 'data/validation/calibration_diagnostics.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Calibration diagnostics saved to {output_path}")
        plt.close()
    
    def _plot_reliability_diagram(self, ax, raw_scores, calibrated_probs, labels):
        """Plot reliability diagram."""
        bin_boundaries = np.linspace(0, 1, self.config.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        # Calculate for raw scores
        raw_accuracies = []
        raw_counts = []
        for i in range(self.config.n_bins):
            bin_mask = (raw_scores >= bin_boundaries[i]) & (raw_scores < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                raw_accuracies.append(np.mean(labels[bin_mask]))
                raw_counts.append(np.sum(bin_mask))
            else:
                raw_accuracies.append(np.nan)
                raw_counts.append(0)
        
        # Calculate for calibrated scores
        cal_accuracies = []
        for i in range(self.config.n_bins):
            bin_mask = (calibrated_probs >= bin_boundaries[i]) & (calibrated_probs < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                cal_accuracies.append(np.mean(labels[bin_mask]))
            else:
                cal_accuracies.append(np.nan)
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(bin_centers, raw_accuracies, 'o-', label='Raw scores', alpha=0.7)
        ax.plot(bin_centers, cal_accuracies, 's-', label='Calibrated', alpha=0.7)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_calibration_curve(self, ax, raw_scores, calibrated_probs):
        """Plot calibration transformation curve."""
        sorted_idx = np.argsort(raw_scores)
        ax.plot(raw_scores[sorted_idx], calibrated_probs[sorted_idx], 'b-', alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('Raw Score')
        ax.set_ylabel('Calibrated Probability')
        ax.set_title('Calibration Transformation')
        ax.grid(True, alpha=0.3)
    
    def _plot_score_distribution(self, ax, raw_scores, calibrated_probs, labels):
        """Plot score distributions."""
        ax.hist(raw_scores[labels == 0], bins=30, alpha=0.5, label='Non-matches (raw)', density=True)
        ax.hist(raw_scores[labels == 1], bins=30, alpha=0.5, label='Matches (raw)', density=True)
        ax.hist(calibrated_probs[labels == 0], bins=30, alpha=0.5, label='Non-matches (cal)', 
               density=True, histtype='step', linewidth=2)
        ax.hist(calibrated_probs[labels == 1], bins=30, alpha=0.5, label='Matches (cal)', 
               density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel('Score / Probability')
        ax.set_ylabel('Density')
        ax.set_title('Score Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_roc_comparison(self, ax, raw_scores, calibrated_probs, labels):
        """Plot ROC curves for raw and calibrated scores."""
        from sklearn.metrics import roc_curve, auc
        
        # Raw scores ROC
        fpr_raw, tpr_raw, _ = roc_curve(labels, raw_scores)
        auc_raw = auc(fpr_raw, tpr_raw)
        
        # Calibrated scores ROC
        fpr_cal, tpr_cal, _ = roc_curve(labels, calibrated_probs)
        auc_cal = auc(fpr_cal, tpr_cal)
        
        ax.plot(fpr_raw, tpr_raw, label=f'Raw (AUC = {auc_raw:.3f})')
        ax.plot(fpr_cal, tpr_cal, label=f'Calibrated (AUC = {auc_cal:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive calibration report.
        
        Returns:
            Dictionary with calibration metrics and diagnostics
        """
        return {
            'method': self.config.method,
            'is_fitted': self.is_fitted,
            'metrics': self.calibration_metrics,
            'calibrators': list(self.calibrators.keys()),
            'config': {
                'cv_folds': self.config.cv_folds,
                'n_bins': self.config.n_bins,
                'confidence_level': self.config.confidence_level
            }
        }
    
    def explain_calibration(self, raw_score: float) -> Dict[str, Any]:
        """
        Explain calibration for a specific score.
        
        Args:
            raw_score: Raw match score
            
        Returns:
            Dictionary with calibration explanation
        """
        if not self.is_fitted:
            return {'error': 'Calibrator not fitted'}
        
        calibrated_prob, ci = self.transform(np.array([raw_score]))
        
        explanation = {
            'raw_score': raw_score,
            'calibrated_probability': calibrated_prob[0],
            'confidence_interval': ci[0].tolist(),
            'method': self.config.method
        }
        
        # Add method-specific details
        if self.config.method == 'platt' and 'platt' in self.calibrators:
            platt = self.calibrators['platt']
            explanation['platt_parameters'] = {
                'A': platt.A,
                'B': platt.B
            }
        elif self.config.method == 'temperature' and 'temperature' in self.calibrators:
            temp = self.calibrators['temperature']
            explanation['temperature'] = temp.temperature
        
        # Interpretation
        if calibrated_prob[0] > 0.9:
            explanation['interpretation'] = 'Very high confidence match'
        elif calibrated_prob[0] > 0.7:
            explanation['interpretation'] = 'High confidence match'
        elif calibrated_prob[0] > 0.5:
            explanation['interpretation'] = 'Probable match'
        elif calibrated_prob[0] > 0.3:
            explanation['interpretation'] = 'Possible match'
        else:
            explanation['interpretation'] = 'Unlikely match'
        
        return explanation