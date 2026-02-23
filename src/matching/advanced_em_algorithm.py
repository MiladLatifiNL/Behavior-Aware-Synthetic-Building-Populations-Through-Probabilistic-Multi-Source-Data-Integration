"""
Advanced Expectation-Maximization algorithm with state-of-the-art enhancements.

This module implements an enhanced EM algorithm with:
- Adaptive learning rates using Adam optimizer
- Multiple random initializations with selection
- Momentum-based updates
- Aitken's acceleration for faster convergence
- Gaussian mixture models for continuous similarities
- Online learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
from scipy.stats import norm
from scipy.special import logsumexp
import warnings
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
import time

logger = logging.getLogger(__name__)


@dataclass
class EMConfig:
    """Configuration for advanced EM algorithm."""
    n_initializations: int = 10
    max_iterations: int = 200
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.01
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_aitken_acceleration: bool = True
    use_mixture_model: bool = True
    n_mixture_components: int = 3
    regularization_lambda: float = 0.01
    min_probability: float = 1e-6
    max_probability: float = 1 - 1e-6
    prior_match: float = 0.1
    adaptive_lr_patience: int = 10
    adaptive_lr_factor: float = 0.5
    early_stopping_patience: int = 20


class AdvancedEMAlgorithm:
    """
    State-of-the-art EM algorithm for Fellegi-Sunter parameter estimation.
    
    Enhancements over classical EM:
    1. Adam optimizer for adaptive learning rates
    2. Multiple random initializations with best selection
    3. Aitken's acceleration for faster convergence
    4. Gaussian mixture models for continuous similarities
    5. Online learning for streaming updates
    """
    
    def __init__(self, field_names: List[str], config: Optional[EMConfig] = None):
        """
        Initialize advanced EM algorithm.
        
        Args:
            field_names: List of comparison field names
            config: Configuration object for algorithm parameters
        """
        self.field_names = field_names
        self.config = config or EMConfig()
        
        # Parameters to learn
        self.m_probs = {}  # P(agreement|match)
        self.u_probs = {}  # P(agreement|non-match)
        
        # Adam optimizer states
        self.adam_m = {}  # First moment estimates
        self.adam_v = {}  # Second moment estimates
        self.adam_t = 0   # Time step
        
        # Momentum states
        self.momentum_m = {}
        self.momentum_u = {}
        
        # Convergence tracking
        self.log_likelihood_history = []
        self.parameter_history = []
        self.best_params = None
        self.best_likelihood = -np.inf
        
        # Mixture models for continuous similarities
        self.mixture_models = {}
        
        # Aitken acceleration states
        self.aitken_prev = None
        self.aitken_curr = None
        
    def fit(self, comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit EM algorithm with multiple initializations and advanced optimization.
        
        Args:
            comparison_data: DataFrame with similarity scores
            
        Returns:
            Dictionary with best parameters and diagnostics
        """
        logger.info(f"Starting advanced EM with {self.config.n_initializations} initializations")
        
        best_result = None
        all_results = []
        
        # Try multiple random initializations
        for init_idx in range(self.config.n_initializations):
            logger.debug(f"Running initialization {init_idx + 1}/{self.config.n_initializations}")
            
            # Reset states for new initialization
            self._reset_states()
            
            # Initialize parameters
            self._initialize_parameters(comparison_data, method='random' if init_idx > 0 else 'frequency')
            
            # Fit mixture models if enabled
            if self.config.use_mixture_model:
                self._fit_mixture_models(comparison_data)
            
            # Run EM iterations
            result = self._run_em_iterations(comparison_data)
            result['initialization'] = init_idx
            all_results.append(result)
            
            # Track best result
            if result['final_likelihood'] > self.best_likelihood:
                self.best_likelihood = result['final_likelihood']
                best_result = result
                self.best_params = {
                    'm_probs': self.m_probs.copy(),
                    'u_probs': self.u_probs.copy()
                }
        
        # Set parameters to best found
        self.m_probs = self.best_params['m_probs']
        self.u_probs = self.best_params['u_probs']
        
        logger.info(f"Best likelihood: {self.best_likelihood:.4f} from initialization {best_result['initialization']}")
        logger.info(f"Best m_probs: {self.m_probs}")
        logger.info(f"Best u_probs: {self.u_probs}")
        
        return {
            'best_result': best_result,
            'all_results': all_results,
            'm_probs': self.m_probs,
            'u_probs': self.u_probs,
            'convergence_iterations': best_result['iterations'],
            'final_likelihood': self.best_likelihood
        }
    
    def _reset_states(self):
        """Reset optimizer and tracking states."""
        self.adam_m = {field: 0 for field in self.field_names}
        self.adam_v = {field: 0 for field in self.field_names}
        self.adam_t = 0
        self.momentum_m = {field: 0 for field in self.field_names}
        self.momentum_u = {field: 0 for field in self.field_names}
        self.log_likelihood_history = []
        self.parameter_history = []
        self.aitken_prev = None
        self.aitken_curr = None
        
    def _initialize_parameters(self, comparison_data: pd.DataFrame, method: str = 'frequency'):
        """
        Initialize parameters using various strategies.
        
        Args:
            comparison_data: DataFrame with similarity scores
            method: 'frequency', 'random', or 'kmeans'
        """
        if method == 'frequency':
            # Use frequency-based initialization
            for field in self.field_names:
                if f'{field}_similarity' in comparison_data.columns:
                    similarities = comparison_data[f'{field}_similarity']
                    agreement_rate = (similarities >= 0.8).mean()
                    
                    # Estimate m and u from agreement rate
                    self.m_probs[field] = min(0.95, agreement_rate / self.config.prior_match)
                    self.u_probs[field] = max(0.05, 
                        (agreement_rate - self.config.prior_match * self.m_probs[field]) / 
                        (1 - self.config.prior_match))
                else:
                    self.m_probs[field] = 0.9
                    self.u_probs[field] = 0.1
                    
        elif method == 'random':
            # Random initialization with constraints
            for field in self.field_names:
                u = np.random.uniform(0.01, 0.3)
                m = np.random.uniform(max(u + 0.4, 0.5), 0.99)
                self.m_probs[field] = m
                self.u_probs[field] = u
                
        elif method == 'kmeans':
            # Use k-means on similarity vectors for initialization
            from sklearn.cluster import KMeans
            
            similarity_matrix = self._get_similarity_matrix(comparison_data)
            kmeans = KMeans(n_clusters=2, random_state=42).fit(similarity_matrix)
            
            # Assume cluster with higher mean similarity is matches
            cluster_means = [similarity_matrix[kmeans.labels_ == i].mean(axis=0) 
                           for i in range(2)]
            match_cluster = np.argmax([cm.mean() for cm in cluster_means])
            
            for i, field in enumerate(self.field_names):
                self.m_probs[field] = cluster_means[match_cluster][i]
                self.u_probs[field] = cluster_means[1 - match_cluster][i]
        
        # Ensure valid probability ranges
        self._enforce_probability_constraints()
        
    def _fit_mixture_models(self, comparison_data: pd.DataFrame):
        """
        Fit Gaussian mixture models for continuous similarity distributions.
        
        Args:
            comparison_data: DataFrame with similarity scores
        """
        if not self.config.use_mixture_model:
            return
            
        logger.debug("Fitting Gaussian mixture models for similarities")
        
        for field in self.field_names:
            if f'{field}_similarity' in comparison_data.columns:
                similarities = comparison_data[f'{field}_similarity'].values.reshape(-1, 1)
                
                # Fit mixture model
                gmm = GaussianMixture(
                    n_components=self.config.n_mixture_components,
                    covariance_type='full',
                    random_state=42
                )
                gmm.fit(similarities)
                self.mixture_models[field] = gmm
    
    def _run_em_iterations(self, comparison_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run EM iterations with advanced optimization techniques.
        
        Args:
            comparison_data: DataFrame with similarity scores
            
        Returns:
            Dictionary with convergence information
        """
        converged = False
        iteration = 0
        best_iter_likelihood = -np.inf
        patience_counter = 0
        learning_rate = self.config.learning_rate
        
        start_time = time.time()
        
        while iteration < self.config.max_iterations and not converged:
            iteration += 1
            
            # E-step: Calculate match probabilities
            match_probs = self._expectation_step(comparison_data)
            
            # M-step: Update parameters with advanced optimization
            old_params = self._get_current_params()
            self._maximization_step(comparison_data, match_probs, learning_rate)
            
            # Apply Aitken acceleration if enabled
            if self.config.use_aitken_acceleration and iteration > 2:
                self._apply_aitken_acceleration()
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(comparison_data, match_probs)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check for improvement
            if log_likelihood > best_iter_likelihood:
                best_iter_likelihood = log_likelihood
                patience_counter = 0
            else:
                patience_counter += 1
                
                # Adaptive learning rate
                if patience_counter >= self.config.adaptive_lr_patience:
                    learning_rate *= self.config.adaptive_lr_factor
                    logger.debug(f"Reducing learning rate to {learning_rate:.6f}")
                    patience_counter = 0
            
            # Check convergence
            if iteration > 1:
                likelihood_change = abs(log_likelihood - self.log_likelihood_history[-2])
                param_change = self._calculate_param_change(old_params)
                
                if likelihood_change < self.config.convergence_threshold:
                    converged = True
                    logger.info(f"Converged after {iteration} iterations")
                elif patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping after {iteration} iterations")
                    break
            
            # Store parameter history
            self.parameter_history.append(self._get_current_params())
            
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: likelihood={log_likelihood:.4f}")
        
        elapsed_time = time.time() - start_time
        
        return {
            'iterations': iteration,
            'converged': converged,
            'final_likelihood': log_likelihood,
            'time_elapsed': elapsed_time,
            'learning_rate_final': learning_rate
        }
    
    def _expectation_step(self, comparison_data: pd.DataFrame) -> np.ndarray:
        """
        Enhanced E-step with mixture model support.
        
        Args:
            comparison_data: DataFrame with similarity scores
            
        Returns:
            Array of match probabilities
        """
        n_pairs = len(comparison_data)
        log_odds_match = np.log(self.config.prior_match / (1 - self.config.prior_match))
        
        for field in self.field_names:
            if f'{field}_similarity' not in comparison_data.columns:
                continue
                
            similarities = comparison_data[f'{field}_similarity'].values
            m = self.m_probs[field]
            u = self.u_probs[field]
            
            if self.config.use_mixture_model and field in self.mixture_models:
                # Use mixture model for more nuanced probability calculation
                gmm = self.mixture_models[field]
                log_probs = gmm.score_samples(similarities.reshape(-1, 1))
                
                # Weight by match/non-match probabilities
                log_odds_match += log_probs * (m - u)
            else:
                # Standard binary agreement model
                agreements = similarities >= 0.8
                log_odds_match += agreements * np.log(m / u) + (1 - agreements) * np.log((1 - m) / (1 - u))
        
        # Convert log-odds to probabilities
        match_probs = 1 / (1 + np.exp(-log_odds_match))
        return match_probs
    
    def _maximization_step(self, comparison_data: pd.DataFrame, match_probs: np.ndarray, 
                          learning_rate: float):
        """
        Enhanced M-step with Adam optimizer.
        
        Args:
            comparison_data: DataFrame with similarity scores
            match_probs: Array of match probabilities
            learning_rate: Current learning rate
        """
        self.adam_t += 1
        
        for field in self.field_names:
            if f'{field}_similarity' not in comparison_data.columns:
                continue
                
            similarities = comparison_data[f'{field}_similarity'].values
            
            # Calculate gradients
            if self.config.use_mixture_model and field in self.mixture_models:
                # Gradient for mixture model parameters
                grad_m, grad_u = self._calculate_mixture_gradients(
                    similarities, match_probs, field
                )
            else:
                # Standard gradient calculation
                agreements = similarities >= 0.8
                grad_m = np.sum(match_probs * agreements) / np.sum(match_probs) - self.m_probs[field]
                grad_u = np.sum((1 - match_probs) * agreements) / np.sum(1 - match_probs) - self.u_probs[field]
            
            # Apply Adam optimizer
            self.adam_m[field] = self.config.adam_beta1 * self.adam_m[field] + (1 - self.config.adam_beta1) * grad_m
            self.adam_v[field] = self.config.adam_beta2 * self.adam_v[field] + (1 - self.config.adam_beta2) * grad_m**2
            
            # Bias correction
            m_hat = self.adam_m[field] / (1 - self.config.adam_beta1**self.adam_t)
            v_hat = self.adam_v[field] / (1 - self.config.adam_beta2**self.adam_t)
            
            # Update parameters
            self.m_probs[field] += learning_rate * m_hat / (np.sqrt(v_hat) + self.config.adam_epsilon)
            
            # Similar for u probabilities
            self.adam_m[field + '_u'] = self.config.adam_beta1 * self.adam_m.get(field + '_u', 0) + (1 - self.config.adam_beta1) * grad_u
            self.adam_v[field + '_u'] = self.config.adam_beta2 * self.adam_v.get(field + '_u', 0) + (1 - self.config.adam_beta2) * grad_u**2
            
            m_hat_u = self.adam_m[field + '_u'] / (1 - self.config.adam_beta1**self.adam_t)
            v_hat_u = self.adam_v[field + '_u'] / (1 - self.config.adam_beta2**self.adam_t)
            
            self.u_probs[field] += learning_rate * m_hat_u / (np.sqrt(v_hat_u) + self.config.adam_epsilon)
            
            # Apply regularization
            self.m_probs[field] -= self.config.regularization_lambda * (self.m_probs[field] - 0.9)
            self.u_probs[field] -= self.config.regularization_lambda * (self.u_probs[field] - 0.1)
        
        # Ensure probability constraints
        self._enforce_probability_constraints()
    
    def _apply_aitken_acceleration(self):
        """
        Apply Aitken's acceleration method for faster convergence.
        
        This method extrapolates the parameter sequence to accelerate convergence.
        """
        if len(self.parameter_history) < 3:
            return
            
        # Get last three parameter sets
        params_n2 = self.parameter_history[-3]
        params_n1 = self.parameter_history[-2]
        params_n = self._get_current_params()
        
        # Calculate Aitken acceleration for each parameter
        for field in self.field_names:
            # For m probabilities
            x_n2 = params_n2['m_probs'][field]
            x_n1 = params_n1['m_probs'][field]
            x_n = self.m_probs[field]
            
            denominator = (x_n - x_n1) - (x_n1 - x_n2)
            if abs(denominator) > 1e-10:
                x_accelerated = x_n - ((x_n - x_n1)**2) / denominator
                # Only apply if acceleration keeps probability valid
                if self.config.min_probability < x_accelerated < self.config.max_probability:
                    self.m_probs[field] = x_accelerated
            
            # For u probabilities
            x_n2 = params_n2['u_probs'][field]
            x_n1 = params_n1['u_probs'][field]
            x_n = self.u_probs[field]
            
            denominator = (x_n - x_n1) - (x_n1 - x_n2)
            if abs(denominator) > 1e-10:
                x_accelerated = x_n - ((x_n - x_n1)**2) / denominator
                if self.config.min_probability < x_accelerated < self.config.max_probability:
                    self.u_probs[field] = x_accelerated
    
    def _calculate_mixture_gradients(self, similarities: np.ndarray, match_probs: np.ndarray, 
                                    field: str) -> Tuple[float, float]:
        """
        Calculate gradients when using mixture models.
        
        Args:
            similarities: Array of similarity scores
            match_probs: Array of match probabilities
            field: Field name
            
        Returns:
            Tuple of (gradient_m, gradient_u)
        """
        gmm = self.mixture_models[field]
        responsibilities = gmm.predict_proba(similarities.reshape(-1, 1))
        
        # Weight responsibilities by match probabilities
        weighted_resp_match = responsibilities * match_probs.reshape(-1, 1)
        weighted_resp_nonmatch = responsibilities * (1 - match_probs).reshape(-1, 1)
        
        # Calculate expected values under each component
        component_means = gmm.means_.flatten()
        
        # Gradient is weighted average of component contributions
        grad_m = np.sum(weighted_resp_match @ component_means) / np.sum(match_probs) - self.m_probs[field]
        grad_u = np.sum(weighted_resp_nonmatch @ component_means) / np.sum(1 - match_probs) - self.u_probs[field]
        
        return grad_m, grad_u
    
    def _calculate_log_likelihood(self, comparison_data: pd.DataFrame, match_probs: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the current parameters.
        
        Args:
            comparison_data: DataFrame with similarity scores
            match_probs: Array of match probabilities
            
        Returns:
            Log-likelihood value
        """
        log_likelihood = 0
        
        # Prior contribution
        log_likelihood += np.sum(match_probs * np.log(self.config.prior_match) + 
                                (1 - match_probs) * np.log(1 - self.config.prior_match))
        
        # Field contributions
        for field in self.field_names:
            if f'{field}_similarity' not in comparison_data.columns:
                continue
                
            similarities = comparison_data[f'{field}_similarity'].values
            m = self.m_probs[field]
            u = self.u_probs[field]
            
            if self.config.use_mixture_model and field in self.mixture_models:
                # Mixture model likelihood
                gmm = self.mixture_models[field]
                log_probs = gmm.score_samples(similarities.reshape(-1, 1))
                log_likelihood += np.sum(match_probs * log_probs * m + (1 - match_probs) * log_probs * u)
            else:
                # Binary agreement likelihood
                agreements = similarities >= 0.8
                log_likelihood += np.sum(
                    match_probs * (agreements * np.log(m) + (1 - agreements) * np.log(1 - m)) +
                    (1 - match_probs) * (agreements * np.log(u) + (1 - agreements) * np.log(1 - u))
                )
        
        return log_likelihood
    
    def _enforce_probability_constraints(self):
        """Ensure all probabilities are within valid ranges and m > u."""
        for field in self.field_names:
            # Clip to valid range
            self.m_probs[field] = np.clip(self.m_probs[field], 
                                         self.config.min_probability, 
                                         self.config.max_probability)
            self.u_probs[field] = np.clip(self.u_probs[field], 
                                         self.config.min_probability, 
                                         self.config.max_probability)
            
            # Ensure m > u with minimum separation
            if self.m_probs[field] <= self.u_probs[field]:
                mid = (self.m_probs[field] + self.u_probs[field]) / 2
                self.m_probs[field] = min(self.config.max_probability, mid + 0.1)
                self.u_probs[field] = max(self.config.min_probability, mid - 0.1)
    
    def _get_current_params(self) -> Dict[str, Dict[str, float]]:
        """Get current parameter values."""
        return {
            'm_probs': self.m_probs.copy(),
            'u_probs': self.u_probs.copy()
        }
    
    def _calculate_param_change(self, old_params: Dict[str, Dict[str, float]]) -> float:
        """Calculate the magnitude of parameter change."""
        change = 0
        for field in self.field_names:
            change += abs(self.m_probs[field] - old_params['m_probs'][field])
            change += abs(self.u_probs[field] - old_params['u_probs'][field])
        return change / (2 * len(self.field_names))
    
    def _get_similarity_matrix(self, comparison_data: pd.DataFrame) -> np.ndarray:
        """Extract similarity matrix from comparison data."""
        similarity_cols = [f'{field}_similarity' for field in self.field_names 
                         if f'{field}_similarity' in comparison_data.columns]
        return comparison_data[similarity_cols].values
    
    def online_update(self, new_comparison_data: pd.DataFrame, learning_rate: float = 0.001):
        """
        Perform online update with new comparison data.
        
        Args:
            new_comparison_data: New comparison pairs
            learning_rate: Learning rate for online update
        """
        # Single E-M iteration on new data
        match_probs = self._expectation_step(new_comparison_data)
        self._maximization_step(new_comparison_data, match_probs, learning_rate)
        
        logger.debug(f"Online update completed with {len(new_comparison_data)} new pairs")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics about the EM algorithm performance.
        
        Returns:
            Dictionary with diagnostic information
        """
        return {
            'convergence_history': self.log_likelihood_history,
            'parameter_evolution': self.parameter_history,
            'final_m_probs': self.m_probs,
            'final_u_probs': self.u_probs,
            'best_likelihood': self.best_likelihood,
            'n_iterations': len(self.log_likelihood_history),
            'parameter_separation': {
                field: self.m_probs[field] - self.u_probs[field] 
                for field in self.field_names
            },
            'mixture_models_fitted': list(self.mixture_models.keys()) if self.config.use_mixture_model else []
        }