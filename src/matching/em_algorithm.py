"""
Expectation-Maximization algorithm for Fellegi-Sunter parameter estimation.

This module implements the EM algorithm to learn m and u probabilities
without labeled training data, following the approach described in
Winkler (1988) and subsequent improvements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

from ..utils.constants import (
    EM_DEFAULT_PRIOR_MATCH,
    EM_MIN_PROBABILITY,
    EM_MAX_PROBABILITY,
    EM_MIN_SEPARATION,
    EM_MAX_ITERATIONS,
    EM_CONVERGENCE_THRESHOLD,
    EM_REGULARIZATION_LAMBDA,
    DEFAULT_M_PROB_NAME,
    DEFAULT_M_PROB_GENERAL,
    DEFAULT_U_PROB
)

logger = logging.getLogger(__name__)


class EMAlgorithm:
    """
    EM algorithm for estimating Fellegi-Sunter model parameters.
    
    This implementation learns m-probabilities (P(agreement|match)) and
    u-probabilities (P(agreement|non-match)) from unlabeled comparison data.
    """
    
    def __init__(self, field_names: List[str], prior_match: float = EM_DEFAULT_PRIOR_MATCH):
        """
        Initialize EM algorithm.
        
        Args:
            field_names: List of comparison field names
            prior_match: Prior probability that a random pair is a match
        """
        self.field_names = field_names
        self.prior_match = prior_match
        
        # Initialize parameters
        self.m_probs = {}  # P(agreement|match) for each field
        self.u_probs = {}  # P(agreement|non-match) for each field
        
        # Convergence tracking
        self.converged = False
        self.iteration = 0
        self.log_likelihood_history = []
        self.parameter_history = []
    
    def initialize_parameters(self, comparison_data: pd.DataFrame,
                            use_frequency_estimates: bool = True):
        """
        Initialize m and u probabilities with reasonable starting values.
        
        Args:
            comparison_data: DataFrame with agreement patterns
            use_frequency_estimates: Whether to use data frequencies for initialization
        """
        logger.info("Initializing EM parameters")
        
        for field in self.field_names:
            if use_frequency_estimates:
                # Use data to estimate initial values
                if f'{field}_similarity' in comparison_data.columns:
                    # Calculate agreement rate
                    agreements = comparison_data[f'{field}_similarity'] >= 0.88
                    agreement_rate = agreements.mean()
                    
                    # Set m_prob high (most matches should agree)
                    self.m_probs[field] = DEFAULT_M_PROB_GENERAL
                    
                    # Estimate u_prob from overall agreement rate
                    # u ≈ (agreement_rate - p*m) / (1-p)
                    estimated_u = max(
                        (agreement_rate - self.prior_match * DEFAULT_M_PROB_GENERAL) / (1 - self.prior_match),
                        EM_MIN_PROBABILITY
                    )
                    self.u_probs[field] = min(estimated_u, 0.5)
                else:
                    # Default values if field not found
                    self.m_probs[field] = DEFAULT_M_PROB_GENERAL
                    self.u_probs[field] = DEFAULT_U_PROB
            else:
                # Use fixed default values
                self.m_probs[field] = DEFAULT_M_PROB_GENERAL
                self.u_probs[field] = DEFAULT_U_PROB
        
        logger.info(f"Initial m-probs: {self.m_probs}")
        logger.info(f"Initial u-probs: {self.u_probs}")
    
    def expectation_step(self, comparison_data: pd.DataFrame) -> pd.DataFrame:
        """
        E-step: Calculate expected match indicators given current parameters.
        
        Args:
            comparison_data: DataFrame with agreement patterns
            
        Returns:
            DataFrame with added 'match_probability' column
        """
        # Calculate likelihood of agreement pattern given match
        log_likelihood_match = np.zeros(len(comparison_data))
        log_likelihood_nonmatch = np.zeros(len(comparison_data))
        
        for field in self.field_names:
            if f'{field}_similarity' in comparison_data.columns:
                similarities = comparison_data[f'{field}_similarity'].values
                
                # Get current parameters with safeguards
                m = self.m_probs.get(field, 0.9)
                u = self.u_probs.get(field, 0.1)
                
                # Prevent log(0) and ensure valid range
                # First, clamp to safe ranges
                m = np.clip(m, EM_MIN_PROBABILITY, EM_MAX_PROBABILITY)
                u = np.clip(u, EM_MIN_PROBABILITY, EM_MAX_PROBABILITY)
                
                # Ensure m > u for logical consistency
                # Use a more robust approach to maintain separation
                if m <= u:
                    # Calculate midpoint and separation
                    midpoint = (m + u) / 2
                    separation = EM_MIN_SEPARATION
                    
                    # Adjust values while staying in bounds
                    if midpoint + separation/2 <= EM_MAX_PROBABILITY and midpoint - separation/2 >= EM_MIN_PROBABILITY:
                        m = midpoint + separation/2
                        u = midpoint - separation/2
                    else:
                        # Fallback to safe defaults if adjustment fails
                        m = DEFAULT_M_PROB_GENERAL
                        u = DEFAULT_U_PROB
                        logger.warning(f"Reset m/u probabilities to defaults for field {field} due to convergence issues")
                
                # For each record, calculate contribution to log-likelihood
                # Using similarity as continuous measure of agreement
                log_likelihood_match += similarities * np.log(m) + (1 - similarities) * np.log(1 - m)
                log_likelihood_nonmatch += similarities * np.log(u) + (1 - similarities) * np.log(1 - u)
        
        # Convert to probabilities using Bayes' theorem
        log_prior_match = np.log(self.prior_match)
        log_prior_nonmatch = np.log(1 - self.prior_match)
        
        log_posterior_match = log_likelihood_match + log_prior_match
        log_posterior_nonmatch = log_likelihood_nonmatch + log_prior_nonmatch
        
        # Normalize to get probabilities
        max_log = np.maximum(log_posterior_match, log_posterior_nonmatch)
        exp_match = np.exp(log_posterior_match - max_log)
        exp_nonmatch = np.exp(log_posterior_nonmatch - max_log)
        
        match_probabilities = exp_match / (exp_match + exp_nonmatch)
        
        # Add to dataframe
        comparison_data = comparison_data.copy()
        comparison_data['match_probability'] = match_probabilities
        
        # Calculate log-likelihood for convergence check
        log_likelihood = np.sum(
            match_probabilities * log_posterior_match +
            (1 - match_probabilities) * log_posterior_nonmatch
        )
        self.log_likelihood_history.append(log_likelihood)
        
        return comparison_data
    
    def maximization_step(self, comparison_data: pd.DataFrame, regularization: float = 0.01):
        """
        M-step: Update m and u probabilities based on expected matches with regularization.
        
        Args:
            comparison_data: DataFrame with match probabilities
            regularization: Regularization parameter to prevent extreme values
        """
        match_weights = comparison_data['match_probability'].values
        nonmatch_weights = 1 - match_weights
        
        # Update parameters for each field
        new_m_probs = {}
        new_u_probs = {}
        
        for field in self.field_names:
            if f'{field}_similarity' in comparison_data.columns:
                similarities = comparison_data[f'{field}_similarity'].values
                
                # Update m probability with regularization (agreement rate for matches)
                numerator_m = np.sum(similarities * match_weights) + regularization
                denominator_m = np.sum(match_weights) + 2 * regularization
                
                # Robust division by zero handling with logging
                if denominator_m > EM_MIN_PROBABILITY:
                    new_m_probs[field] = numerator_m / denominator_m
                else:
                    # No matches found in this iteration - use safe default
                    logger.warning(f"No match weight for field {field}, using default m_prob")
                    new_m_probs[field] = self.m_probs.get(field, DEFAULT_M_PROB_GENERAL)
                
                # Update u probability with regularization (agreement rate for non-matches)
                numerator_u = np.sum(similarities * nonmatch_weights) + regularization
                denominator_u = np.sum(nonmatch_weights) + 2 * regularization
                
                # Robust division by zero handling with logging
                if denominator_u > EM_MIN_PROBABILITY:
                    new_u_probs[field] = numerator_u / denominator_u
                else:
                    # No non-matches found - this is unusual, use safe default
                    logger.warning(f"No non-match weight for field {field}, using default u_prob")
                    new_u_probs[field] = self.u_probs.get(field, DEFAULT_U_PROB)
                
                # Apply bounds to prevent extreme values
                new_m_probs[field] = max(0.5, min(0.99, new_m_probs[field]))
                new_u_probs[field] = max(0.01, min(0.5, new_u_probs[field]))
                
                # Ensure m > u with minimum gap
                min_gap = 0.2
                if new_m_probs[field] - new_u_probs[field] < min_gap:
                    # Adjust to maintain minimum gap
                    mid_point = (new_m_probs[field] + new_u_probs[field]) / 2
                    new_m_probs[field] = min(mid_point + min_gap/2, 0.95)
                    new_u_probs[field] = max(mid_point - min_gap/2, 0.05)
                
                # Additional safety check
                if new_m_probs[field] <= new_u_probs[field]:
                    new_m_probs[field] = min(0.8, new_u_probs[field] + 0.3)
                    new_u_probs[field] = max(0.05, new_m_probs[field] - 0.3)
            else:
                # Keep previous values if field not found
                new_m_probs[field] = self.m_probs.get(field, 0.9)
                new_u_probs[field] = self.u_probs.get(field, 0.1)
        
        # Store parameter history
        self.parameter_history.append({
            'm_probs': self.m_probs.copy(),
            'u_probs': self.u_probs.copy()
        })
        
        # Update parameters
        self.m_probs = new_m_probs
        self.u_probs = new_u_probs
    
    def check_convergence(self, tolerance: float = 0.0001, patience: int = 5) -> bool:
        """
        Check if algorithm has converged based on parameter changes and likelihood plateau.
        
        Args:
            tolerance: Maximum allowed change in parameters
            patience: Number of iterations to check for plateau
            
        Returns:
            True if converged
        """
        if len(self.parameter_history) < 2:
            return False
        
        prev_params = self.parameter_history[-2]
        
        # Check maximum change in any parameter
        max_change = 0.0
        
        for field in self.field_names:
            m_change = abs(self.m_probs[field] - prev_params['m_probs'].get(field, 0))
            u_change = abs(self.u_probs[field] - prev_params['u_probs'].get(field, 0))
            max_change = max(max_change, m_change, u_change)
        
        # Check log-likelihood convergence with plateau detection
        ll_converged = False
        if len(self.log_likelihood_history) >= patience:
            # Check if likelihood has plateaued over last 'patience' iterations
            recent_lls = self.log_likelihood_history[-patience:]
            ll_std = np.std(recent_lls)
            ll_mean = np.mean(recent_lls)
            
            # Consider converged if std is very small relative to mean
            if ll_mean != 0:
                ll_converged = (ll_std / abs(ll_mean)) < tolerance
            
            # Also check absolute change
            ll_change = abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2])
            ll_converged = ll_converged or (ll_change < tolerance * max(1, abs(self.log_likelihood_history[-1])))
        
        return max_change < tolerance and ll_converged
    
    def fit(self, comparison_data: pd.DataFrame, max_iterations: int = 100,
            tolerance: float = 0.0001, verbose: bool = True) -> Tuple[Dict, Dict]:
        """
        Run EM algorithm until convergence.
        
        Args:
            comparison_data: DataFrame with similarity scores for each field
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            Tuple of (m_probabilities, u_probabilities)
        """
        # Initialize if not already done
        if not self.m_probs:
            self.initialize_parameters(comparison_data)
        
        logger.info(f"Starting EM algorithm with {len(comparison_data)} comparisons")
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # E-step
            comparison_data = self.expectation_step(comparison_data)
            
            # M-step
            self.maximization_step(comparison_data)
            
            # Check convergence
            if self.check_convergence(tolerance):
                self.converged = True
                logger.info(f"EM converged in {iteration + 1} iterations")
                break
            
            # Progress logging
            if verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: LL = {self.log_likelihood_history[-1]:.2f}")
                self._log_current_parameters()
        
        if not self.converged:
            logger.warning(f"EM did not converge in {max_iterations} iterations")
        
        # Final parameter logging
        logger.info("Final EM parameters:")
        self._log_current_parameters()
        
        return self.m_probs, self.u_probs
    
    def _log_current_parameters(self):
        """Log current parameter values."""
        for field in self.field_names:
            logger.info(f"  {field}: m={self.m_probs[field]:.4f}, u={self.u_probs[field]:.4f}")
    
    def get_convergence_diagnostics(self) -> Dict:
        """
        Get diagnostics about algorithm convergence.
        
        Returns:
            Dictionary with convergence information
        """
        diagnostics = {
            'converged': self.converged,
            'iterations': self.iteration + 1,
            'final_log_likelihood': self.log_likelihood_history[-1] if self.log_likelihood_history else None,
            'log_likelihood_history': self.log_likelihood_history,
            'parameter_changes': []
        }
        
        # Calculate parameter changes over iterations
        for i in range(1, len(self.parameter_history)):
            prev = self.parameter_history[i-1]
            curr = self.parameter_history[i]
            
            max_change = 0.0
            for field in self.field_names:
                m_change = abs(curr['m_probs'].get(field, 0) - prev['m_probs'].get(field, 0))
                u_change = abs(curr['u_probs'].get(field, 0) - prev['u_probs'].get(field, 0))
                max_change = max(max_change, m_change, u_change)
            
            diagnostics['parameter_changes'].append(max_change)
        
        return diagnostics
    
    def calculate_weights_distribution(self, comparison_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate distribution of weights given learned parameters.
        
        Args:
            comparison_data: DataFrame with similarity scores
            
        Returns:
            Array of weights for all comparisons
        """
        weights = []
        
        for _, row in comparison_data.iterrows():
            weight = 0.0
            
            for field in self.field_names:
                if f'{field}_similarity' in row:
                    similarity = row[f'{field}_similarity']
                    m = self.m_probs[field]
                    u = self.u_probs[field]
                    
                    # Calculate log-weight contribution
                    if similarity >= 0.88:  # Agreement
                        weight += np.log2(m / u)
                    elif similarity <= 0.66:  # Disagreement
                        weight += np.log2((1 - m) / (1 - u))
                    else:  # Partial agreement
                        weight += similarity * np.log2(m / u) + (1 - similarity) * np.log2((1 - m) / (1 - u))
            
            weights.append(weight)
        
        return np.array(weights)


def create_em_summary_report(em_algorithm: EMAlgorithm) -> str:
    """
    Create a summary report of EM algorithm results.
    
    Args:
        em_algorithm: Fitted EMAlgorithm instance
        
    Returns:
        Summary report string
    """
    diagnostics = em_algorithm.get_convergence_diagnostics()
    
    report = []
    report.append("=" * 60)
    report.append("EM ALGORITHM SUMMARY")
    report.append("=" * 60)
    report.append(f"Converged: {diagnostics['converged']}")
    report.append(f"Iterations: {diagnostics['iterations']}")
    report.append(f"Final log-likelihood: {diagnostics['final_log_likelihood']:.2f}")
    report.append("")
    
    report.append("Learned Parameters:")
    report.append("-" * 40)
    report.append(f"{'Field':<20} {'m-prob':<10} {'u-prob':<10} {'Ratio':<10}")
    report.append("-" * 40)
    
    for field in em_algorithm.field_names:
        m = em_algorithm.m_probs[field]
        u = em_algorithm.u_probs[field]
        ratio = m / u if u > 0 else float('inf')
        report.append(f"{field:<20} {m:<10.4f} {u:<10.4f} {ratio:<10.2f}")
    
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test EM algorithm
    print("Testing EM Algorithm\n")
    
    # Create sample comparison data
    np.random.seed(42)
    n_matches = 100
    n_nonmatches = 900
    
    # Generate match comparisons (high similarity)
    match_data = pd.DataFrame({
        'state_similarity': np.random.beta(9, 1, n_matches),  # Mostly high
        'income_similarity': np.random.beta(7, 3, n_matches),
        'size_similarity': np.random.beta(8, 2, n_matches),
        'is_match': 1
    })
    
    # Generate non-match comparisons (low similarity)
    nonmatch_data = pd.DataFrame({
        'state_similarity': np.random.beta(1, 9, n_nonmatches),  # Mostly low
        'income_similarity': np.random.beta(2, 8, n_nonmatches),
        'size_similarity': np.random.beta(1.5, 8.5, n_nonmatches),
        'is_match': 0
    })
    
    # Combine data (EM doesn't see the true labels)
    comparison_data = pd.concat([match_data, nonmatch_data], ignore_index=True)
    comparison_data = comparison_data.drop('is_match', axis=1)  # Remove labels
    
    # Initialize and run EM
    field_names = ['state', 'income', 'size']
    em = EMAlgorithm(field_names, prior_match=0.1)
    
    m_probs, u_probs = em.fit(comparison_data, verbose=True)
    
    # Print results
    print("\n" + create_em_summary_report(em))
    
    # Show convergence
    print("\nConvergence diagnostics:")
    diagnostics = em.get_convergence_diagnostics()
    print(f"Parameter changes: {diagnostics['parameter_changes'][:10]}")  # First 10