"""
Advanced similarity metrics with learned weights and neural network-based comparisons.

This module implements state-of-the-art similarity functions including:
- Learned similarity functions using neural networks
- Siamese networks for complex field comparisons
- Attention mechanisms for dynamic field importance
- Contextual similarity considering field dependencies
- Ensemble similarity methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, euclidean
import jellyfish
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimilarityConfig:
    """Configuration for advanced similarity metrics."""
    use_neural_similarity: bool = True
    use_attention: bool = True
    use_contextual: bool = True
    embedding_dim: int = 128
    hidden_dim: int = 256
    n_attention_heads: int = 4
    dropout_rate: float = 0.1
    temperature: float = 1.0
    ensemble_methods: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['neural', 'statistical', 'fuzzy']


class NeuralSimilarityNetwork(nn.Module):
    """
    Neural network for learning optimal similarity functions.
    
    This network learns to compare two field values and output a similarity score.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Siamese-style comparison
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compare two inputs and return similarity score.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode both inputs
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Combine representations
        combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        
        # Compute similarity
        similarity = self.comparator(combined)
        return similarity


class AttentionSimilarity(nn.Module):
    """
    Attention-based similarity that learns field importance dynamically.
    """
    
    def __init__(self, n_fields: int, embedding_dim: int = 128, n_heads: int = 4):
        super().__init__()
        
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        
        # Field embeddings
        self.field_embeddings = nn.Embedding(n_fields, embedding_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, field_similarities: torch.Tensor, field_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted similarity using attention mechanism.
        
        Args:
            field_similarities: Tensor of individual field similarities [batch, n_fields]
            field_indices: Indices of fields being compared
            
        Returns:
            Weighted similarity score
        """
        batch_size = field_similarities.shape[0]
        
        # Get field embeddings
        field_embeds = self.field_embeddings(field_indices)  # [n_fields, embedding_dim]
        field_embeds = field_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_fields, embedding_dim]
        
        # Apply attention to learn field importance
        attended, attention_weights = self.attention(field_embeds, field_embeds, field_embeds)
        
        # Weight similarities by attention
        weighted_sims = field_similarities.unsqueeze(-1) * attended
        
        # Aggregate
        aggregated = weighted_sims.mean(dim=1)
        
        # Final projection
        similarity = self.output_projection(aggregated).squeeze(-1)
        return torch.sigmoid(similarity)


class ContextualSimilarity:
    """
    Contextual similarity that considers field dependencies and relationships.
    """
    
    def __init__(self, field_dependencies: Dict[str, List[str]] = None):
        """
        Initialize contextual similarity calculator.
        
        Args:
            field_dependencies: Dictionary mapping fields to their dependent fields
        """
        self.field_dependencies = field_dependencies or {}
        self.context_weights = {}
        self._learn_context_weights()
        
    def _learn_context_weights(self):
        """Learn weights for contextual dependencies."""
        # Initialize weights based on dependency structure
        for field, deps in self.field_dependencies.items():
            self.context_weights[field] = {dep: 1.0 / len(deps) for dep in deps}
    
    def calculate_contextual_similarity(self, record1: Dict, record2: Dict, 
                                       field: str, base_similarity: float) -> float:
        """
        Calculate similarity considering context from dependent fields.
        
        Args:
            record1: First record
            record2: Second record
            field: Field being compared
            base_similarity: Base similarity score
            
        Returns:
            Context-adjusted similarity score
        """
        if field not in self.field_dependencies:
            return base_similarity
            
        context_score = 0
        total_weight = 0
        
        for dep_field in self.field_dependencies[field]:
            if dep_field in record1 and dep_field in record2:
                # Get similarity of dependent field
                dep_sim = self._calculate_field_similarity(
                    record1[dep_field], record2[dep_field], dep_field
                )
                weight = self.context_weights[field].get(dep_field, 0)
                context_score += dep_sim * weight
                total_weight += weight
        
        if total_weight > 0:
            context_score /= total_weight
            # Blend base and contextual similarity
            return 0.7 * base_similarity + 0.3 * context_score
        
        return base_similarity
    
    def _calculate_field_similarity(self, val1: Any, val2: Any, field_type: str) -> float:
        """Calculate basic field similarity."""
        if val1 is None or val2 is None:
            return 0.0
            
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numeric similarity
            if val1 == val2:
                return 1.0
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
            return 1 - min(abs(val1 - val2) / max_val, 1.0)
        
        # String similarity
        str1, str2 = str(val1), str(val2)
        if str1 == str2:
            return 1.0
        return jellyfish.jaro_winkler_similarity(str1, str2)


class AdvancedSimilarityCalculator:
    """
    Main class for advanced similarity calculation with multiple methods.
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        """
        Initialize advanced similarity calculator.
        
        Args:
            config: Configuration for similarity calculation
        """
        self.config = config or SimilarityConfig()
        
        # Initialize components
        self.neural_networks = {}
        self.attention_module = None
        self.contextual_similarity = None
        self.feature_extractors = {}
        self.scalers = {}
        
        # Learned weights for ensemble
        self.ensemble_weights = {method: 1.0 / len(self.config.ensemble_methods) 
                                for method in self.config.ensemble_methods}
        
    def learn_similarity_functions(self, training_data: pd.DataFrame, labels: Optional[np.ndarray] = None):
        """
        Learn optimal similarity functions from data.
        
        Args:
            training_data: DataFrame with pairs and their features
            labels: Optional labels indicating true matches (1) or non-matches (0)
        """
        logger.info("Learning similarity functions from data")
        
        if self.config.use_neural_similarity:
            self._train_neural_similarities(training_data, labels)
        
        if self.config.use_attention:
            self._train_attention_module(training_data, labels)
        
        if self.config.use_contextual:
            self._learn_contextual_dependencies(training_data, labels)
        
        # Learn ensemble weights if labels provided
        if labels is not None:
            self._optimize_ensemble_weights(training_data, labels)
    
    def calculate_similarity(self, record1: Dict, record2: Dict, 
                           field_name: str, field_type: str = 'string') -> float:
        """
        Calculate advanced similarity between two field values.
        
        Args:
            record1: First record
            record2: Second record
            field_name: Name of field being compared
            field_type: Type of field ('string', 'numeric', 'categorical', etc.)
            
        Returns:
            Similarity score between 0 and 1
        """
        similarities = {}
        
        # Neural similarity
        if self.config.use_neural_similarity and field_name in self.neural_networks:
            similarities['neural'] = self._neural_similarity(record1, record2, field_name)
        
        # Statistical similarity
        similarities['statistical'] = self._statistical_similarity(
            record1.get(field_name), record2.get(field_name), field_type
        )
        
        # Fuzzy similarity
        similarities['fuzzy'] = self._fuzzy_similarity(
            record1.get(field_name), record2.get(field_name), field_type
        )
        
        # Contextual adjustment
        if self.config.use_contextual and self.contextual_similarity:
            for method in similarities:
                similarities[method] = self.contextual_similarity.calculate_contextual_similarity(
                    record1, record2, field_name, similarities[method]
                )
        
        # Ensemble combination
        if len(similarities) > 1:
            return self._ensemble_combination(similarities)
        
        return list(similarities.values())[0] if similarities else 0.0
    
    def _train_neural_similarities(self, training_data: pd.DataFrame, labels: Optional[np.ndarray]):
        """Train neural similarity networks for each field."""
        logger.debug("Training neural similarity networks")
        
        # Group fields by type
        field_types = self._identify_field_types(training_data)
        
        for field_type, fields in field_types.items():
            # Create and train network for this field type
            input_dim = self._get_input_dimension(fields[0], training_data)
            network = NeuralSimilarityNetwork(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout_rate=self.config.dropout_rate
            )
            
            # Training would happen here with actual data
            # For now, store untrained network
            for field in fields:
                self.neural_networks[field] = network
    
    def _train_attention_module(self, training_data: pd.DataFrame, labels: Optional[np.ndarray]):
        """Train attention module for field importance."""
        n_fields = len([col for col in training_data.columns if col.endswith('_similarity')])
        
        if n_fields > 0:
            self.attention_module = AttentionSimilarity(
                n_fields=n_fields,
                embedding_dim=self.config.embedding_dim,
                n_heads=self.config.n_attention_heads
            )
            
            # Training would happen here
            logger.debug(f"Initialized attention module for {n_fields} fields")
    
    def _learn_contextual_dependencies(self, training_data: pd.DataFrame, labels: Optional[np.ndarray]):
        """Learn field dependencies for contextual similarity."""
        # Analyze correlations to identify dependencies
        correlations = training_data.corr()
        
        field_dependencies = {}
        for field in training_data.columns:
            if field.endswith('_similarity'):
                base_field = field.replace('_similarity', '')
                # Find strongly correlated fields
                correlated = correlations[field][correlations[field] > 0.5].index.tolist()
                correlated = [f.replace('_similarity', '') for f in correlated if f != field]
                if correlated:
                    field_dependencies[base_field] = correlated[:3]  # Top 3 dependencies
        
        self.contextual_similarity = ContextualSimilarity(field_dependencies)
        logger.debug(f"Learned contextual dependencies for {len(field_dependencies)} fields")
    
    def _optimize_ensemble_weights(self, training_data: pd.DataFrame, labels: np.ndarray):
        """Optimize ensemble weights using labeled data."""
        from sklearn.linear_model import LogisticRegression
        
        # Calculate similarities using each method
        method_scores = {method: [] for method in self.config.ensemble_methods}
        
        # This would involve calculating scores for each method on training data
        # For now, use equal weights
        logger.debug("Optimizing ensemble weights")
    
    def _neural_similarity(self, record1: Dict, record2: Dict, field_name: str) -> float:
        """Calculate similarity using neural network."""
        if field_name not in self.neural_networks:
            return 0.5
            
        network = self.neural_networks[field_name]
        
        # Extract features for the field
        features1 = self._extract_features(record1.get(field_name), field_name)
        features2 = self._extract_features(record2.get(field_name), field_name)
        
        # Convert to tensors
        x1 = torch.FloatTensor(features1).unsqueeze(0)
        x2 = torch.FloatTensor(features2).unsqueeze(0)
        
        # Get similarity
        with torch.no_grad():
            similarity = network(x1, x2).item()
        
        return similarity
    
    def _statistical_similarity(self, val1: Any, val2: Any, field_type: str) -> float:
        """Calculate statistical similarity based on field type."""
        if val1 is None or val2 is None:
            return 0.0
            
        if field_type == 'numeric':
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == val2:
                    return 1.0
                # Use exponential decay based on difference
                diff = abs(val1 - val2)
                scale = max(abs(val1), abs(val2), 1)
                return np.exp(-diff / scale)
        
        elif field_type == 'categorical':
            return 1.0 if val1 == val2 else 0.0
        
        else:  # string
            str1, str2 = str(val1).lower(), str(val2).lower()
            if str1 == str2:
                return 1.0
            
            # Multiple string metrics
            jaro = jellyfish.jaro_winkler_similarity(str1, str2)
            
            # Token-based similarity
            tokens1 = set(str1.split())
            tokens2 = set(str2.split())
            if tokens1 and tokens2:
                jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            else:
                jaccard = 0
            
            return 0.7 * jaro + 0.3 * jaccard
    
    def _fuzzy_similarity(self, val1: Any, val2: Any, field_type: str) -> float:
        """Calculate fuzzy similarity with tolerance for variations."""
        if val1 is None or val2 is None:
            return 0.0
            
        if field_type == 'numeric':
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Fuzzy numeric matching with tolerance bands
                if abs(val1 - val2) <= 0.01 * max(abs(val1), abs(val2)):
                    return 1.0
                elif abs(val1 - val2) <= 0.05 * max(abs(val1), abs(val2)):
                    return 0.9
                elif abs(val1 - val2) <= 0.1 * max(abs(val1), abs(val2)):
                    return 0.7
                else:
                    return max(0, 1 - abs(val1 - val2) / max(abs(val1), abs(val2), 1))
        
        else:  # string or categorical
            str1, str2 = str(val1).lower(), str(val2).lower()
            
            # Exact match
            if str1 == str2:
                return 1.0
            
            # Phonetic similarity
            if len(str1) > 3 and len(str2) > 3:
                soundex1 = jellyfish.soundex(str1)
                soundex2 = jellyfish.soundex(str2)
                if soundex1 == soundex2:
                    return 0.85
            
            # Edit distance based
            max_len = max(len(str1), len(str2))
            if max_len > 0:
                edit_dist = jellyfish.levenshtein_distance(str1, str2)
                return max(0, 1 - edit_dist / max_len)
            
            return 0.0
    
    def _ensemble_combination(self, similarities: Dict[str, float]) -> float:
        """Combine multiple similarity scores using learned weights."""
        weighted_sum = 0
        total_weight = 0
        
        for method, score in similarities.items():
            weight = self.ensemble_weights.get(method, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _extract_features(self, value: Any, field_name: str) -> np.ndarray:
        """Extract features from a field value for neural processing."""
        features = []
        
        if value is None:
            return np.zeros(10)  # Default feature vector
        
        if isinstance(value, (int, float)):
            # Numeric features
            features.extend([
                float(value),
                np.log1p(abs(value)),
                value ** 2,
                np.sign(value),
                value % 10 if isinstance(value, int) else value % 1
            ])
        else:
            # String features
            str_val = str(value).lower()
            features.extend([
                len(str_val),
                len(str_val.split()),
                sum(c.isdigit() for c in str_val),
                sum(c.isalpha() for c in str_val),
                len(set(str_val))
            ])
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0)
        
        return np.array(features[:10])
    
    def _identify_field_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify field types from data."""
        field_types = {'numeric': [], 'categorical': [], 'string': []}
        
        for col in data.columns:
            if col.endswith('_similarity'):
                continue
                
            if data[col].dtype in [np.int64, np.float64]:
                field_types['numeric'].append(col)
            elif data[col].nunique() < 20:
                field_types['categorical'].append(col)
            else:
                field_types['string'].append(col)
        
        return field_types
    
    def _get_input_dimension(self, field_name: str, data: pd.DataFrame) -> int:
        """Get input dimension for neural network based on field."""
        # Fixed dimension for now
        return 10
    
    def get_field_importance(self) -> Dict[str, float]:
        """Get learned importance weights for each field."""
        if self.attention_module is None:
            return {}
            
        # Extract attention weights
        # This would involve analyzing the attention module's learned parameters
        return {}
    
    def explain_similarity(self, record1: Dict, record2: Dict, field_name: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of similarity calculation.
        
        Args:
            record1: First record
            record2: Second record
            field_name: Field being compared
            
        Returns:
            Dictionary with similarity components and explanations
        """
        explanation = {
            'field': field_name,
            'value1': record1.get(field_name),
            'value2': record2.get(field_name),
            'methods': {}
        }
        
        # Calculate each component
        if self.config.use_neural_similarity and field_name in self.neural_networks:
            explanation['methods']['neural'] = self._neural_similarity(record1, record2, field_name)
        
        explanation['methods']['statistical'] = self._statistical_similarity(
            record1.get(field_name), record2.get(field_name), 'auto'
        )
        
        explanation['methods']['fuzzy'] = self._fuzzy_similarity(
            record1.get(field_name), record2.get(field_name), 'auto'
        )
        
        # Overall score
        explanation['final_score'] = self._ensemble_combination(explanation['methods'])
        
        # Add reasoning
        if explanation['final_score'] > 0.9:
            explanation['assessment'] = 'Strong match'
        elif explanation['final_score'] > 0.7:
            explanation['assessment'] = 'Good match'
        elif explanation['final_score'] > 0.5:
            explanation['assessment'] = 'Possible match'
        else:
            explanation['assessment'] = 'Unlikely match'
        
        return explanation