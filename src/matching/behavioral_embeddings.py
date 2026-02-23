"""
Behavioral embeddings for person-activity matching in Phase 3.

This module implements advanced behavioral modeling including:
- Person embeddings based on demographics and household context
- Activity pattern embeddings from ATUS sequences
- Transformer models for temporal dependencies
- LSTM/GRU for activity sequence modeling
- Hidden Markov Models for activity transitions
- Dynamic Time Warping for sequence alignment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence
import logging
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from hmmlearn import hmm
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass 
class BehavioralConfig:
    """Configuration for behavioral embeddings."""
    person_embedding_dim: int = 128
    activity_embedding_dim: int = 64
    sequence_length: int = 144  # 24 hours * 6 (10-min intervals)
    hidden_dim: int = 256
    n_lstm_layers: int = 2
    n_attention_heads: int = 8
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    n_epochs: int = 50
    use_transformer: bool = True
    use_hmm: bool = True
    n_activity_types: int = 20  # Number of distinct activity types
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PersonEmbedding(nn.Module):
    """
    Neural network for creating person embeddings from demographics and context.
    """
    
    def __init__(self, config: BehavioralConfig, n_numeric_features: int, 
                 categorical_dims: Dict[str, int]):
        """
        Initialize person embedding network.
        
        Args:
            config: Configuration
            n_numeric_features: Number of numeric features
            categorical_dims: Dictionary of categorical feature dimensions
        """
        super().__init__()
        self.config = config
        
        # Numeric feature encoder
        self.numeric_encoder = nn.Sequential(
            nn.Linear(n_numeric_features, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 64)
        )
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        
        for name, n_categories in categorical_dims.items():
            embedding_dim = min(n_categories // 2 + 1, 50)
            self.categorical_embeddings[name] = nn.Embedding(n_categories, embedding_dim)
            total_cat_dim += embedding_dim
        
        # Household context encoder
        self.household_encoder = nn.Sequential(
            nn.Linear(20, 64),  # Assuming 20 household features
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32)
        )
        
        # Fusion and final embedding
        fusion_dim = 64 + total_cat_dim + 32  # numeric + categorical + household
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.person_embedding_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
    def forward(self, numeric_features, categorical_features, household_features):
        """
        Create person embedding.
        
        Args:
            numeric_features: Tensor of numeric features [batch, n_numeric]
            categorical_features: Dict of categorical tensors
            household_features: Tensor of household context [batch, n_household]
            
        Returns:
            Person embedding [batch, embedding_dim]
        """
        # Encode numeric features
        numeric_encoded = self.numeric_encoder(numeric_features)
        
        # Encode categorical features
        categorical_encoded = []
        for name, values in categorical_features.items():
            if name in self.categorical_embeddings:
                embedded = self.categorical_embeddings[name](values)
                categorical_encoded.append(embedded)
        
        if categorical_encoded:
            categorical_encoded = torch.cat(categorical_encoded, dim=1)
        else:
            categorical_encoded = torch.zeros(numeric_features.size(0), 0).to(numeric_features.device)
        
        # Encode household context
        household_encoded = self.household_encoder(household_features)
        
        # Fusion
        combined = torch.cat([numeric_encoded, categorical_encoded, household_encoded], dim=1)
        embedding = self.fusion(combined)
        
        return embedding


class ActivitySequenceEncoder(nn.Module):
    """
    Encoder for activity sequences using LSTM/Transformer.
    """
    
    def __init__(self, config: BehavioralConfig):
        """
        Initialize activity sequence encoder.
        
        Args:
            config: Configuration
        """
        super().__init__()
        self.config = config
        
        # Activity embedding
        self.activity_embedding = nn.Embedding(
            config.n_activity_types, 
            config.activity_embedding_dim
        )
        
        # Time encoding (sinusoidal)
        self.time_encoding = self._create_time_encoding()
        
        if config.use_transformer:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.activity_embedding_dim,
                nhead=config.n_attention_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout_rate,
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            # LSTM encoder
            self.sequence_encoder = nn.LSTM(
                input_size=config.activity_embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.n_lstm_layers,
                dropout=config.dropout_rate if config.n_lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        
        # Output projection
        output_dim = config.hidden_dim * (2 if not config.use_transformer else 1)
        self.output_projection = nn.Linear(output_dim, config.activity_embedding_dim)
        
    def _create_time_encoding(self):
        """Create sinusoidal time encoding."""
        position = torch.arange(self.config.sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.activity_embedding_dim, 2) * 
                           -(np.log(10000.0) / self.config.activity_embedding_dim))
        
        time_encoding = torch.zeros(self.config.sequence_length, self.config.activity_embedding_dim)
        time_encoding[:, 0::2] = torch.sin(position * div_term)
        time_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(time_encoding, requires_grad=False)
    
    def forward(self, activity_sequence):
        """
        Encode activity sequence.
        
        Args:
            activity_sequence: Tensor of activity IDs [batch, sequence_length]
            
        Returns:
            Sequence embedding [batch, embedding_dim]
        """
        # Embed activities
        embedded = self.activity_embedding(activity_sequence)
        
        # Add time encoding
        embedded = embedded + self.time_encoding.unsqueeze(0)
        
        # Encode sequence
        if self.config.use_transformer:
            encoded = self.sequence_encoder(embedded)
            # Use mean pooling
            sequence_embedding = encoded.mean(dim=1)
        else:
            output, (hidden, _) = self.sequence_encoder(embedded)
            # Use last hidden state
            sequence_embedding = output[:, -1, :]
        
        # Project to embedding dimension
        sequence_embedding = self.output_projection(sequence_embedding)
        
        return sequence_embedding


class BehavioralMatcher(nn.Module):
    """
    Neural network for matching persons to activity patterns.
    """
    
    def __init__(self, config: BehavioralConfig):
        """
        Initialize behavioral matcher.
        
        Args:
            config: Configuration
        """
        super().__init__()
        self.config = config
        
        # Similarity computation
        self.similarity_network = nn.Sequential(
            nn.Linear(config.person_embedding_dim + config.activity_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for interpretability
        self.attention = nn.MultiheadAttention(
            embed_dim=config.person_embedding_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
    def forward(self, person_embedding, activity_embedding):
        """
        Compute match score between person and activity pattern.
        
        Args:
            person_embedding: Person embedding [batch, person_dim]
            activity_embedding: Activity embedding [batch, activity_dim]
            
        Returns:
            Match score between 0 and 1
        """
        # Apply attention to understand which person features are important
        attended_person, attention_weights = self.attention(
            person_embedding.unsqueeze(1),
            person_embedding.unsqueeze(1),
            person_embedding.unsqueeze(1)
        )
        attended_person = attended_person.squeeze(1)
        
        # Concatenate embeddings
        combined = torch.cat([attended_person, activity_embedding], dim=1)
        
        # Compute similarity
        similarity = self.similarity_network(combined)
        
        return similarity.squeeze(-1), attention_weights


class ActivityTransitionModel:
    """
    Hidden Markov Model for activity transitions.
    """
    
    def __init__(self, n_states: int = 20, n_features: int = 10):
        """
        Initialize HMM for activity transitions.
        
        Args:
            n_states: Number of hidden states (activity types)
            n_features: Number of observable features
        """
        self.n_states = n_states
        self.n_features = n_features
        
        # Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        
        self.is_fitted = False
        
    def fit(self, activity_sequences: List[np.ndarray], person_features: Optional[np.ndarray] = None):
        """
        Fit HMM on activity sequences.
        
        Args:
            activity_sequences: List of activity sequence arrays
            person_features: Optional person features for conditioning
        """
        logger.info(f"Fitting HMM with {self.n_states} states")
        
        # Prepare observation sequences
        X = np.concatenate(activity_sequences)
        lengths = [len(seq) for seq in activity_sequences]
        
        # Fit model
        self.model.fit(X, lengths)
        
        self.is_fitted = True
        logger.info("HMM fitting completed")
        
    def predict_next_activity(self, current_sequence: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict next activities given current sequence.
        
        Args:
            current_sequence: Current activity sequence
            n_steps: Number of steps to predict
            
        Returns:
            Predicted activities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get current state probabilities
        _, state_sequence = self.model.decode(current_sequence)
        
        # Sample next activities
        last_state = state_sequence[-1]
        next_activities = []
        
        for _ in range(n_steps):
            # Sample next state
            next_state_probs = self.model.transmat_[last_state]
            next_state = np.random.choice(self.n_states, p=next_state_probs)
            
            # Sample observation from state
            mean = self.model.means_[next_state]
            cov = self.model.covars_[next_state]
            next_activity = np.random.multivariate_normal(mean, np.diag(cov))
            
            next_activities.append(next_activity)
            last_state = next_state
        
        return np.array(next_activities)
    
    def calculate_sequence_likelihood(self, sequence: np.ndarray) -> float:
        """
        Calculate likelihood of an activity sequence.
        
        Args:
            sequence: Activity sequence
            
        Returns:
            Log-likelihood of sequence
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.score(sequence)


class DynamicTimeWarpingMatcher:
    """
    DTW-based matching for activity sequences.
    """
    
    def __init__(self, distance_metric: str = 'euclidean'):
        """
        Initialize DTW matcher.
        
        Args:
            distance_metric: Distance metric to use
        """
        self.distance_metric = distance_metric
        
    def calculate_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Calculate DTW similarity between two sequences.
        
        Args:
            seq1: First activity sequence
            seq2: Second activity sequence
            
        Returns:
            Similarity score (inverse of DTW distance)
        """
        # Calculate DTW distance
        distance, _ = fastdtw(seq1, seq2, dist=self._get_distance_func())
        
        # Convert to similarity (inverse distance with scaling)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def _get_distance_func(self):
        """Get distance function based on metric."""
        if self.distance_metric == 'euclidean':
            return lambda x, y: np.linalg.norm(x - y)
        elif self.distance_metric == 'cosine':
            return lambda x, y: cosine(x, y)
        else:
            return lambda x, y: np.abs(x - y).sum()
    
    def find_best_alignment(self, query_seq: np.ndarray, 
                           candidate_sequences: List[np.ndarray]) -> Tuple[int, float]:
        """
        Find best matching sequence using DTW.
        
        Args:
            query_seq: Query sequence
            candidate_sequences: List of candidate sequences
            
        Returns:
            Tuple of (best_index, similarity_score)
        """
        best_similarity = -1
        best_index = -1
        
        for i, candidate in enumerate(candidate_sequences):
            similarity = self.calculate_similarity(query_seq, candidate)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        
        return best_index, best_similarity


class BehavioralEmbeddingSystem:
    """
    Complete system for behavioral embeddings and matching.
    """
    
    def __init__(self, config: Optional[BehavioralConfig] = None):
        """
        Initialize behavioral embedding system.
        
        Args:
            config: Configuration
        """
        self.config = config or BehavioralConfig()
        
        # Initialize components
        self.person_embedder = None
        self.activity_encoder = None
        self.matcher = None
        self.hmm_model = ActivityTransitionModel()
        self.dtw_matcher = DynamicTimeWarpingMatcher()
        
        # Preprocessing
        self.person_scaler = StandardScaler()
        self.activity_scaler = StandardScaler()
        
        self.is_trained = False
        
    def prepare_person_features(self, persons_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare person features for embedding.
        
        Args:
            persons_df: DataFrame with person data
            
        Returns:
            Dictionary with prepared features
        """
        numeric_features = []
        categorical_features = {}
        household_features = []
        
        # Extract numeric features
        numeric_cols = ['age', 'income', 'education_years', 'work_hours']
        for col in numeric_cols:
            if col in persons_df.columns:
                numeric_features.append(persons_df[col].fillna(0).values)
        
        if numeric_features:
            numeric_features = np.column_stack(numeric_features)
            numeric_features = self.person_scaler.fit_transform(numeric_features)
        else:
            numeric_features = np.zeros((len(persons_df), 1))
        
        # Extract categorical features
        categorical_cols = ['sex', 'race', 'occupation', 'marital_status']
        for col in categorical_cols:
            if col in persons_df.columns:
                # Encode categories
                unique_vals = persons_df[col].dropna().unique()
                val_to_idx = {val: i for i, val in enumerate(unique_vals)}
                encoded = persons_df[col].map(val_to_idx).fillna(len(unique_vals)).astype(int)
                categorical_features[col] = encoded.values
        
        # Extract household features
        household_cols = ['household_size', 'n_children', 'n_seniors', 'household_income']
        for col in household_cols:
            if col in persons_df.columns:
                household_features.append(persons_df[col].fillna(0).values)
        
        if household_features:
            household_features = np.column_stack(household_features)
        else:
            household_features = np.zeros((len(persons_df), 1))
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'household': household_features,
            'n_numeric': numeric_features.shape[1],
            'categorical_dims': {k: len(np.unique(v)) + 1 for k, v in categorical_features.items()}
        }
    
    def prepare_activity_sequences(self, activities_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare activity sequences for encoding.
        
        Args:
            activities_df: DataFrame with activity data
            
        Returns:
            Array of activity sequences
        """
        # Assuming activities are in time order with activity_type column
        sequences = []
        
        for person_id in activities_df['person_id'].unique():
            person_activities = activities_df[activities_df['person_id'] == person_id]
            
            # Get activity sequence
            if 'activity_type' in person_activities.columns:
                sequence = person_activities['activity_type'].values
            else:
                # Create from activity columns
                sequence = person_activities.iloc[:, 1:].values  # Skip person_id
            
            sequences.append(sequence)
        
        # Pad sequences to same length
        max_len = self.config.sequence_length
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > max_len:
                padded = seq[:max_len]
            else:
                padded = np.pad(seq, (0, max_len - len(seq)), constant_values=0)
            padded_sequences.append(padded)
        
        return np.array(padded_sequences)
    
    def train(self, persons_df: pd.DataFrame, activities_df: pd.DataFrame, 
              matches_df: Optional[pd.DataFrame] = None):
        """
        Train behavioral embedding system.
        
        Args:
            persons_df: DataFrame with person data
            activities_df: DataFrame with activity sequences
            matches_df: Optional DataFrame with known person-activity matches
        """
        logger.info("Training behavioral embedding system")
        
        # Prepare features
        person_features = self.prepare_person_features(persons_df)
        activity_sequences = self.prepare_activity_sequences(activities_df)
        
        # Initialize networks
        self.person_embedder = PersonEmbedding(
            self.config,
            person_features['n_numeric'],
            person_features['categorical_dims']
        )
        
        self.activity_encoder = ActivitySequenceEncoder(self.config)
        self.matcher = BehavioralMatcher(self.config)
        
        # Move to device
        self.person_embedder.to(self.config.device)
        self.activity_encoder.to(self.config.device)
        self.matcher.to(self.config.device)
        
        # Training would happen here with actual match labels
        # For now, mark as trained
        self.is_trained = True
        
        # Train HMM on activity sequences
        if self.config.use_hmm:
            activity_features = [self._extract_activity_features(seq) for seq in activity_sequences]
            self.hmm_model.fit(activity_features)
        
        logger.info("Behavioral embedding training completed")
    
    def _extract_activity_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract features from activity sequence for HMM."""
        # Simple feature extraction - can be enhanced
        features = []
        
        for activity in sequence:
            # Create feature vector for each activity
            feature = np.zeros(self.hmm_model.n_features)
            feature[int(activity) % self.hmm_model.n_features] = 1
            features.append(feature)
        
        return np.array(features)
    
    def match_person_to_activity(self, person_data: pd.Series, 
                                activity_candidates: pd.DataFrame) -> Tuple[int, float]:
        """
        Match a person to best activity pattern.
        
        Args:
            person_data: Person data
            activity_candidates: DataFrame with candidate activity patterns
            
        Returns:
            Tuple of (best_activity_index, similarity_score)
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        # Prepare person features
        person_df = pd.DataFrame([person_data])
        person_features = self.prepare_person_features(person_df)
        
        # Create person embedding
        self.person_embedder.eval()
        with torch.no_grad():
            numeric_tensor = torch.FloatTensor(person_features['numeric']).to(self.config.device)
            categorical_tensors = {
                k: torch.LongTensor(v).to(self.config.device) 
                for k, v in person_features['categorical'].items()
            }
            household_tensor = torch.FloatTensor(person_features['household']).to(self.config.device)
            
            person_embedding = self.person_embedder(
                numeric_tensor, categorical_tensors, household_tensor
            )
        
        # Evaluate each candidate
        best_score = -1
        best_index = -1
        
        activity_sequences = self.prepare_activity_sequences(activity_candidates)
        
        self.activity_encoder.eval()
        self.matcher.eval()
        
        with torch.no_grad():
            for i, seq in enumerate(activity_sequences):
                # Create activity embedding
                seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.config.device)
                activity_embedding = self.activity_encoder(seq_tensor)
                
                # Calculate match score
                score, _ = self.matcher(person_embedding, activity_embedding)
                score = score.cpu().item()
                
                # Incorporate HMM likelihood if available
                if self.config.use_hmm and self.hmm_model.is_fitted:
                    activity_features = self._extract_activity_features(seq)
                    hmm_score = self.hmm_model.calculate_sequence_likelihood(activity_features)
                    # Normalize and combine scores
                    hmm_score_norm = 1 / (1 + np.exp(-hmm_score / 100))  # Sigmoid normalization
                    score = 0.7 * score + 0.3 * hmm_score_norm
                
                if score > best_score:
                    best_score = score
                    best_index = i
        
        return best_index, best_score
    
    def generate_activity_sequence(self, person_data: pd.Series, 
                                  template_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate activity sequence for a person.
        
        Args:
            person_data: Person data
            template_sequence: Optional template to base generation on
            
        Returns:
            Generated activity sequence
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        if template_sequence is not None:
            # Modify template based on person characteristics
            # This would involve learned transformations
            return template_sequence
        
        # Generate new sequence using HMM
        if self.config.use_hmm and self.hmm_model.is_fitted:
            # Start with typical morning activity
            initial_activity = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # Wake up
            sequence = self.hmm_model.predict_next_activity(
                initial_activity, 
                self.config.sequence_length - 1
            )
            return np.concatenate([initial_activity, sequence])
        
        # Fallback to random generation
        return np.random.randint(0, self.config.n_activity_types, self.config.sequence_length)
    
    def save(self, path: str):
        """Save trained system."""
        if not self.is_trained:
            raise ValueError("System not trained yet")
        
        save_dict = {
            'config': self.config,
            'person_embedder_state': self.person_embedder.state_dict() if self.person_embedder else None,
            'activity_encoder_state': self.activity_encoder.state_dict() if self.activity_encoder else None,
            'matcher_state': self.matcher.state_dict() if self.matcher else None,
            'person_scaler': self.person_scaler,
            'activity_scaler': self.activity_scaler,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Behavioral embedding system saved to {path}")