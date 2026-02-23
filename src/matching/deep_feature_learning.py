"""
Deep feature learning module using autoencoders and neural networks.

This module implements advanced feature learning techniques including:
- Variational Autoencoders (VAE) for feature representation
- Denoising autoencoders for robust features
- Multi-modal autoencoders for heterogeneous data
- Feature interaction learning
- Automated feature selection using attention
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder training."""
    input_dim: int = 100
    encoding_dim: int = 32
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 128
    n_epochs: int = 100
    use_variational: bool = True
    beta_vae: float = 1.0  # Weight for KL divergence in VAE
    use_denoising: bool = True
    noise_factor: float = 0.1
    use_attention: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 48]


class RecordDataset(Dataset):
    """PyTorch dataset for record data."""
    
    def __init__(self, data: np.ndarray, add_noise: bool = False, noise_factor: float = 0.1):
        """
        Initialize dataset.
        
        Args:
            data: Numpy array of features
            add_noise: Whether to add noise for denoising autoencoder
            noise_factor: Noise level
        """
        self.data = torch.FloatTensor(data)
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.add_noise:
            # Add Gaussian noise
            noise = torch.randn_like(sample) * self.noise_factor
            noisy_sample = sample + noise
            return noisy_sample, sample
        
        return sample, sample


class AttentionLayer(nn.Module):
    """Self-attention layer for feature importance learning."""
    
    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.dim_per_head = input_dim // n_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.query(x).view(batch_size, self.n_heads, self.dim_per_head)
        K = self.key(x).view(batch_size, self.n_heads, self.dim_per_head)
        V = self.value(x).view(batch_size, self.n_heads, self.dim_per_head)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dim_per_head)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, -1)
        
        # Output projection
        output = self.output(attended)
        
        return output, attention_weights


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for learning latent representations.
    
    Learns a probabilistic encoding of features.
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # Build encoder
        encoder_layers = []
        current_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE specific layers
        self.fc_mu = nn.Linear(current_dim, config.encoding_dim)
        self.fc_logvar = nn.Linear(current_dim, config.encoding_dim)
        
        # Attention layer if enabled
        if config.use_attention:
            self.attention = AttentionLayer(config.encoding_dim)
        
        # Build decoder
        decoder_layers = []
        current_dim = config.encoding_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(current_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_activation(self):
        """Get activation function."""
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'elu':
            return nn.ELU()
        elif self.config.activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Apply attention if enabled
        if self.config.use_attention:
            z, attention_weights = self.attention(z)
        else:
            attention_weights = None
        
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar, attention_weights
    
    def get_embedding(self, x):
        """Get latent embedding for input."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


class MultiModalAutoencoder(nn.Module):
    """
    Multi-modal autoencoder for heterogeneous data types.
    
    Handles numeric, categorical, and text features separately.
    """
    
    def __init__(self, numeric_dim: int, categorical_dims: Dict[str, int], 
                 text_dim: int, encoding_dim: int = 32):
        super().__init__()
        
        # Numeric encoder
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Categorical encoders (one per categorical variable)
        self.categorical_encoders = nn.ModuleDict()
        categorical_output_dim = 0
        
        for name, n_categories in categorical_dims.items():
            self.categorical_encoders[name] = nn.Embedding(n_categories, 16)
            categorical_output_dim += 16
        
        # Text encoder (assuming pre-processed to fixed dim)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Fusion layer
        fusion_input_dim = 32 + categorical_output_dim + 32  # numeric + categorical + text
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoders
        self.numeric_decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, numeric_dim)
        )
        
        self.categorical_decoders = nn.ModuleDict()
        for name, n_categories in categorical_dims.items():
            self.categorical_decoders[name] = nn.Linear(encoding_dim, n_categories)
        
        self.text_decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, text_dim)
        )
        
    def forward(self, numeric_features, categorical_features, text_features):
        """
        Forward pass through multi-modal autoencoder.
        
        Args:
            numeric_features: Tensor of numeric features
            categorical_features: Dict of categorical feature tensors
            text_features: Tensor of text features
            
        Returns:
            Reconstructions and latent encoding
        """
        # Encode each modality
        numeric_encoded = self.numeric_encoder(numeric_features)
        
        categorical_encoded = []
        for name, features in categorical_features.items():
            if name in self.categorical_encoders:
                encoded = self.categorical_encoders[name](features)
                categorical_encoded.append(encoded)
        
        if categorical_encoded:
            categorical_encoded = torch.cat(categorical_encoded, dim=1)
        else:
            categorical_encoded = torch.zeros(numeric_features.size(0), 0).to(numeric_features.device)
        
        text_encoded = self.text_encoder(text_features)
        
        # Fusion
        fused = torch.cat([numeric_encoded, categorical_encoded, text_encoded], dim=1)
        latent = self.fusion(fused)
        
        # Decode
        numeric_recon = self.numeric_decoder(latent)
        
        categorical_recon = {}
        for name in categorical_features.keys():
            if name in self.categorical_decoders:
                categorical_recon[name] = self.categorical_decoders[name](latent)
        
        text_recon = self.text_decoder(latent)
        
        return numeric_recon, categorical_recon, text_recon, latent


class DeepFeatureLearner:
    """
    Main class for deep feature learning and extraction.
    """
    
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        """
        Initialize deep feature learner.
        
        Args:
            config: Configuration for autoencoder
        """
        self.config = config or AutoencoderConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_importance = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare data for autoencoder training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (feature_matrix, metadata)
        """
        numeric_features = []
        categorical_features = {}
        
        metadata = {
            'numeric_columns': [],
            'categorical_columns': [],
            'n_features': 0
        }
        
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                # Numeric feature
                numeric_features.append(df[col].values)
                metadata['numeric_columns'].append(col)
            elif df[col].dtype == 'object' or df[col].nunique() < 50:
                # Categorical feature
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(df[col].fillna('missing'))
                else:
                    encoded = self.label_encoders[col].transform(df[col].fillna('missing'))
                
                # One-hot encode
                n_categories = len(self.label_encoders[col].classes_)
                one_hot = np.zeros((len(df), n_categories))
                one_hot[np.arange(len(df)), encoded] = 1
                numeric_features.append(one_hot)
                
                categorical_features[col] = encoded
                metadata['categorical_columns'].append(col)
        
        # Combine all features
        if numeric_features:
            feature_matrix = np.hstack(numeric_features)
        else:
            feature_matrix = np.zeros((len(df), 1))
        
        metadata['n_features'] = feature_matrix.shape[1]
        
        # Standardize
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix, metadata
    
    def train_autoencoder(self, df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None):
        """
        Train autoencoder on data.
        
        Args:
            df: Training DataFrame
            validation_df: Optional validation DataFrame
        """
        logger.info("Training deep feature autoencoder")
        
        # Prepare data
        X_train, metadata = self.prepare_data(df)
        
        if validation_df is not None:
            X_val, _ = self.prepare_data(validation_df)
        else:
            # Split training data
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            X_train = X_train[:split_idx]
        
        # Update config with actual input dimension
        self.config.input_dim = X_train.shape[1]
        
        # Create model
        if self.config.use_variational:
            self.model = VariationalAutoencoder(self.config)
        else:
            # Use standard autoencoder (not implemented here, but similar structure)
            self.model = VariationalAutoencoder(self.config)
        
        self.model.to(self.config.device)
        
        # Create data loaders
        train_dataset = RecordDataset(X_train, self.config.use_denoising, self.config.noise_factor)
        val_dataset = RecordDataset(X_val, False, 0)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_input, batch_target in train_loader:
                batch_input = batch_input.to(self.config.device)
                batch_target = batch_target.to(self.config.device)
                
                # Forward pass
                if self.config.use_variational:
                    recon, mu, logvar, _ = self.model(batch_input)
                    
                    # VAE loss
                    recon_loss = F.mse_loss(recon, batch_target, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.config.beta_vae * kl_loss
                else:
                    recon = self.model(batch_input)
                    loss = F.mse_loss(recon, batch_target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_input, batch_target in val_loader:
                    batch_input = batch_input.to(self.config.device)
                    batch_target = batch_target.to(self.config.device)
                    
                    if self.config.use_variational:
                        recon, mu, logvar, _ = self.model(batch_input)
                        recon_loss = F.mse_loss(recon, batch_target, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + self.config.beta_vae * kl_loss
                    else:
                        recon = self.model(batch_input)
                        loss = F.mse_loss(recon, batch_target)
                    
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_trained = True
        
        # Extract feature importance if using attention
        if self.config.use_attention:
            self._extract_feature_importance(X_train)
        
        logger.info("Autoencoder training completed")
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract learned features from data.
        
        Args:
            df: DataFrame to extract features from
            
        Returns:
            Learned feature representations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_autoencoder first.")
        
        # Prepare data
        X, _ = self.prepare_data(df)
        X_tensor = torch.FloatTensor(X).to(self.config.device)
        
        # Extract embeddings
        self.model.eval()
        with torch.no_grad():
            if self.config.use_variational:
                embeddings = self.model.get_embedding(X_tensor)
            else:
                # For standard autoencoder, use encoder output
                embeddings = self.model.encoder(X_tensor)
        
        return embeddings.cpu().numpy()
    
    def _extract_feature_importance(self, X_train: np.ndarray):
        """Extract feature importance from attention weights."""
        if not self.config.use_attention:
            return
        
        X_tensor = torch.FloatTensor(X_train[:1000]).to(self.config.device)  # Sample
        
        self.model.eval()
        with torch.no_grad():
            _, _, _, attention_weights = self.model(X_tensor)
            
            if attention_weights is not None:
                # Average attention weights
                avg_attention = attention_weights.mean(dim=0).cpu().numpy()
                self.feature_importance = avg_attention
                
                logger.info(f"Extracted feature importance from attention: "
                          f"top features {np.argsort(avg_attention)[-5:]}")
    
    def learn_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Learn and create feature interactions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional interaction features
        """
        logger.info("Learning feature interactions")
        
        # Extract learned representations
        embeddings = self.extract_features(df)
        
        # Use gradient boosting to identify important interactions
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        
        # Create polynomial features from embeddings
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction_features = poly.fit_transform(embeddings[:, :10])  # Use top 10 dimensions
        
        # Add interaction features to dataframe
        interaction_df = pd.DataFrame(
            interaction_features,
            columns=[f'interaction_{i}' for i in range(interaction_features.shape[1])],
            index=df.index
        )
        
        # Combine with original features
        enhanced_df = pd.concat([df, interaction_df], axis=1)
        
        logger.info(f"Added {interaction_features.shape[1]} interaction features")
        
        return enhanced_df
    
    def save_model(self, path: str):
        """Save trained model and preprocessing objects."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessing objects."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.config = save_dict['config']
        self.scaler = save_dict['scaler']
        self.label_encoders = save_dict['label_encoders']
        self.feature_importance = save_dict['feature_importance']
        
        # Recreate model
        if self.config.use_variational:
            self.model = VariationalAutoencoder(self.config)
        else:
            self.model = VariationalAutoencoder(self.config)
        
        self.model.load_state_dict(save_dict['model_state'])
        self.model.to(self.config.device)
        self.model.eval()
        
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_similarity(self, record1: pd.Series, record2: pd.Series) -> float:
        """
        Calculate similarity between two records in learned feature space.
        
        Args:
            record1: First record
            record2: Second record
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrames
        df1 = pd.DataFrame([record1])
        df2 = pd.DataFrame([record2])
        
        # Extract embeddings
        emb1 = self.extract_features(df1)[0]
        emb2 = self.extract_features(df2)[0]
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return float(similarity)