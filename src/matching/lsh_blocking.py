"""
Locality Sensitive Hashing (LSH) for efficient blocking.

This module implements advanced blocking strategies using LSH for:
- Efficient similarity-based candidate generation
- Scalable blocking for millions of records
- Multiple hash functions for different data types
- Adaptive blocking with dynamic thresholds
- Ensemble blocking combining multiple LSH schemes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from collections import defaultdict
import hashlib
from scipy.spatial.distance import cosine, jaccard
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_extraction.text import HashingVectorizer
try:
    import mmh3  # MurmurHash3 for fast hashing
    _HAS_MMH3 = True
except Exception:
    mmh3 = None
    _HAS_MMH3 = False
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class LSHConfig:
    """Configuration for LSH blocking."""
    n_hash_functions: int = 128
    n_bands: int = 32
    n_rows_per_band: int = 4  # n_hash_functions = n_bands * n_rows_per_band
    similarity_threshold: float = 0.5
    hash_type: str = 'minhash'  # 'minhash', 'simhash', 'random_projection'
    vector_dim: int = 100
    use_ensemble: bool = True
    ensemble_methods: List[str] = None
    max_candidates_per_record: int = 1000
    min_candidates_per_record: int = 10
    adaptive_threshold: bool = True
    use_canopy_clustering: bool = False
    canopy_t1: float = 0.9
    canopy_t2: float = 0.7
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['minhash', 'simhash', 'random_projection']
        # Ensure consistency
        self.n_hash_functions = self.n_bands * self.n_rows_per_band


class MinHashLSH:
    """
    MinHash-based LSH for Jaccard similarity.
    
    Efficient for set-based data (e.g., tokens, categories).
    """
    
    def __init__(self, config: LSHConfig):
        """
        Initialize MinHash LSH.
        
        Args:
            config: LSH configuration
        """
        self.config = config
        self.hash_functions = []
        self.buckets = defaultdict(lambda: defaultdict(set))
        
        # Generate hash function parameters
        self._initialize_hash_functions()
        
    def _initialize_hash_functions(self):
        """Initialize hash functions for MinHash."""
        # Use different seeds for independent hash functions
        self.hash_seeds = np.random.randint(0, 2**32, size=self.config.n_hash_functions)
        
    def _minhash_signature(self, tokens: Set[str]) -> np.ndarray:
        """
        Generate MinHash signature for a set of tokens.
        
        Args:
            tokens: Set of tokens
            
        Returns:
            MinHash signature vector
        """
        if not tokens:
            return np.zeros(self.config.n_hash_functions)
            
        signature = np.zeros(self.config.n_hash_functions, dtype=np.uint32)
        
        for i, seed in enumerate(self.hash_seeds):
            min_hash = float('inf')
            for token in tokens:
                # Hash token with seed
                if _HAS_MMH3:
                    hash_val = mmh3.hash(token, seed, signed=False)
                else:
                    # Fallback: use Python's hashlib (slower but available)
                    # Combine token and seed deterministically
                    h = hashlib.blake2b(digest_size=8)
                    h.update(str(seed).encode('utf-8'))
                    h.update(token.encode('utf-8', errors='ignore'))
                    hash_val = int.from_bytes(h.digest(), byteorder='little', signed=False)
                min_hash = min(min_hash, hash_val)
            signature[i] = min_hash
            
        return signature
    
    def _banding(self, signature: np.ndarray) -> List[str]:
        """
        Apply banding to signature for LSH.
        
        Args:
            signature: MinHash signature
            
        Returns:
            List of band hashes
        """
        band_hashes = []
        rows_per_band = self.config.n_rows_per_band
        
        for band_idx in range(self.config.n_bands):
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band
            band = signature[start_idx:end_idx]
            
            # Hash the band to get bucket key
            band_hash = hashlib.md5(band.tobytes()).hexdigest()
            band_hashes.append(f"band_{band_idx}_{band_hash}")
            
        return band_hashes
    
    def index(self, records: pd.DataFrame, id_column: str, text_columns: List[str]):
        """
        Index records for LSH blocking.
        
        Args:
            records: DataFrame with records to index
            id_column: Column name for record IDs
            text_columns: List of text columns to use for blocking
        """
        logger.info(f"Indexing {len(records)} records with MinHash LSH")
        
        for idx, row in records.iterrows():
            # Extract tokens from text columns
            tokens = set()
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    # Tokenize and add to set
                    text = str(row[col]).lower()
                    tokens.update(text.split())
            
            if not tokens:
                continue
                
            # Generate signature
            signature = self._minhash_signature(tokens)
            
            # Apply banding
            band_hashes = self._banding(signature)
            
            # Store in buckets
            record_id = row[id_column] if id_column in row else idx
            for band_hash in band_hashes:
                self.buckets[band_hash]['records'].add(record_id)
                
        logger.info(f"Created {len(self.buckets)} LSH buckets")
    
    def query(self, record: pd.Series, text_columns: List[str], 
              max_candidates: Optional[int] = None) -> Set[Any]:
        """
        Query for similar records.
        
        Args:
            record: Record to query
            text_columns: Text columns to use
            max_candidates: Maximum number of candidates to return
            
        Returns:
            Set of candidate record IDs
        """
        # Extract tokens
        tokens = set()
        for col in text_columns:
            if col in record and pd.notna(record[col]):
                text = str(record[col]).lower()
                tokens.update(text.split())
        
        if not tokens:
            return set()
            
        # Generate signature
        signature = self._minhash_signature(tokens)
        
        # Apply banding
        band_hashes = self._banding(signature)
        
        # Collect candidates from matching buckets
        candidates = set()
        for band_hash in band_hashes:
            if band_hash in self.buckets:
                candidates.update(self.buckets[band_hash]['records'])
        
        # Limit candidates if specified
        if max_candidates and len(candidates) > max_candidates:
            # Random sampling for now (could rank by similarity)
            candidates = set(np.random.choice(list(candidates), max_candidates, replace=False))
            
        return candidates


class SimHashLSH:
    """
    SimHash-based LSH for cosine similarity.
    
    Efficient for high-dimensional sparse vectors.
    """
    
    def __init__(self, config: LSHConfig):
        """
        Initialize SimHash LSH.
        
        Args:
            config: LSH configuration
        """
        self.config = config
        self.random_hyperplanes = None
        self.buckets = defaultdict(lambda: defaultdict(set))
        
        # Initialize random hyperplanes
        self._initialize_hyperplanes()
        
    def _initialize_hyperplanes(self):
        """Initialize random hyperplanes for SimHash."""
        # Random unit vectors as hyperplanes
        self.random_hyperplanes = np.random.randn(self.config.n_hash_functions, 
                                                  self.config.vector_dim)
        # Normalize to unit vectors
        norms = np.linalg.norm(self.random_hyperplanes, axis=1, keepdims=True)
        self.random_hyperplanes /= norms
        
    def _simhash_signature(self, vector: np.ndarray) -> np.ndarray:
        """
        Generate SimHash signature for a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Binary SimHash signature
        """
        if len(vector) != self.config.vector_dim:
            # Pad or truncate
            if len(vector) < self.config.vector_dim:
                vector = np.pad(vector, (0, self.config.vector_dim - len(vector)))
            else:
                vector = vector[:self.config.vector_dim]
                
        # Project onto hyperplanes
        projections = np.dot(self.random_hyperplanes, vector)
        
        # Convert to binary (1 if positive, 0 if negative)
        signature = (projections >= 0).astype(int)
        
        return signature
    
    def _hamming_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> int:
        """Calculate Hamming distance between two signatures."""
        return np.sum(sig1 != sig2)
    
    def index(self, records: pd.DataFrame, id_column: str, vector_columns: List[str]):
        """
        Index records using SimHash.
        
        Args:
            records: DataFrame with records
            id_column: ID column name
            vector_columns: Columns to use for vector construction
        """
        logger.info(f"Indexing {len(records)} records with SimHash LSH")
        
        for idx, row in records.iterrows():
            # Construct vector from specified columns
            vector = []
            for col in vector_columns:
                if col in row and pd.notna(row[col]):
                    if isinstance(row[col], (int, float)):
                        vector.append(float(row[col]))
                    else:
                        # Hash categorical values
                        vector.append(float(hash(str(row[col])) % 1000) / 1000)
            
            if not vector:
                continue
                
            vector = np.array(vector)
            
            # Generate signature
            signature = self._simhash_signature(vector)
            
            # Convert to hash for bucketing
            signature_hash = hashlib.md5(signature.tobytes()).hexdigest()
            
            record_id = row[id_column] if id_column in row else idx
            self.buckets[signature_hash]['records'].add(record_id)
            self.buckets[signature_hash]['signature'] = signature
            
        logger.info(f"Created {len(self.buckets)} SimHash buckets")


class RandomProjectionLSH:
    """
    Random Projection LSH for Euclidean distance.
    
    Uses random projections to create hash functions.
    """
    
    def __init__(self, config: LSHConfig):
        """
        Initialize Random Projection LSH.
        
        Args:
            config: LSH configuration
        """
        self.config = config
        self.projector = None
        self.buckets = defaultdict(lambda: defaultdict(set))
        self.thresholds = None
        
        # Initialize projector
        self._initialize_projector()
        
    def _initialize_projector(self):
        """Initialize random projection matrix."""
        self.projector = GaussianRandomProjection(
            n_components=self.config.n_hash_functions,
            random_state=42
        )
        
        # Generate random thresholds for discretization
        self.thresholds = np.random.randn(self.config.n_hash_functions)
        
    def _projection_hash(self, vector: np.ndarray) -> str:
        """
        Generate hash using random projection.
        
        Args:
            vector: Input vector
            
        Returns:
            Hash string
        """
        # Project vector
        if not hasattr(self.projector, 'components_'):
            # Fit projector if not fitted
            self.projector.fit([vector])
            
        projected = self.projector.transform([vector])[0]
        
        # Discretize using thresholds
        binary = (projected > self.thresholds).astype(int)
        
        # Convert to hash
        return hashlib.md5(binary.tobytes()).hexdigest()
    
    def index(self, records: pd.DataFrame, id_column: str, feature_columns: List[str]):
        """
        Index records using random projection.
        
        Args:
            records: DataFrame with records
            id_column: ID column name
            feature_columns: Feature columns to use
        """
        logger.info(f"Indexing {len(records)} records with Random Projection LSH")
        
        # Extract feature matrix
        features = []
        record_ids = []
        
        for idx, row in records.iterrows():
            feature_vec = []
            for col in feature_columns:
                if col in row and pd.notna(row[col]):
                    if isinstance(row[col], (int, float)):
                        feature_vec.append(float(row[col]))
                    else:
                        # Simple hash for categorical
                        feature_vec.append(float(hash(str(row[col])) % 1000) / 1000)
                else:
                    feature_vec.append(0.0)
            
            if feature_vec:
                features.append(feature_vec)
                record_ids.append(row[id_column] if id_column in row else idx)
        
        if not features:
            return
            
        features = np.array(features)
        
        # Fit projector if needed
        if not hasattr(self.projector, 'components_'):
            self.projector.fit(features)
        
        # Project all features
        projected = self.projector.transform(features)
        
        # Discretize and bucket
        for i, (proj, record_id) in enumerate(zip(projected, record_ids)):
            binary = (proj > self.thresholds).astype(int)
            hash_key = hashlib.md5(binary.tobytes()).hexdigest()
            self.buckets[hash_key]['records'].add(record_id)
            
        logger.info(f"Created {len(self.buckets)} Random Projection buckets")


class CanopyBlocking:
    """
    Canopy clustering for adaptive blocking.
    
    Creates overlapping canopies based on cheap distance metric.
    """
    
    def __init__(self, t1: float = 0.9, t2: float = 0.7):
        """
        Initialize canopy blocking.
        
        Args:
            t1: Loose distance threshold
            t2: Tight distance threshold (t2 < t1)
        """
        self.t1 = t1
        self.t2 = t2
        self.canopies = []
        self.record_to_canopies = defaultdict(list)
        
    def create_canopies(self, records: pd.DataFrame, distance_func, sample_size: int = 1000):
        """
        Create canopies from records.
        
        Args:
            records: DataFrame with records
            distance_func: Distance function to use
            sample_size: Sample size for canopy centers
        """
        logger.info(f"Creating canopies from {len(records)} records")
        
        # Sample potential canopy centers
        if len(records) > sample_size:
            centers = records.sample(n=sample_size, random_state=42)
        else:
            centers = records
        
        # Create canopies
        remaining = set(records.index)
        
        for center_idx in centers.index:
            if center_idx not in remaining:
                continue
                
            # Create new canopy
            canopy = {'center': center_idx, 'members': set()}
            
            # Find members within t1 distance
            to_remove = set()
            for idx in remaining:
                dist = distance_func(records.loc[center_idx], records.loc[idx])
                
                if dist < self.t1:
                    canopy['members'].add(idx)
                    self.record_to_canopies[idx].append(len(self.canopies))
                    
                    # Remove if within t2 distance
                    if dist < self.t2:
                        to_remove.add(idx)
            
            self.canopies.append(canopy)
            remaining -= to_remove
            
            if not remaining:
                break
                
        logger.info(f"Created {len(self.canopies)} canopies")
    
    def get_candidates(self, record_idx: int) -> Set[int]:
        """
        Get candidate records from same canopies.
        
        Args:
            record_idx: Record index
            
        Returns:
            Set of candidate indices
        """
        candidates = set()
        
        for canopy_idx in self.record_to_canopies[record_idx]:
            candidates.update(self.canopies[canopy_idx]['members'])
            
        # Remove self
        candidates.discard(record_idx)
        
        return candidates


class EnsembleLSHBlocker:
    """
    Ensemble LSH blocking combining multiple methods.
    
    Provides robust blocking by combining different LSH schemes.
    """
    
    def __init__(self, config: Optional[LSHConfig] = None):
        """
        Initialize ensemble LSH blocker.
        
        Args:
            config: LSH configuration
        """
        self.config = config or LSHConfig()
        self.blockers = {}
        self.is_indexed = False
        
        # Initialize individual blockers
        self._initialize_blockers()
        
    def _initialize_blockers(self):
        """Initialize individual LSH blockers."""
        if 'minhash' in self.config.ensemble_methods:
            self.blockers['minhash'] = MinHashLSH(self.config)
            
        if 'simhash' in self.config.ensemble_methods:
            self.blockers['simhash'] = SimHashLSH(self.config)
            
        if 'random_projection' in self.config.ensemble_methods:
            self.blockers['random_projection'] = RandomProjectionLSH(self.config)
            
        if self.config.use_canopy_clustering:
            self.blockers['canopy'] = CanopyBlocking(
                t1=self.config.canopy_t1,
                t2=self.config.canopy_t2
            )
            
        logger.info(f"Initialized {len(self.blockers)} LSH blockers")
    
    def index(self, df1: pd.DataFrame, df2: pd.DataFrame, 
              text_columns: List[str] = None,
              numeric_columns: List[str] = None) -> None:
        """
        Index both datasets for blocking.
        
        Args:
            df1: First dataset (e.g., PUMS)
            df2: Second dataset (e.g., RECS)
            text_columns: Columns with text data
            numeric_columns: Columns with numeric data
        """
        start_time = time.time()
        logger.info(f"Indexing {len(df1)} x {len(df2)} records with ensemble LSH")
        
        # Prepare column lists
        if text_columns is None:
            text_columns = [col for col in df1.columns if df1[col].dtype == 'object']
        if numeric_columns is None:
            numeric_columns = [col for col in df1.columns if df1[col].dtype in ['int64', 'float64']]
        
        # Index with different blockers
        if 'minhash' in self.blockers and text_columns:
            self.blockers['minhash'].index(df1, 'index', text_columns)
            self.blockers['minhash'].index(df2, 'index', text_columns)
            
        if 'simhash' in self.blockers and numeric_columns:
            self.blockers['simhash'].index(df1, 'index', numeric_columns)
            self.blockers['simhash'].index(df2, 'index', numeric_columns)
            
        if 'random_projection' in self.blockers:
            all_columns = text_columns + numeric_columns
            if all_columns:
                self.blockers['random_projection'].index(df1, 'index', all_columns)
                self.blockers['random_projection'].index(df2, 'index', all_columns)
        
        self.is_indexed = True
        elapsed = time.time() - start_time
        logger.info(f"LSH indexing completed in {elapsed:.2f} seconds")
    
    def generate_candidate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                text_columns: List[str] = None,
                                numeric_columns: List[str] = None) -> pd.DataFrame:
        """
        Generate candidate pairs using ensemble blocking.
        
        Args:
            df1: First dataset
            df2: Second dataset
            text_columns: Text columns for blocking
            numeric_columns: Numeric columns for blocking
            
        Returns:
            DataFrame with candidate pairs (idx1, idx2)
        """
        if not self.is_indexed:
            self.index(df1, df2, text_columns, numeric_columns)
        
        logger.info("Generating candidate pairs from ensemble LSH")
        
        all_pairs = set()
        blocker_pairs = {}
        
        # Collect pairs from each blocker
        for blocker_name, blocker in self.blockers.items():
            if blocker_name == 'canopy':
                continue  # Handle separately
                
            pairs = set()
            
            # Query each record in df1 against df2
            for idx1 in df1.index:
                if blocker_name == 'minhash' and text_columns:
                    candidates = blocker.query(df1.loc[idx1], text_columns, 
                                             self.config.max_candidates_per_record)
                elif blocker_name in ['simhash', 'random_projection']:
                    # Simple bucket matching for now
                    candidates = self._get_bucket_candidates(blocker, idx1, df2.index)
                else:
                    candidates = set()
                
                for idx2 in candidates:
                    if idx2 in df2.index:
                        pairs.add((idx1, idx2))
            
            blocker_pairs[blocker_name] = pairs
            all_pairs.update(pairs)
            logger.info(f"{blocker_name}: {len(pairs)} candidate pairs")
        
        # Apply adaptive thresholding if needed
        if self.config.adaptive_threshold:
            all_pairs = self._apply_adaptive_threshold(all_pairs, df1, df2)
        
        # Ensure minimum candidates per record
        all_pairs = self._ensure_minimum_candidates(all_pairs, df1, df2)
        
        # Convert to DataFrame
        if all_pairs:
            pairs_df = pd.DataFrame(list(all_pairs), columns=['idx1', 'idx2'])
        else:
            pairs_df = pd.DataFrame(columns=['idx1', 'idx2'])
            
        logger.info(f"Generated {len(pairs_df)} total candidate pairs")
        
        # Calculate blocking statistics
        self._calculate_blocking_statistics(pairs_df, df1, df2, blocker_pairs)
        
        return pairs_df
    
    def _get_bucket_candidates(self, blocker, query_idx, candidate_indices) -> Set:
        """Get candidates from same buckets."""
        candidates = set()
        
        # Find which bucket the query record is in
        for bucket_key, bucket_data in blocker.buckets.items():
            if query_idx in bucket_data['records']:
                # Add all records from this bucket
                candidates.update(bucket_data['records'] & set(candidate_indices))
                
        return candidates
    
    def _apply_adaptive_threshold(self, pairs: Set[Tuple], df1: pd.DataFrame, 
                                 df2: pd.DataFrame) -> Set[Tuple]:
        """Apply adaptive thresholding to candidate pairs."""
        # Keep pairs that appear in multiple blockers
        pair_counts = defaultdict(int)
        
        # Count would be done if we tracked per-blocker pairs
        # For now, return as-is
        return pairs
    
    def _ensure_minimum_candidates(self, pairs: Set[Tuple], df1: pd.DataFrame,
                                  df2: pd.DataFrame) -> Set[Tuple]:
        """Ensure each record has minimum number of candidates."""
        pairs_by_record = defaultdict(list)
        
        for idx1, idx2 in pairs:
            pairs_by_record[idx1].append(idx2)
        
        # Check each record in df1
        for idx1 in df1.index:
            candidates = pairs_by_record.get(idx1, [])
            
            if len(candidates) < self.config.min_candidates_per_record:
                # Add random candidates to meet minimum
                needed = self.config.min_candidates_per_record - len(candidates)
                available = set(df2.index) - set(candidates)
                
                if available:
                    additional = np.random.choice(list(available), 
                                                min(needed, len(available)), 
                                                replace=False)
                    for idx2 in additional:
                        pairs.add((idx1, idx2))
                        
        return pairs
    
    def _calculate_blocking_statistics(self, pairs_df: pd.DataFrame, df1: pd.DataFrame,
                                      df2: pd.DataFrame, blocker_pairs: Dict):
        """Calculate and log blocking statistics."""
        total_possible = len(df1) * len(df2)
        reduction_ratio = 1 - (len(pairs_df) / total_possible) if total_possible > 0 else 0
        
        # Coverage statistics
        covered_df1 = pairs_df['idx1'].nunique() if len(pairs_df) > 0 else 0
        covered_df2 = pairs_df['idx2'].nunique() if len(pairs_df) > 0 else 0
        
        coverage_df1 = covered_df1 / len(df1) if len(df1) > 0 else 0
        coverage_df2 = covered_df2 / len(df2) if len(df2) > 0 else 0
        
        # Pairs per record statistics
        if len(pairs_df) > 0:
            pairs_per_record = pairs_df.groupby('idx1').size()
            avg_pairs = pairs_per_record.mean()
            max_pairs = pairs_per_record.max()
            min_pairs = pairs_per_record.min()
        else:
            avg_pairs = max_pairs = min_pairs = 0
        
        logger.info(f"""
        LSH Blocking Statistics:
        - Reduction Ratio: {reduction_ratio:.4f}
        - Coverage df1: {coverage_df1:.2%} ({covered_df1}/{len(df1)})
        - Coverage df2: {coverage_df2:.2%} ({covered_df2}/{len(df2)})
        - Avg candidates per record: {avg_pairs:.1f}
        - Max candidates: {max_pairs}
        - Min candidates: {min_pairs}
        """)
        
        # Per-blocker statistics
        for blocker_name, pairs in blocker_pairs.items():
            logger.info(f"  {blocker_name}: {len(pairs)} pairs, "
                       f"overlap with ensemble: {len(pairs & set(map(tuple, pairs_df.values)))}")


class LearnedBlocking:
    """
    Machine learning based blocking that learns which record pairs to compare.
    
    Uses a classifier to predict whether two records should be compared.
    """
    
    def __init__(self, base_blocker: EnsembleLSHBlocker = None):
        """
        Initialize learned blocking.
        
        Args:
            base_blocker: Base LSH blocker to use for initial candidates
        """
        self.base_blocker = base_blocker or EnsembleLSHBlocker()
        self.blocking_classifier = None
        self.feature_columns = None
        
    def train(self, training_pairs: pd.DataFrame, labels: np.ndarray,
             df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Train blocking classifier.
        
        Args:
            training_pairs: DataFrame with training pairs (idx1, idx2)
            labels: Labels (1 if should compare, 0 if not)
            df1: First dataset
            df2: Second dataset
        """
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Training learned blocking classifier")
        
        # Extract features for training pairs
        features = []
        for _, row in training_pairs.iterrows():
            idx1, idx2 = row['idx1'], row['idx2']
            pair_features = self._extract_blocking_features(
                df1.loc[idx1], df2.loc[idx2]
            )
            features.append(pair_features)
        
        features = np.array(features)
        
        # Train classifier
        self.blocking_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.blocking_classifier.fit(features, labels)
        
        # Get feature importance
        importance = self.blocking_classifier.feature_importances_
        logger.info(f"Top blocking features: {np.argsort(importance)[-5:]}")
    
    def _extract_blocking_features(self, record1: pd.Series, record2: pd.Series) -> np.ndarray:
        """Extract features for blocking decision."""
        features = []
        
        # Simple features for blocking decision
        # Geographic agreement
        if 'STATE' in record1 and 'STATE' in record2:
            features.append(1 if record1['STATE'] == record2['STATE'] else 0)
        else:
            features.append(0)
            
        if 'REGION' in record1 and 'REGION' in record2:
            features.append(1 if record1['REGION'] == record2['REGION'] else 0)
        else:
            features.append(0)
            
        # Size similarity
        if 'household_size' in record1 and 'household_size' in record2:
            size_diff = abs(record1['household_size'] - record2['household_size'])
            features.append(1.0 / (1 + size_diff))
        else:
            features.append(0.5)
            
        # Income similarity
        if 'income_quintile' in record1 and 'income_quintile' in record2:
            features.append(1 if record1['income_quintile'] == record2['income_quintile'] else 0)
        else:
            features.append(0)
            
        # Pad to fixed size
        while len(features) < 10:
            features.append(0)
            
        return np.array(features[:10])
    
    def predict_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame,
                     initial_candidates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict which pairs should be compared.
        
        Args:
            df1: First dataset
            df2: Second dataset
            initial_candidates: Optional initial candidate pairs
            
        Returns:
            DataFrame with predicted candidate pairs
        """
        if self.blocking_classifier is None:
            logger.warning("Blocking classifier not trained, using base blocker")
            return self.base_blocker.generate_candidate_pairs(df1, df2)
        
        # Get initial candidates if not provided
        if initial_candidates is None:
            initial_candidates = self.base_blocker.generate_candidate_pairs(df1, df2)
        
        # Predict which candidates to keep
        features = []
        for _, row in initial_candidates.iterrows():
            idx1, idx2 = row['idx1'], row['idx2']
            pair_features = self._extract_blocking_features(
                df1.loc[idx1], df2.loc[idx2]
            )
            features.append(pair_features)
        
        features = np.array(features)
        predictions = self.blocking_classifier.predict_proba(features)[:, 1]
        
        # Keep pairs with high probability
        threshold = 0.3  # Lower threshold to avoid missing matches
        keep_mask = predictions >= threshold
        
        filtered_pairs = initial_candidates[keep_mask].copy()
        
        logger.info(f"Learned blocking: {len(initial_candidates)} -> {len(filtered_pairs)} pairs")
        
        return filtered_pairs