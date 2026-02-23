"""
Performance Optimization Utilities for Large-Scale Processing.

This module provides optimizations for processing 1.4 million buildings efficiently:
- Parallel processing with multiprocessing
- Chunked data processing to manage memory
- Memory-optimized data types
- Progress tracking
- Checkpoint/resume capability
"""

import os
import pickle
import json
import logging
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Main class for performance optimization."""
    
    def __init__(self, config: Dict = None):
        """Initialize performance optimizer with system detection."""
        self.config = config or {}

        # Detect system capabilities
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Set optimal parameters based on system
        # Default: use all CPUs unless overridden
        default_workers = max(1, self.cpu_count)
        self.n_workers = int(self.config.get('n_workers', default_workers))

        # Derive chunk size and GPU capability
        self.chunk_size = int(self.config.get('chunk_size', self._calculate_optimal_chunk_size()))
        self.use_gpu = bool(self.config.get('use_gpu', self._detect_gpu()))

        # Checkpoint settings
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = int(self.config.get('checkpoint_frequency', 1000))

        logger.info(f"System detected: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
        logger.info(f"Optimization settings: {self.n_workers} workers, chunk_size={self.chunk_size}")
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Estimate ~1MB per building with all data
        mb_per_building = 1.0
        
        # Use 50% of available memory for processing
        available_mb = (self.available_memory_gb * 1024) * 0.5
        
        # Calculate chunk size (buildings per chunk)
        workers = max(1, getattr(self, 'n_workers', 1))
        chunk_size = int(available_mb / mb_per_building / workers)

        # Ensure reasonable bounds (allow larger upper bound for high-RAM systems)
        chunk_size = max(100, min(chunk_size, 32000))

        return chunk_size
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available for acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to efficient dtypes.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame with reduced memory footprint
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize numeric columns
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Integer optimization
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                # Float optimization
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            
            # Optimize object columns
            else:
                # Convert strings to categories if low cardinality
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction_pct = 100 * (initial_memory - final_memory) / initial_memory
        
        logger.info(f"Memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction_pct:.1f}% reduction)")
        
        return df
    
    def parallel_process_chunks(self, 
                              data: pd.DataFrame,
                              process_func: Callable,
                              desc: str = "Processing",
                              **kwargs) -> pd.DataFrame:
        """
        Process DataFrame in parallel chunks.
        
        Args:
            data: Input DataFrame
            process_func: Function to apply to each chunk
            desc: Description for progress bar
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed DataFrame
        """
        n_rows = len(data)
        n_chunks = max(1, n_rows // self.chunk_size)
        
        # Split data into chunks
        chunks = np.array_split(data, n_chunks)
        
        logger.info(f"Processing {n_rows} rows in {n_chunks} chunks using {self.n_workers} workers")
        
        # Process chunks in parallel with progress bar
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_func, chunk, **kwargs): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=n_chunks, desc=desc) as pbar:
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        result = future.result()
                        results.append((chunk_idx, result))
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx}: {e}")
                        results.append((chunk_idx, None))
                        pbar.update(1)
        
        # Sort results by original chunk order and combine
        results.sort(key=lambda x: x[0])
        processed_chunks = [r[1] for r in results if r[1] is not None]
        
        if processed_chunks:
            return pd.concat(processed_chunks, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def create_checkpoint(self, data: Any, phase: str, iteration: int):
        """
        Create a checkpoint for resumable processing.
        
        Args:
            data: Data to checkpoint
            phase: Current phase name
            iteration: Current iteration/batch number
        """
        checkpoint_file = self.checkpoint_dir / f"{phase}_checkpoint_{iteration}.pkl"
        metadata_file = self.checkpoint_dir / f"{phase}_checkpoint_metadata.json"
        
        # Save data
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        metadata = {
            'phase': phase,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'data_file': str(checkpoint_file)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint created: {checkpoint_file}")
    
    def load_latest_checkpoint(self, phase: str) -> Tuple[Optional[Any], Optional[int]]:
        """
        Load the latest checkpoint for a phase.
        
        Args:
            phase: Phase name to load checkpoint for
            
        Returns:
            Tuple of (data, iteration) or (None, None) if no checkpoint
        """
        metadata_file = self.checkpoint_dir / f"{phase}_checkpoint_metadata.json"
        
        if not metadata_file.exists():
            return None, None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        checkpoint_file = Path(metadata['data_file'])
        if not checkpoint_file.exists():
            return None, None
        
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded checkpoint from iteration {metadata['iteration']}")
        return data, metadata['iteration']
    
    def cleanup_checkpoints(self, phase: str):
        """Remove checkpoint files for a completed phase."""
        for checkpoint_file in self.checkpoint_dir.glob(f"{phase}_checkpoint_*"):
            checkpoint_file.unlink()
        logger.info(f"Cleaned up checkpoints for {phase}")


class ParallelMatcher:
    """Parallel implementation of matching algorithms."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        """Initialize parallel matcher with optimizer."""
        self.optimizer = optimizer
        self.n_workers = optimizer.n_workers
    
    def parallel_blocking(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         blocking_keys: List[str]) -> List[Tuple[int, int]]:
        """
        Perform blocking in parallel to generate candidate pairs.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            blocking_keys: Columns to use for blocking
            
        Returns:
            List of candidate pairs (idx1, idx2)
        """
        # Create blocks
        blocks = {}
        for key in blocking_keys:
            if key in df1.columns and key in df2.columns:
                # Group indices by blocking key
                for idx1, val1 in df1[key].items():
                    if pd.notna(val1):
                        if val1 not in blocks:
                            blocks[val1] = {'df1': [], 'df2': []}
                        blocks[val1]['df1'].append(idx1)
                
                for idx2, val2 in df2[key].items():
                    if pd.notna(val2):
                        if val2 in blocks:
                            blocks[val2]['df2'].append(idx2)
        
        # Generate pairs in parallel
        def generate_pairs_for_block(block_data):
            pairs = []
            for idx1 in block_data['df1']:
                for idx2 in block_data['df2']:
                    pairs.append((idx1, idx2))
            return pairs
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(generate_pairs_for_block, block) 
                      for block in blocks.values()]
            
            all_pairs = []
            for future in as_completed(futures):
                pairs = future.result()
                all_pairs.extend(pairs)
        
        return all_pairs
    
    def parallel_similarity_computation(self, pairs: List[Tuple[int, int]], 
                                       df1: pd.DataFrame, df2: pd.DataFrame,
                                       comparison_func: Callable) -> np.ndarray:
        """
        Compute similarity scores in parallel.
        
        Args:
            pairs: List of record pairs to compare
            df1: First DataFrame
            df2: Second DataFrame
            comparison_func: Function to compute similarity
            
        Returns:
            Array of similarity scores
        """
        def compute_batch(batch_pairs):
            scores = []
            for idx1, idx2 in batch_pairs:
                record1 = df1.iloc[idx1]
                record2 = df2.iloc[idx2]
                score = comparison_func(record1, record2)
                scores.append(score)
            return scores
        
        # Split pairs into batches
        batch_size = max(1, len(pairs) // (self.n_workers * 10))
        batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
        
        # Compute in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(compute_batch, batch) for batch in batches]
            
            all_scores = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Computing similarities"):
                scores = future.result()
                all_scores.extend(scores)
        
        return np.array(all_scores)


class StreamProcessor:
    """Stream processing for extremely large datasets."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        """Initialize stream processor."""
        self.optimizer = optimizer
        self.buffer_size = optimizer.chunk_size
    
    def stream_process_file(self, filepath: Path, process_func: Callable,
                           output_file: Path, file_type: str = 'csv'):
        """
        Process large file in streaming fashion.
        
        Args:
            filepath: Input file path
            process_func: Function to apply to each chunk
            output_file: Output file path
            file_type: Type of file ('csv' or 'parquet')
        """
        logger.info(f"Stream processing {filepath}")
        
        # Process in chunks
        first_chunk = True
        total_rows = 0
        
        if file_type == 'csv':
            reader = pd.read_csv(filepath, chunksize=self.buffer_size)
        elif file_type == 'parquet':
            # For parquet, we'll read in chunks manually
            df = pd.read_parquet(filepath)
            n_chunks = len(df) // self.buffer_size + 1
            reader = [df[i*self.buffer_size:(i+1)*self.buffer_size] 
                     for i in range(n_chunks)]
        
        with tqdm(desc="Processing chunks") as pbar:
            for chunk in reader:
                # Optimize memory
                chunk = self.optimizer.optimize_datatypes(chunk)
                
                # Process chunk
                processed_chunk = process_func(chunk)
                
                # Write to output
                if first_chunk:
                    processed_chunk.to_parquet(output_file, index=False)
                    first_chunk = False
                else:
                    processed_chunk.to_parquet(output_file, index=False, 
                                              engine='fastparquet', append=True)
                
                total_rows += len(processed_chunk)
                pbar.update(1)
                pbar.set_postfix({'rows': total_rows})
        
        logger.info(f"Processed {total_rows} rows to {output_file}")


def create_optimized_pipeline(config: Dict) -> Dict:
    """
    Create an optimized pipeline configuration.
    
    Args:
        config: Base configuration
        
    Returns:
        Optimized configuration
    """
    optimizer = PerformanceOptimizer(config)
    
    optimized_config = config.copy()
    optimized_config.update({
        'n_workers': optimizer.n_workers,
        'chunk_size': optimizer.chunk_size,
        'use_parallel': True,
        'use_chunking': True,
        'optimize_memory': True,
        'checkpoint_enabled': True,
        'checkpoint_frequency': optimizer.checkpoint_frequency,
        'system_info': {
            'cpu_count': optimizer.cpu_count,
            'memory_gb': optimizer.memory_gb,
            'gpu_available': optimizer.use_gpu
        }
    })
    
    return optimized_config