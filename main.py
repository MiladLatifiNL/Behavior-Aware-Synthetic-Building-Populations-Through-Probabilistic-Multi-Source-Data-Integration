#!/usr/bin/env python3
"""
PUMS Enrichment Pipeline - Main Orchestrator

This script provides the command-line interface for running the 4-phase
building energy data integration pipeline.

Usage:
    python main.py --phase 1 --sample-size 100
    python main.py --phase all --full-data
    python main.py --validate-only
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import logging
import json
import pickle
import psutil
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Callable
import gc  # For garbage collection
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_config, ConfigurationError
from src.utils.logging_setup import setup_logging, create_performance_summary
from src.utils.memory_manager import MemoryManager, get_memory_manager, clear_dataframe_list
from src.processing.phase1_pums_integration import process_phase1, save_phase1_output
from src.validation.phase_validators import run_all_validations

# Module-level function for parallel processing (pickleable)
def process_phase1_chunk_wrapper(chunk):
    """Wrapper function for processing Phase 1 chunks in parallel.
    
    This function is at module level so it can be pickled for multiprocessing.
    """
    from src.processing.phase1_pums_integration import process_phase1_chunk
    from src.data_loading.pums_loader import PUMSDataLoader
    from src.utils.config_loader import get_config
    
    config = get_config()
    loader = PUMSDataLoader(config)
    return process_phase1_chunk(chunk, loader, config)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PUMS Enrichment Pipeline - Create synthetic building population with energy characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 with 100 buildings (development mode)
  python main.py --phase 1 --sample-size 100
  
  # Run Phase 1 with full data
  python main.py --phase 1 --full-data
  
  # Run all phases with full data (default for --phase all)
  python main.py --phase all
  
  # Validate existing output without reprocessing
  python main.py --validate-only
  
  # Run specific phase with custom sample size
  python main.py --phase 2 --sample-size 1000
        """
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '2', '3', '4', 'all'],
        default='1',
        help='Phase(s) to run (default: 1)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of buildings to process (overrides config default)'
    )
    
    parser.add_argument(
        '--full-data',
        action='store_true',
        help='Process full dataset (overrides sample-size)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation on existing output'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation after processing'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last successful phase'
    )

    parser.add_argument(
        '--from-phase',
        type=int,
        choices=[1,2,3,4],
        default=None,
        help='(With --resume) Force resume starting at this phase even if later phase outputs exist'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Performance optimization arguments
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (enabled by default)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
    default=None,
    help='Number of parallel workers (default: auto = all CPUs)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Chunk size for batch processing (default: auto-calculate)'
    )
    
    parser.add_argument(
        '--no-optimize-memory',
        action='store_true',
        help='Disable memory optimization (enabled by default)'
    )
    
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpointing (enabled by default)'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=None,
        help='Maximum memory usage in GB (default: auto = 90% of system RAM)'
    )
    
    # Streaming defaults to True; allow explicit disable via --no-streaming
    parser.add_argument(
        '--streaming',
        dest='streaming',
        action='store_true',
        help='Use streaming mode for minimal memory footprint (default: enabled)'
    )
    parser.add_argument(
        '--no-streaming',
        dest='streaming',
        action='store_false',
        help='Disable streaming mode'
    )
    parser.set_defaults(streaming=True)
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for streaming mode (default: auto-calculate)'
    )

    # Phase 2 specific knobs
    parser.add_argument(
        '--phase2-sub-batch',
        type=int,
        default=None,
        help='Override Phase 2 sub-batch size per shard (default: auto 500-2000)'
    )
    parser.add_argument(
        '--phase2-max-candidates',
        type=int,
        default=None,
        help='Override Phase 2 max candidates per PUMS record (default from config, e.g., 200)'
    )
    
    return parser.parse_args()


class PerformanceOptimizer:
    """Performance optimization utilities for large-scale processing."""
    
    def __init__(self, args):
        """Initialize optimizer with command-line arguments."""
        self.args = args
        # Hardware characteristics
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Memory manager (user limit may be None -> uses default percent inside manager)
        self.memory_manager = get_memory_manager(memory_limit_gb=args.memory_limit)

        # Feature flags (defaults enabled unless explicitly disabled by CLI)
        self.parallel = not args.no_parallel
        self.optimize_memory = not args.no_optimize_memory
        self.checkpoint = not args.no_checkpoint
        self.streaming = args.streaming

        # Streaming default is controlled by CLI; do not override explicit user choice.

        # Worker selection
        if args.workers is not None:
            self.n_workers = args.workers
        else:
            # Use all CPUs by default to maximize throughput when parallel is enabled
            self.n_workers = self.cpu_count if self.parallel else 1

        # Initialize chunk size placeholder & run calibration to set it
        self.chunk_size = 0
        self.measured_memory_per_building_mb = None
        self._calibrate_and_set_chunk_size(user_batch_size=args.batch_size)

        # Checkpointing
        self.checkpoint_dir = Path('data/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = 1000

        # GPU detection
        self.device, self.gpu_properties = self._detect_device()
        self.use_gpu = self.device == 'cuda'

        # Configure threading / BLAS
        self._configure_threading()

        # Logging summary
        logger = logging.getLogger('pums_enrichment.main')
        logger.info("Performance Optimizer initialized:")
        logger.info(f"  System: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
        logger.info(f"  Memory limit: {self.memory_manager.memory_limit_gb:.1f}GB")
        if self.use_gpu and self.gpu_properties is not None:
            logger.info(
                f"  GPU: {self.gpu_properties['name']} | VRAM: {self.gpu_properties['total_memory_gb']:.1f}GB | SMs: {self.gpu_properties['multi_processor_count']}"
            )
        else:
            logger.info("  GPU: Not available")
        logger.info(f"  Mode: {'Streaming' if self.streaming else 'Standard'} (auto-enabled: {self.streaming and args.streaming is False})")
        if self.measured_memory_per_building_mb:
            logger.info(f"  Calibrated memory per building: {self.measured_memory_per_building_mb:.3f} MB")
        logger.info(f"  Settings: Workers={self.n_workers}, Chunk size={self.chunk_size}")
        logger.info(f"  Features: Parallel={self.parallel}, Memory opt={self.optimize_memory}, Checkpoint={self.checkpoint}")
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Estimate ~1MB per building
        mb_per_building = 1.0
        available_mb = (self.available_memory_gb * 1024) * 0.5  # Use 50% of available
        chunk_size = int(available_mb / mb_per_building / max(self.n_workers, 1))
        return max(100, min(chunk_size, 10000))
    
    def _calculate_streaming_batch_size(self) -> int:
        """Calculate batch size for streaming mode (minimal memory)."""
        # In streaming mode, use very small batches
        return min(100, max(10, int(self.available_memory_gb * 10)))

    def _calibrate_and_set_chunk_size(self, user_batch_size: Optional[int]):
        """Derive optimal chunk size using a lightweight calibration sample.

        Strategy:
          1. If user supplied batch size -> trust it.
          2. Else try to load a small (e.g., 3000) household sample to estimate per-building memory.
          3. Use 45% of currently available memory for active batches divided across workers.
          4. Clamp into [1000, 20000] for stability. Smaller for streaming fallback.
        Calibration failure silently falls back to legacy heuristic.
        """
        if user_batch_size:
            self.chunk_size = user_batch_size
            return

        # If not parallel or very small memory just fallback
        try:
            if self.available_memory_gb < 4:  # Low memory system
                self.chunk_size = 1000 if self.streaming else 500
                return
        except Exception:
            pass

        # Attempt calibration only once
        try:
            from src.data_loading.pums_loader import load_pums_households
            calib_n = 3000 if self.streaming else 5000
            df_calib = load_pums_households(sample_size=calib_n)
            # Optimize types if possible
            try:
                df_calib = self.memory_manager.optimize_dataframe(df_calib, deep=True)
            except Exception:
                pass
            bytes_total = df_calib.memory_usage(deep=True).sum()
            per_building_mb = bytes_total / calib_n / (1024**2)
            # Guard against unrealistic values
            if per_building_mb <= 0 or per_building_mb > 5:
                raise ValueError("Unreasonable per-building memory estimate")
            self.measured_memory_per_building_mb = per_building_mb
            # Use 45% of *available* memory (dynamic) to remain safe, split across workers
            target_mb_pool = self.available_memory_gb * 1024 * 0.45
            raw_chunk = int(target_mb_pool / (per_building_mb * max(self.n_workers, 1)))
            # Clamp (more conservative for streaming to create more batches)
            lower = 1000 if self.streaming else 2000
            upper = 5000 if self.streaming else 30000
            self.chunk_size = max(lower, min(upper, raw_chunk))
            # Streaming special case: avoid excessively tiny batches
            if self.streaming and self.chunk_size < 500:
                self.chunk_size = 500
            # Clean up calibration sample aggressively
            del df_calib
            gc.collect()
        except Exception:
            # Fallback to legacy heuristics
            if self.streaming:
                self.chunk_size = self._calculate_streaming_batch_size()
            else:
                self.chunk_size = self._calculate_optimal_chunk_size()
    
    def _detect_device(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Detect best compute device and properties."""
        try:
            import torch
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                gpu_info = {
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor,
                    'device_index': device_index,
                }
                # Enable TF32 for speed on Ampere+ where acceptable
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
                return 'cuda', gpu_info
        except Exception:
            pass
        return 'cpu', None

    def _configure_threading(self) -> None:
        """Configure BLAS and Torch threading to avoid oversubscription."""
        # Cap threads to reasonable number to reduce contention
        max_threads = max(1, min(self.cpu_count, 8))
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            os.environ.setdefault(var, str(max_threads))
        try:
            import numpy as _np  # noqa: F401
        except Exception:
            pass
        try:
            import torch
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(max(1, max_threads // 2))
            if self.use_gpu:
                # Allow cuDNN benchmarking for fixed-size workloads
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
        except Exception:
            pass
    
    def optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if not self.optimize_memory:
            return df
        
        # Check if df is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            return df
            
        # Use memory manager's optimization
        return self.memory_manager.optimize_dataframe(df, deep=True)
    
    def parallel_process_chunks(self, data: pd.DataFrame, process_func: Callable,
                              desc: str = "Processing", **kwargs) -> pd.DataFrame:
        """Process DataFrame in parallel chunks."""
        if not self.parallel or len(data) < 1000:
            # Skip parallel for small datasets
            return process_func(data, **kwargs)
        
        n_chunks = max(1, len(data) // self.chunk_size)
        chunks = np.array_split(data, n_chunks)
        
        logger = logging.getLogger('pums_enrichment.main')
        logger.info(f"Processing {len(data)} rows in {n_chunks} chunks using {self.n_workers} workers")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_func, chunk, **kwargs): i 
                      for i, chunk in enumerate(chunks)}
            
            with tqdm(total=n_chunks, desc=desc, disable=not self.args.verbose) as pbar:
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        result = future.result()
                        results.append((chunk_idx, result))
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error in chunk {chunk_idx}: {e}")
                        pbar.update(1)
        
        # Sort and combine results
        results.sort(key=lambda x: x[0])
        processed_chunks = [r[1] for r in results if r[1] is not None]
        
        if processed_chunks:
            return pd.concat(processed_chunks, ignore_index=True)
        return pd.DataFrame()
    
    def save_checkpoint(self, data: Any, phase: str, iteration: int):
        """Save checkpoint for resumable processing."""
        if not self.checkpoint:
            return
        
        checkpoint_file = self.checkpoint_dir / f"phase{phase}_checkpoint_{iteration}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        
        metadata = {
            'phase': phase,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.checkpoint_dir / f"phase{phase}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, phase: str) -> Tuple[Optional[Any], Optional[int]]:
        """Load checkpoint if exists."""
        if not self.checkpoint:
            return None, None
        
        metadata_file = self.checkpoint_dir / f"phase{phase}_metadata.json"
        if not metadata_file.exists():
            return None, None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        checkpoint_file = self.checkpoint_dir / f"phase{phase}_checkpoint_{metadata['iteration']}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            return data, metadata['iteration']
        
        return None, None


def stream_load_pums_households(loader, sample_size: Optional[int], optimizer) -> pd.DataFrame:
    """Stream load PUMS households for memory efficiency."""
    chunks = []
    total_loaded = 0
    target_size = sample_size or float('inf')
    config = get_config()
    
    # Process state by state to manage memory
    pums_dir = Path(config.get_data_path('pums_household'))
    if pums_dir.is_file():
        pums_dir = pums_dir.parent
    
    state_files = list(pums_dir.glob('psam_hus*.csv'))
    
    with tqdm(total=min(len(state_files), target_size), desc="Loading PUMS data") as pbar:
        for state_file in state_files:
            if total_loaded >= target_size:
                break
            
            # Read state file in chunks
            for chunk in pd.read_csv(state_file, chunksize=optimizer.chunk_size):
                if total_loaded >= target_size:
                    break
                
                # Take only what we need
                remaining = target_size - total_loaded
                if len(chunk) > remaining:
                    chunk = chunk.head(remaining)
                
                chunks.append(chunk)
                total_loaded += len(chunk)
                pbar.update(len(chunk))
    
    return pd.concat(chunks, ignore_index=True)


def iter_pums_household_chunks(target_size: int, optimizer, logger: logging.Logger):
    """Yield unique household chunks sequentially across PUMS files without repetition.

    This avoids repeatedly loading the first N rows (behavior of load_pums_households with sample_size)
    and instead streams each CSV file in chunked fashion until target_size reached.
    """
    from src.utils.config_loader import get_config
    import pandas as pd
    import numpy as np
    from pathlib import Path

    config = get_config()
    household_file = config.get_data_path('pums_household')
    household_file_a = Path(household_file)
    household_file_b = household_file_a.parent / household_file_a.name.replace('husa', 'husb')
    files = [f for f in [household_file_a, household_file_b] if f.exists()]
    if not files:
        raise FileNotFoundError(f"No household PUMS files found starting from {household_file_a}")

    # Dtype strategy mirrors pums_loader for consistency (subset of columns to limit memory)
    dtype_base = {
        'RT': 'category', 'SERIALNO': 'object', 'DIVISION': 'int8', 'PUMA': 'object', 'REGION': 'int8',
        'STATE': 'object', 'NP': 'int8', 'TYPEHUGQ': 'int8', 'WGTP': 'int16', 'HINCP': 'float32',
        'YRBLT': 'float32', 'BDSP': 'float32', 'RMSP': 'float32', 'VEH': 'float32', 'HFL': 'float32',
        'ELEP': 'float32', 'GASP': 'float32', 'FULP': 'float32', 'BLD': 'float32', 'TEN': 'float32',
        'ACR': 'float32', 'BATH': 'float32', 'KIT': 'float32', 'HHT': 'float32', 'GRPIP': 'float32',
        'OCPIP': 'float32', 'R18': 'float32', 'R60': 'float32', 'R65': 'float32'
    }
    # Add weight replicate columns
    for i in range(1, 81):
        dtype_base[f'WGTP{i}'] = 'int16'

    # Determine chunk size for raw CSV reads: align with calibrated chunk_size (allow more batches)
    csv_chunk = max(2000, min(optimizer.chunk_size, 10000))
    emitted = 0
    seen_serials = set()

    for file_path in files:
        logger.info(f"Streaming households from {file_path.name} in chunks of {csv_chunk}")
        for chunk in pd.read_csv(file_path, dtype=dtype_base, chunksize=csv_chunk, low_memory=False):
            # Filter to housing units (RT='H', TYPEHUGQ ==1)
            chunk = chunk[(chunk['RT'] == 'H') & (chunk['TYPEHUGQ'] == 1)]
            if chunk.empty:
                continue
            # Drop already seen SERIALNO to avoid duplication
            before = len(chunk)
            chunk = chunk[~chunk['SERIALNO'].isin(seen_serials)]
            after = len(chunk)
            if after == 0:
                continue
            # Update seen serials
            seen_serials.update(chunk['SERIALNO'].tolist())
            # Trim to target size
            remaining = target_size - emitted
            if after > remaining:
                chunk = chunk.head(remaining)
            # Add building_id now
            chunk['building_id'] = 'BLDG_' + chunk['SERIALNO'].astype(str)
            yield chunk
            emitted += len(chunk)
            if emitted >= target_size:
                return
    # If target not reached and no sample size provided, just yield everything (full dataset) without cap
    if target_size == float('inf'):
        return


def _phase_metadata_path(phase: int, config) -> Path:
    mapping = {
        1: 'phase1_metadata.json',
        2: 'phase2_metadata.json',
        3: 'phase3_metadata.json',
        4: 'phase4_metadata.json'
    }
    base = Path(config.get_data_path('phase1_output')).parent  # processed dir
    return base / mapping[phase]


def _phase_output_path(phase: int, config) -> Path:
    keys = {
        1: 'phase1_output',
        2: 'phase2_output',
        3: 'phase3_output',
        4: 'phase4_output'
    }
    return Path(config.get_data_path(keys[phase]))


def _is_phase_complete(phase: int, config) -> bool:
    """Heuristic to determine if a phase is complete: output + metadata + non-zero size."""
    try:
        out_path = _phase_output_path(phase, config)
        if not (out_path.exists() and out_path.stat().st_size > 0):
            return False
        meta_path = _phase_metadata_path(phase, config)
        if not meta_path.exists():
            # Allow Phase 1 to pass without metadata (legacy)
            return phase == 1
        # Basic sanity: metadata must be valid JSON and contain end_time or similar
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # Accept if any indicative key present
        indicative_keys = {'end_time','processing_time_seconds','buildings_processed','matches_found'}
        if not any(k in meta for k in indicative_keys):
            return False
        # Optional: consistency check with previous phase count
        if phase > 1:
            try:
                prev_out = _phase_output_path(phase-1, config)
                if prev_out.exists() and prev_out.stat().st_size > 0:
                    # Light-weight count consistency: ensure phase file size not absurdly smaller (<10%)
                    prev_size = prev_out.stat().st_size
                    curr_size = out_path.stat().st_size
                    if curr_size < 0.1 * prev_size:
                        return False
            except Exception:
                pass
        return True
    except Exception:
        return False


def get_last_completed_phase(force_start: Optional[int] = None) -> int:
    """Determine last completed phase, honoring forced start override."""
    config = get_config()
    if force_start is not None:
        # Validate earlier phases actually exist up to force_start-1
        return force_start - 1
    last = 0
    for p in range(1,5):
        if _is_phase_complete(p, config):
            last = p
        else:
            break
    return last


# ---------------------------------------------
# Data counting helpers for logging at key stages
# ---------------------------------------------
def _sum_persons(df: pd.DataFrame) -> int:
    """Sum total persons embedded in the 'persons' column if available.
    Falls back to 'actual_person_count' or zero.
    """
    if isinstance(df, pd.DataFrame):
        if 'persons' in df.columns:
            def _safe_len(x):
                try:
                    return len(x)
                except Exception:
                    return 0
            try:
                return int(df['persons'].map(_safe_len).sum())
            except Exception:
                pass
        for c in ['actual_person_count', 'num_persons', 'person_count', 'n_persons']:
            if c in df.columns:
                try:
                    return int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())
                except Exception:
                    continue
    return 0


def _safe_upcast_small_uints(df: pd.DataFrame) -> pd.DataFrame:
    """Upcast any uint8 columns to uint16 to avoid overflow in downstream assignments.
    No-op if df is not a DataFrame or has no such columns.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            return df
        for c in df.select_dtypes(include=['uint8']).columns:
            try:
                df[c] = df[c].astype('uint16')
            except Exception:
                continue
    except Exception:
        pass
    return df


def _compute_raw_counts() -> Tuple[int, int]:
    """Compute raw counts from PUMS CSVs without loading full data into memory.
    Returns (households_count, persons_count).
    """
    cfg = get_config()
    # Household files (A + B)
    hh_a = Path(cfg.get_data_path('pums_household'))
    hh_b = hh_a.parent / hh_a.name.replace('husa', 'husb')
    hh_files = [p for p in [hh_a, hh_b] if p.exists()]
    # Person files (A + B)
    pr_a = Path(cfg.get_data_path('pums_person'))
    pr_b = pr_a.parent / pr_a.name.replace('pusa', 'pusb')
    pr_files = [p for p in [pr_a, pr_b] if p.exists()]

    hh_count = 0
    pr_count = 0

    try:
        # Count households: RT == 'H' and TYPEHUGQ == 1
        for fp in hh_files:
            try:
                for chunk in pd.read_csv(
                    fp,
                    usecols=['RT', 'TYPEHUGQ'],
                    dtype={'RT': 'category', 'TYPEHUGQ': 'int8'},
                    chunksize=200000,
                    low_memory=False,
                ):
                    hh_count += int(((chunk['RT'] == 'H') & (chunk['TYPEHUGQ'] == 1)).sum())
            except Exception:
                continue
    except Exception:
        pass

    try:
        # Count persons: RT == 'P'
        for fp in pr_files:
            try:
                for chunk in pd.read_csv(
                    fp,
                    usecols=['RT'],
                    dtype={'RT': 'category'},
                    chunksize=200000,
                    low_memory=False,
                ):
                    pr_count += int((chunk['RT'] == 'P').sum())
            except Exception:
                continue
    except Exception:
        pass

    return hh_count, pr_count


def run_phase1(sample_size: Optional[int], skip_validation: bool) -> bool:
    """
    Run Phase 1 processing with optimization support and memory management.
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger('pums_enrichment.main')
    
    try:
        logger.info("Starting Phase 1: PUMS Household-Person Integration")
        
        # Monitor memory throughout Phase 1
        with optimizer.memory_manager.monitor_operation("Phase 1"):
            
            # Validate inputs
            if sample_size is not None and sample_size <= 0:
                raise ValueError(f"Invalid sample size: {sample_size}")
            
            # Check for checkpoint if enabled
            if optimizer.checkpoint and optimizer.args.resume:
                checkpoint_data, iteration = optimizer.load_checkpoint('phase1')
                if checkpoint_data is not None:
                    logger.info(f"Resuming Phase 1 from checkpoint (iteration {iteration})")
                    # Skip to validation if checkpoint is complete
                    buildings = checkpoint_data
                    if not skip_validation:
                        from src.validation.phase_validators import run_all_validations
                        validation_results = run_all_validations(buildings, current_phase=1)
                        if not validation_results.get('overall_valid', True):
                            logger.error("Phase 1 validation failed")
                            return False
                    return True
            
            # Check if we're in streaming mode
            if optimizer.streaming or (sample_size and sample_size > 10000):
                logger.info(f"Using streaming mode with calibrated batch size {optimizer.chunk_size}")
                target_size = sample_size or float('inf')
                processed = 0
                batch_idx = 0
                # Prepare shard output directory and manifest
                out_base = _phase_output_path(1, get_config())
                shards_dir = out_base.parent / 'phase1_shards'
                shards_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = shards_dir / 'manifest.json'
                shard_files: List[str] = []
                sample_parts: List[pd.DataFrame] = []
                try:
                    for hh_chunk in iter_pums_household_chunks(target_size, optimizer, logger):
                        if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.05):
                            logger.warning("Memory pressure before Phase 1 chunk processing - forcing cleanup")
                            optimizer.memory_manager.force_cleanup(level=2)
                            if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.05):
                                logger.error("Insufficient memory to continue Phase 1 streaming")
                                break
                        # Optimize and process this chunk
                        hh_chunk = optimizer.optimize_datatypes(hh_chunk)
                        result = process_phase1(
                            sample_size=len(hh_chunk),
                            validate=False,
                            input_data=hh_chunk,
                            save_output=False
                        )
                        batch_buildings = result[0] if isinstance(result, tuple) else result
                        # Save shard
                        shard_path = shards_dir / f'phase1_part_{batch_idx:05d}.pkl'
                        try:
                            batch_buildings.to_pickle(shard_path)
                            shard_files.append(str(shard_path))
                        except Exception as e:
                            logger.error(f"Failed to save Phase 1 shard {batch_idx}: {e}")
                            return False
                        # Accumulate small sample for quick checks
                        if len(sample_parts) < 3:
                            sample_parts.append(batch_buildings.head(4000))
                        processed += len(batch_buildings)
                        batch_idx += 1
                        logger.info(f"Processed {processed}{'' if target_size==float('inf') else f' / {target_size}'} unique households")
                        # Periodic cleanup
                        if batch_idx % 3 == 0:
                            optimizer.memory_manager.force_cleanup(level=2)
                        gc.collect()
                        if processed >= target_size:
                            break
                except Exception as e:
                    logger.error(f"Streaming Phase 1 failed mid-run: {e}")
                    return False
                # Finalize manifest and sample output
                try:
                    with open(manifest_path, 'w') as f:
                        json.dump({'n_shards': batch_idx, 'files': shard_files, 'total_buildings': int(processed)}, f, indent=2)
                    sample_out = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
                    # Write a small sample to canonical path for downstream checks
                    # Use save_phase1_output to ensure metadata and sample CSV are created
                    meta = {
                        'phase': 'phase1',
                        'end_time': datetime.now().isoformat(),
                        'buildings_processed': int(processed),
                        'sharded': True
                    }
                    try:
                        save_phase1_output(sample_out, meta)
                    except Exception:
                        # Fallback direct pickle write if helper fails
                        out_path = _phase_output_path(1, get_config())
                        sample_out.to_pickle(out_path)
                        with open(_phase_metadata_path(1, get_config()), 'w') as f:
                            json.dump(meta, f, indent=2)
                    buildings = sample_out
                except Exception as e:
                    logger.warning(f"Could not finalize Phase 1 outputs: {e}")
                    buildings = pd.DataFrame()
                
            else:
                # Standard processing for small datasets
                result = process_phase1(
                    sample_size=sample_size,
                    validate=False
                )
                # Check if result is a tuple (buildings, metadata) or just buildings
                if isinstance(result, tuple):
                    buildings, metadata = result
                else:
                    buildings = result
            
            # Optimize final output
            try:
                if isinstance(buildings, pd.DataFrame):
                    buildings = optimizer.optimize_datatypes(buildings)
                else:
                    logger.warning(f"Buildings is not a DataFrame: {type(buildings)}")
            except Exception as e:
                logger.warning(f"Could not optimize datatypes: {e}")
            
            # Validate if not skipped
            if not skip_validation:
                from src.validation.phase_validators import run_all_validations
                validation_results = run_all_validations(buildings, current_phase=1)
                if not validation_results.get('overall_valid', True):
                    logger.error("Phase 1 validation failed")
                    return False
            
            logger.info(f"Phase 1 completed successfully with {len(buildings)} buildings")
            
            # Track memory usage
            memory_usage = optimizer.memory_manager.get_memory_usage()
            optimizer.memory_manager.track_phase_memory("Phase 1", memory_usage['rss_gb'])
            
            # Force cleanup before next phase
            optimizer.memory_manager.force_cleanup(level=2)
            
            return True
        
    except ImportError as e:
        logger.error(f"Missing required module for Phase 1: {str(e)}")
        return False
    except ValueError as e:
        logger.error(f"Invalid parameter for Phase 1: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Phase 1 failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return False


def run_phase2(sample_size: Optional[int], skip_validation: bool) -> bool:
    """
    Run Phase 2 processing - RECS matching with memory management.
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger('pums_enrichment.main')
    config = get_config()
    
    try:
        logger.info("Starting Phase 2: RECS Probabilistic Matching")

        # Import Phase 2 processor
        from src.processing.phase2_recs_matching import Phase2RECSMatcher
        processor = Phase2RECSMatcher()
        # Apply CLI overrides for Phase 2 if provided
        try:
            # Use CLI overrides via optimizer.args (global)
            p2_max_cand = getattr(optimizer.args, 'phase2_max_candidates', None)
            if p2_max_cand:
                processor.max_candidates_per_record = int(p2_max_cand)
        except Exception:
            pass

        # Monitor memory throughout Phase 2
        with optimizer.memory_manager.monitor_operation("Phase 2"):
            # If streaming mode, process in batches
            if optimizer.streaming:
                logger.info(f"Using streaming mode for Phase 2 with calibrated batch size {optimizer.chunk_size}")

                target_size = sample_size or float('inf')
                processed = 0
                batch_idx = 0
                # Prepare shard output directory and manifest
                out_base = _phase_output_path(2, config)
                shards_dir = out_base.parent / 'phase2_shards'
                shards_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = shards_dir / 'manifest.json'
                shard_files: List[str] = []
                sample_df_list: List[pd.DataFrame] = []

                # Prefer streaming from Phase 1 shards if available
                p1_shards_dir = _phase_output_path(1, config).parent / 'phase1_shards'
                p1_manifest = p1_shards_dir / 'manifest.json'
                use_p1_shards = p1_shards_dir.exists() and p1_manifest.exists()
                if use_p1_shards:
                    logger.info("Detected Phase 1 shards - streaming Phase 2 over Phase 1 shard files")
                    try:
                        with open(p1_manifest, 'r') as f:
                            p1_info = json.load(f)
                            p1_files = p1_info.get('files', [])
                    except Exception as e:
                        logger.warning(f"Failed to read Phase 1 manifest; falling back to raw streaming: {e}")
                        use_p1_shards = False

                try:
                    if use_p1_shards:
                        for fp in p1_files:
                            if processed >= target_size:
                                break
                            try:
                                buildings_chunk = pd.read_pickle(fp)
                            except Exception as e:
                                logger.warning(f"Skipping unreadable Phase 1 shard {fp}: {e}")
                                continue
                            # Apply sample cap across shards
                            remaining = target_size - processed
                            if remaining != float('inf') and len(buildings_chunk) > remaining:
                                buildings_chunk = buildings_chunk.head(remaining)

                            # Sub-batch this shard to keep memory bounded
                            p2_sub_batch = getattr(optimizer.args, 'phase2_sub_batch', None)
                            sub_batch_size = p2_sub_batch if p2_sub_batch else max(500, min(2000, optimizer.chunk_size // 2 if optimizer.chunk_size else 2000))
                            n = len(buildings_chunk)
                            for start in range(0, n, sub_batch_size):
                                if processed >= target_size:
                                    break
                                end = min(start + sub_batch_size, n)
                                sub_df = buildings_chunk.iloc[start:end]
                                if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.1):
                                    logger.warning("Memory pressure before Phase 2 sub-batch - forcing cleanup")
                                    optimizer.memory_manager.force_cleanup(level=3)
                                    # Re-check after cleanup; abort if still insufficient to avoid thrashing
                                    if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.1):
                                        logger.error("Insufficient memory to continue Phase 2 after cleanup. "
                                                     "Reduce --phase2-sub-batch, lower --phase2-max-candidates, "
                                                     "or increase --memory-limit and retry.")
                                        return False
                                # Run Phase 2 matcher on sub-batch
                                try:
                                    matched_chunk = processor.run(sample_size=len(sub_df), input_data=sub_df, save_output=False)
                                except Exception as e:
                                    logger.error(f"Phase 2 sub-batch failed: {e}")
                                    return False
                                # Save shard piece
                                shard_path = shards_dir / f'phase2_part_{batch_idx:05d}.pkl'
                                try:
                                    matched_chunk.to_pickle(shard_path)
                                    shard_files.append(str(shard_path))
                                except Exception as e:
                                    logger.error(f"Failed to save Phase 2 shard {batch_idx}: {e}")
                                    return False
                                if len(sample_df_list) < 3:
                                    sample_df_list.append(matched_chunk.head(4000))
                                processed += len(matched_chunk)
                                batch_idx += 1
                                if batch_idx % 5 == 0:
                                    optimizer.memory_manager.force_cleanup(level=2)
                                del sub_df, matched_chunk
                                gc.collect()
                    else:
                        # Strict dependency: use ONLY Phase 1 output
                        p1_output_path = _phase_output_path(1, config)
                        if not p1_output_path.exists():
                            logger.error("Phase 1 output not found. Run Phase 1 first to produce Phase 1 shards or output.")
                            return False
                        if target_size == float('inf'):
                            logger.error("Full-data Phase 2 requires Phase 1 shards for streaming. Re-run Phase 1 to create shards.")
                            return False
                        try:
                            buildings_full = pd.read_pickle(p1_output_path)
                        except Exception as e:
                            logger.error(f"Failed to load Phase 1 output: {e}")
                            return False
                        # Apply sample cap
                        if isinstance(target_size, (int, float)) and target_size != float('inf'):
                            buildings_full = buildings_full.head(int(target_size))
                        # Sub-batch from Phase 1 single output
                        p2_sub_batch = getattr(optimizer.args, 'phase2_sub_batch', None)
                        sub_batch_size = p2_sub_batch if p2_sub_batch else max(500, min(2000, optimizer.chunk_size // 2 if optimizer.chunk_size else 2000))
                        n = len(buildings_full)
                        for start in range(0, n, sub_batch_size):
                            end = min(start + sub_batch_size, n)
                            sub_df = buildings_full.iloc[start:end]
                            if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.1):
                                logger.warning("Memory pressure before Phase 2 sub-batch - forcing cleanup")
                                optimizer.memory_manager.force_cleanup(level=3)
                                # Re-check after cleanup; abort if still insufficient to avoid thrashing
                                if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.1):
                                    logger.error("Insufficient memory to continue Phase 2 after cleanup. "
                                                 "Reduce --phase2-sub-batch, lower --phase2-max-candidates, "
                                                 "or increase --memory-limit and retry.")
                                    return False
                            try:
                                matched_chunk = processor.run(sample_size=len(sub_df), input_data=sub_df, save_output=False)
                            except Exception as e:
                                logger.error(f"Phase 2 sub-batch failed: {e}")
                                return False
                            shard_path = shards_dir / f'phase2_part_{batch_idx:05d}.pkl'
                            try:
                                matched_chunk.to_pickle(shard_path)
                                shard_files.append(str(shard_path))
                            except Exception as e:
                                logger.error(f"Failed to save Phase 2 shard {batch_idx}: {e}")
                                return False
                            if len(sample_df_list) < 3:
                                sample_df_list.append(matched_chunk.head(4000))
                            processed += len(matched_chunk)
                            batch_idx += 1
                            if batch_idx % 5 == 0:
                                optimizer.memory_manager.force_cleanup(level=2)
                            del sub_df, matched_chunk
                            gc.collect()

                    # Write manifest and small sample file
                    try:
                        with open(manifest_path, 'w') as f:
                            json.dump({'n_shards': batch_idx, 'files': shard_files, 'total_buildings': int(processed)}, f, indent=2)
                        sample_out = pd.concat(sample_df_list, ignore_index=True) if sample_df_list else pd.DataFrame()
                        out_path = _phase_output_path(2, config)
                        sample_out.to_pickle(out_path)
                        with open(_phase_metadata_path(2, config), 'w') as f:
                            json.dump({'end_time': datetime.now().isoformat(), 'buildings_processed': int(processed), 'sharded': True}, f, indent=2)
                        matched_buildings = sample_out
                    except Exception as e:
                        logger.warning(f"Could not finalize Phase 2 outputs: {e}")
                        matched_buildings = pd.DataFrame()
                except Exception as e:
                    logger.error(f"Streaming Phase 2 failed mid-run: {e}")
                    return False
            else:
                # Standard processing
                matched_buildings = processor.run(sample_size=sample_size)
            
            # Optimize output
            try:
                if isinstance(matched_buildings, pd.DataFrame):
                    matched_buildings = optimizer.optimize_datatypes(matched_buildings)
            except Exception as e:
                logger.warning(f"Could not optimize datatypes: {e}")
            
            logger.info(f"Phase 2 completed successfully with {len(matched_buildings)} buildings")
            
            # Track memory usage
            memory_usage = optimizer.memory_manager.get_memory_usage()
            optimizer.memory_manager.track_phase_memory("Phase 2", memory_usage['rss_gb'])
            
            # Run validation if not skipped
            if not skip_validation:
                validation_results = run_all_validations(matched_buildings, current_phase=2)
                if not validation_results.get('overall_valid', True):
                    logger.error("Phase 2 validation failed")
                    return False
            
            # Force cleanup before next phase
            optimizer.memory_manager.force_cleanup(level=2)
            
            return True
        
    except ImportError as e:
        logger.error(f"Missing required module for Phase 2: {str(e)}")
        return False
    except ValueError as e:
        logger.error(f"Invalid parameter for Phase 2: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return False


def run_phase3(sample_size: Optional[int], skip_validation: bool) -> bool:
    """
    Run Phase 3 processing - ATUS activity pattern matching with memory optimization.
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger('pums_enrichment.main')
    config = get_config()
    
    try:
        logger.info("Starting Phase 3: ATUS Activity Pattern Matching")

        # Fast skip if output already exists and --resume provided
        if optimizer.args.resume:
            try:
                phase3_output_path = Path(get_config().get_data_path('phase3_output'))
                if phase3_output_path.exists() and phase3_output_path.stat().st_size > 0:
                    logger.info("Phase 3 output already exists; skipping Phase 3 due to --resume")
                    return True
            except Exception:
                pass
        
        # Monitor memory throughout Phase 3
        with optimizer.memory_manager.monitor_operation("Phase 3"):
            
            # Check memory before starting
            if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.3):
                logger.error("Insufficient memory to start Phase 3")
                return False
            
            # Import Phase 3 processor - use optimized version
            from src.processing.phase3_atus_matching_optimized import OptimizedPhase3Matcher
            
            # Initialize processor
            processor = OptimizedPhase3Matcher()
            
            # Load ATUS data efficiently
            if optimizer.streaming:
                logger.info("Using memory-efficient ATUS loading")
                # The processor will handle memory-efficient loading internally
            
            # Enforce dependency: require Phase 2 output/shards
            try:
                p2_out = Path(get_config().get_data_path('phase2_output'))
                p2_shards = p2_out.parent / 'phase2_shards' / 'manifest.json'
                if not p2_out.exists() and not p2_shards.exists():
                    logger.error("Phase 2 output not found. Run Phase 2 first to produce output or shards.")
                    return False
            except Exception:
                pass

            # Run Phase 3 matching
            # Derive a memory-aware batch size for k-NN matching of persons
            # Reuse optimizer.chunk_size as a baseline but cap to avoid huge GPU allocations
            # Smaller batches to avoid GPU/CPU memory spikes
            match_batch_size = max(2000, min(optimizer.chunk_size * 2, 12000)) if optimizer.streaming else None
            buildings_with_activities = processor.run(sample_size=sample_size, match_batch_size=match_batch_size)

            # Save checkpoint (final) for potential downstream debugging
            if optimizer.checkpoint:
                try:
                    optimizer.save_checkpoint(buildings_with_activities.head(1000), 'phase3', len(buildings_with_activities))
                except Exception:
                    logger.debug("Could not save Phase 3 checkpoint", exc_info=True)
            
            # Optimize output
            try:
                if isinstance(buildings_with_activities, pd.DataFrame):
                    buildings_with_activities = optimizer.optimize_datatypes(buildings_with_activities)
            except Exception as e:
                logger.warning(f"Could not optimize datatypes: {e}")
            
            logger.info(f"Phase 3 completed successfully with {len(buildings_with_activities)} buildings")
            
            # Track memory usage
            memory_usage = optimizer.memory_manager.get_memory_usage()
            optimizer.memory_manager.track_phase_memory("Phase 3", memory_usage['rss_gb'])
            
            # Run validation if not skipped
            if not skip_validation:
                validation_results = run_all_validations(buildings_with_activities, current_phase=3)
                if not validation_results.get('overall_valid', True):
                    logger.error("Phase 3 validation failed")
                    return False
            
            # Force aggressive cleanup before Phase 4 (which is memory-intensive)
            optimizer.memory_manager.force_cleanup(level=3)
            
            return True
        
    except ImportError as e:
        logger.error(f"Missing required module for Phase 3: {str(e)}")
        return False
    except ValueError as e:
        logger.error(f"Invalid parameter for Phase 3: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Phase 3 failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return False


def run_phase4(sample_size: Optional[int], skip_validation: bool) -> bool:
    """
    Run Phase 4 processing - Weather integration with activity alignment.
    Uses memory-efficient weather referencing instead of embedding.
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger('pums_enrichment.main')
    config = get_config()
    
    try:
        logger.info("Starting Phase 4: Weather Integration with Activity Alignment")

        # Fast skip if output already exists and --resume provided
        if optimizer.args.resume:
            try:
                phase4_output_path = Path(get_config().get_data_path('phase4_output'))
                if phase4_output_path.exists() and phase4_output_path.stat().st_size > 0:
                    logger.info("Phase 4 output already exists; skipping Phase 4 due to --resume")
                    return True
            except Exception:
                pass
        
        # Monitor memory throughout Phase 4
        with optimizer.memory_manager.monitor_operation("Phase 4"):
            
            # Check memory before starting - Phase 4 can be memory-intensive
            if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.5):
                logger.error("Insufficient memory to start Phase 4")
                logger.info("Attempting aggressive cleanup...")
                optimizer.memory_manager.force_cleanup(level=3)
                
                # Check again
                if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.3):
                    logger.error("Still insufficient memory after cleanup")
                    return False
            
            # Import Phase 4 processor
            from src.processing.phase4_weather_integration import Phase4WeatherIntegrator
            from datetime import datetime
            
            # Initialize processor with memory-efficient settings
            processor = Phase4WeatherIntegrator(
                cache_size_mb=100 if optimizer.streaming else 500,  # Limit cache size
                use_references=True  # Use weather references instead of embedding
            )
            
            # Enforce dependency: require Phase 3 output/shards
            try:
                p3_out = Path(get_config().get_data_path('phase3_output'))
                p3_shards = p3_out.parent / 'phase3_shards' / 'manifest.json'
                if not p3_out.exists() and not p3_shards.exists():
                    logger.error("Phase 3 output not found. Run Phase 3 first to produce output or shards.")
                    return False
            except Exception:
                pass

            # Run Phase 4 integration - default to January 1, 2023
            default_date = datetime(2023, 1, 1)
            
            # Enhanced resume-aware Phase 4 execution with per-state checkpointing
            buildings_with_weather = None
            resume_states_processed = set()
            resume_partial_df = None
            resume_iteration = 0
            if optimizer.checkpoint and optimizer.args.resume:
                try:
                    ckpt_data, iteration = optimizer.load_checkpoint('phase4')
                    if ckpt_data is not None:
                        if isinstance(ckpt_data, dict):
                            # Expect structure {'data': DataFrame, 'states': list[str]}
                            resume_partial_df = ckpt_data.get('data')
                            resume_states_processed = set(ckpt_data.get('states', []))
                            resume_iteration = iteration or 0
                            logger.info(f"Resuming Phase 4 from checkpoint: {len(resume_states_processed)} states already processed")
                        elif isinstance(ckpt_data, pd.DataFrame):
                            # Legacy style DataFrame only
                            resume_partial_df = ckpt_data
                            logger.info(f"Resuming Phase 4 from legacy checkpoint with {len(resume_partial_df)} rows")
                except Exception:
                    logger.debug("Could not load Phase 4 checkpoint", exc_info=True)

            # Execute processor.run normally, but intercept internal state loop via monkeypatch if streaming
            # If not streaming, just run once (fast) and checkpoint final
            if not optimizer.streaming:
                buildings_with_weather = processor.run(
                    sample_size=sample_size,
                    use_cached_weather=True,
                    date=default_date
                )
            else:
                # We cannot easily inject into processor internal loop without editing source; so perform a two-step approach:
                # 1. Run processor in streaming_mode to generate batches (processor already supports streaming_mode & batch_size)
                # 2. If resume requested, we still have to re-run full since internal state segmentation is opaque; fallback to full run
                # NOTE: For future optimization, expose state-level iterator inside processor.
                # Ensure multiple batches for safety in streaming mode
                phase4_batch = max(1000, min(optimizer.chunk_size, 5000))
                buildings_with_weather = processor.run(
                    sample_size=sample_size,
                    use_cached_weather=True,
                    date=default_date,
                    streaming_mode=True,
                    batch_size=phase4_batch
                )

            # After run, augment with resume partial if any (avoid duplication)
            if resume_partial_df is not None and buildings_with_weather is not None:
                try:
                    # Concatenate and drop duplicates if overlapping
                    combined = pd.concat([resume_partial_df, buildings_with_weather], ignore_index=True)
                    if 'building_id' in combined.columns:
                        combined = combined.drop_duplicates(subset=['building_id'])
                    buildings_with_weather = combined
                except Exception:
                    logger.debug("Failed to merge resume partial Phase 4 data", exc_info=True)

            # Save final checkpoint snapshot (small sample) for potential troubleshooting
            if optimizer.checkpoint:
                try:
                    snapshot = buildings_with_weather.head(1000) if isinstance(buildings_with_weather, pd.DataFrame) else buildings_with_weather
                    optimizer.save_checkpoint(snapshot, 'phase4', len(buildings_with_weather) if isinstance(buildings_with_weather, pd.DataFrame) else 0)
                except Exception:
                    logger.debug("Could not save Phase 4 checkpoint", exc_info=True)
            
            # Optimize output
            try:
                if isinstance(buildings_with_weather, pd.DataFrame):
                    buildings_with_weather = optimizer.optimize_datatypes(buildings_with_weather)
            except Exception as e:
                logger.warning(f"Could not optimize datatypes: {e}")
            
            logger.info(f"Phase 4 completed successfully with {len(buildings_with_weather)} buildings")
            logger.info(f"Weather data aligned for date: {default_date.strftime('%Y-%m-%d')}")
            
            # Track memory usage
            memory_usage = optimizer.memory_manager.get_memory_usage()
            optimizer.memory_manager.track_phase_memory("Phase 4", memory_usage['rss_gb'])
            
            # Run validation if not skipped
            if not skip_validation:
                validation_results = run_all_validations(buildings_with_weather, current_phase=4)
                if not validation_results.get('overall_valid', True):
                    logger.error("Phase 4 validation failed")
                    return False
            
            return True
        
    except ImportError as e:
        logger.error(f"Missing required module for Phase 4: {str(e)}")
        return False
    except ValueError as e:
        logger.error(f"Invalid parameter for Phase 4: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Phase 4 failed: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        return False


def validate_existing_output():
    """Validate existing pipeline output."""
    logger = logging.getLogger('pums_enrichment.main')
    
    # Local helper to make nested objects JSON-serializable (handles numpy/pandas types)
    def _to_serializable(obj):
        try:
            import numpy as _np
            import pandas as _pd
        except Exception:
            _np = None
            _pd = None
        
        # Numpy scalar types
        if _np is not None and isinstance(obj, (_np.integer, _np.int8, _np.int16, _np.int32, _np.int64)):
            return int(obj)
        if _np is not None and isinstance(obj, (_np.floating, _np.float16, _np.float32, _np.float64)):
            return float(obj)
        if _np is not None and isinstance(obj, _np.bool_):
            return bool(obj)
        # Numpy arrays
        if _np is not None and isinstance(obj, _np.ndarray):
            return obj.tolist()
        # Pandas types
        if _pd is not None and isinstance(obj, _pd.Series):
            return obj.tolist()
        if _pd is not None and isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        # Built-ins and containers
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_serializable(v) for v in list(obj)]
        # Objects exposing .item() (numpy scalars)
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            try:
                return obj.item()
            except Exception:
                pass
        return obj
    
    # Determine which phase output exists
    last_phase = get_last_completed_phase()
    
    if last_phase == 0:
        logger.error("No phase output found to validate")
        return False
    
    logger.info(f"Found output for phases 1-{last_phase}")
    
    # Load the latest output
    config = get_config()
    phase_loaders = {
        1: 'phase1_output',
        2: 'phase2_output',
        3: 'phase3_output',
        4: 'phase4_output'
    }
    
    try:
        import pickle
        output_path = Path(config.get_data_path(phase_loaders[last_phase]))
        
        with open(output_path, 'rb') as f:
            buildings = pickle.load(f)
        
        logger.info(f"Loaded {len(buildings)} buildings from Phase {last_phase}")
        
        # Run validation
        validation_results = run_all_validations(buildings, current_phase=last_phase)
        
        # Log results
        if validation_results['overall_valid']:
            logger.info("Validation PASSED")
        else:
            logger.error("Validation FAILED")
        
        # Save validation results (ensure JSON-serializable types)
        results_path = output_path.parent / f'validation_results_phase{last_phase}.json'
        with open(results_path, 'w') as f:
            json.dump(_to_serializable(validation_results), f, indent=2)
        
        logger.info(f"Validation results saved to: {results_path}")
        
        return validation_results['overall_valid']
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
    except SystemExit:
        # Argparse exits on -h/--help or errors
        raise
    except Exception as e:
        print(f"Error parsing arguments: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize performance optimizer
    global optimizer
    optimizer = PerformanceOptimizer(args)
    
    # Set up logging
    try:
        log_level = 'DEBUG' if args.verbose else 'INFO'
        logger = setup_logging('main', console=True)
        
        # Override log level if verbose
        if args.verbose:
            logger.setLevel(logging.DEBUG)
    except Exception as e:
        print(f"Error setting up logging: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("PUMS ENRICHMENT PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load configuration with error handling
        try:
            config = get_config()
            logger.info(f"Configuration loaded from: {args.config}")
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
        
        # Determine sample size
        # Special case: --phase all defaults to full data unless explicitly specified otherwise
        if args.full_data:
            # Explicit --full-data flag
            sample_size = None
            logger.info("Running with FULL dataset (--full-data flag)")
        elif args.sample_size is not None:
            # Explicit sample size provided
            sample_size = args.sample_size
            logger.info(f"Running with specified sample size: {sample_size}")
        elif args.phase == 'all':
            # --phase all without explicit size means FULL data
            sample_size = None
            logger.info("Running --phase all with FULL dataset (default for all phases)")
        else:
            # Single phase without explicit size - use config default
            sample_size = config.get_sample_size()
            if sample_size:
                logger.info(f"Running with default sample size from config: {sample_size}")
            else:
                logger.info("Running with FULL dataset (no sample size in config)")
        
        # Validate only mode
        if args.validate_only:
            logger.info("Running validation only")
            success = validate_existing_output()
            sys.exit(0 if success else 1)
        
        # Determine phases to run
        if args.resume:
            start_phase = get_last_completed_phase(force_start=args.from_phase) + 1
            if start_phase > 4:
                logger.info("All phases already completed")
                sys.exit(0)
            
            if args.phase == 'all':
                phases_to_run = list(range(start_phase, 5))
            else:
                phase_num = int(args.phase)
                if phase_num < start_phase:
                    logger.warning(f"Phase {phase_num} already completed")
                    sys.exit(0)
                phases_to_run = [phase_num]
        else:
            if args.phase == 'all':
                phases_to_run = [1, 2, 3, 4]
            else:
                phases_to_run = [int(args.phase)]
        
        logger.info(f"Phases to run: {phases_to_run}")
        
        # Log raw data counts before running
        try:
            raw_hh, raw_pr = _compute_raw_counts()
            logger.info(f"RAW DATA COUNTS - households: {raw_hh:,} | persons: {raw_pr:,}")
        except Exception as e:
            logger.warning(f"Could not compute raw data counts: {e}")

        # Run phases
        phase_runners = {
            1: run_phase1,
            2: run_phase2,
            3: run_phase3,
            4: run_phase4
        }
        
        overall_success = True
        phases_completed_this_run: List[int] = []
        start_time = time.time()
        
        for phase in phases_to_run:
            phase_start = time.time()
            
            # Check memory before each phase
            memory_before = optimizer.memory_manager.get_memory_usage()
            logger.info(f"Memory before Phase {phase}: {memory_before['rss_gb']:.2f} GB")
            
            if phase in phase_runners:
                success = phase_runners[phase](sample_size, args.skip_validation)
                
                if not success:
                    logger.error(f"Phase {phase} failed - stopping pipeline")
                    overall_success = False
                    break
                
                phase_time = time.time() - phase_start
                memory_after = optimizer.memory_manager.get_memory_usage()
                memory_delta = memory_after['rss_gb'] - memory_before['rss_gb']
                
                logger.info(f"Phase {phase} completed in {phase_time:.2f} seconds")
                logger.info(f"Memory after Phase {phase}: {memory_after['rss_gb']:.2f} GB (Δ{memory_delta:+.2f} GB)")
                phases_completed_this_run.append(phase)

                # After-phase counts: load corresponding output and log buildings/persons
                try:
                    out_path = _phase_output_path(phase, config)
                    if out_path.exists():
                        out_df = pd.read_pickle(out_path)
                        out_df = _safe_upcast_small_uints(out_df)
                        b = len(out_df)
                        p = _sum_persons(out_df)
                        if phase == 1:
                            logger.info(f"COUNTS AFTER PHASE 1 - buildings: {b:,} | persons: {p:,}")
                        elif phase == 2:
                            logger.info(f"COUNTS AFTER PHASE 2 - buildings: {b:,} | persons: {p:,}")
                        elif phase == 3:
                            logger.info(f"COUNTS AFTER PHASE 3 - buildings: {b:,} | persons: {p:,}")
                        elif phase == 4:
                            logger.info(f"COUNTS AFTER PHASE 4 - buildings: {b:,} | persons: {p:,}")
                        # If streaming shards exist, also report total buildings from manifest
                        shards_dir = out_path.parent / f"phase{phase}_shards"
                        manifest_path = shards_dir / 'manifest.json'
                        if shards_dir.exists() and manifest_path.exists():
                            try:
                                with open(manifest_path, 'r') as mf:
                                    _m = json.load(mf)
                                total_buildings = _m.get('total_buildings') or _m.get('total') or None
                                n_shards = _m.get('n_shards') or (len(_m.get('files', [])) if isinstance(_m.get('files'), list) else None)
                                if total_buildings:
                                    logger.info(f"Streaming output detected: total buildings across shards: {int(total_buildings):,} (shards: {n_shards})")
                                    logger.info("Note: counts above are from sample file; shard manifest reflects full output.")
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Could not compute counts after Phase {phase}: {e}")
                
                # Memory cleanup between phases
                if phase < max(phases_to_run):  # Not the last phase
                    logger.debug("Cleaning up memory before next phase...")
                    optimizer.memory_manager.force_cleanup(level=2)
                    gc.collect()
                    
                    # Check if we can continue
                    if not optimizer.memory_manager.check_can_continue(estimated_next_gb=0.5):
                        logger.error("Insufficient memory to continue to next phase")
                        overall_success = False
                        break
            else:
                logger.error(f"Invalid phase number: {phase}")
                overall_success = False
                break
        
        # Final summary
        total_time = time.time() - start_time
        
        # Count phases that were actually completed successfully in this run
        phases_completed = phases_completed_this_run.copy()
        
        performance_metrics = {
            'total_time_seconds': total_time,
            'phases_completed': len(phases_completed),
            'phases_attempted': phases_to_run,
            'sample_size': sample_size or 'full',
            'success': overall_success
        }
        
        create_performance_summary('pipeline', performance_metrics, logger)
        
        # Print memory usage report
        logger.info(optimizer.memory_manager.get_memory_report())
        
        # Add performance projection for 1.4M buildings
        if sample_size and sample_size > 0:
            buildings_processed = performance_metrics.get('phases_completed', 0) * sample_size
            if buildings_processed > 0 and total_time > 0:
                throughput = buildings_processed / total_time
                projected_time = 1400000 / throughput / 3600
                # Avoid non-ASCII characters to prevent Windows console encoding issues
                logger.info("\nPERFORMANCE PROJECTION FOR 1.4M BUILDINGS:")
                logger.info(f"   Current throughput: {throughput:.1f} buildings/second")
                logger.info(f"   Estimated time: {projected_time:.1f} hours")
                
                # Memory projection
                current_memory = optimizer.memory_manager.get_memory_usage()
                memory_per_building = current_memory['rss_gb'] / max(sample_size, 1)
                projected_memory = memory_per_building * 1400000
                logger.info(f"   Estimated memory: {projected_memory:.1f} GB")
                
                if projected_time <= 1 and projected_memory < optimizer.memory_manager.memory_limit_gb:
                    logger.info("   OK: Meets 1-hour and memory targets")
                else:
                    if projected_time > 1:
                        speedup_needed = projected_time
                        logger.info(f"   Needs {speedup_needed:.1f}x speedup to meet 1-hour target")
                        logger.info(f"   Tip: Try --workers {int(optimizer.n_workers * speedup_needed)}")
                    if projected_memory > optimizer.memory_manager.memory_limit_gb:
                        logger.info("   Memory usage would exceed limit")
                        logger.info(f"   Tip: Try --streaming --batch-size {int(sample_size * 0.1)}")
        
        if overall_success:
            logger.info("\nPipeline completed successfully.")
            if optimizer.streaming:
                logger.info("   Running in streaming mode - optimized for memory usage")
            sys.exit(0)
        else:
            logger.error("\nPipeline failed.")
            sys.exit(1)
            
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()