"""
Memory management utilities for the PUMS Enrichment Pipeline.

This module provides tools for monitoring and managing memory usage throughout
the pipeline to prevent crashes and optimize performance.
"""

import gc
import os
import psutil
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

logger = logging.getLogger('pums_enrichment.memory')


class MemoryManager:
    """Manages memory usage throughout the pipeline."""
    
    def __init__(self, memory_limit_gb: Optional[float] = None, 
                 auto_cleanup: bool = True,
                 aggressive_gc: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            memory_limit_gb: Maximum memory usage in GB (None for auto-detect)
            auto_cleanup: Whether to automatically trigger cleanup on high memory
            aggressive_gc: Whether to use aggressive garbage collection
        """
        self.process = psutil.Process(os.getpid())
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Set memory limit (default to 80% of available RAM)
        if memory_limit_gb is None:
            self.memory_limit_gb = self.total_memory_gb * 0.8
        else:
            self.memory_limit_gb = min(memory_limit_gb, self.total_memory_gb * 0.9)
        
        self.auto_cleanup = auto_cleanup
        self.aggressive_gc = aggressive_gc
        self.cleanup_threshold = 0.7  # Trigger cleanup at 70% of limit
        
        # Track memory usage over time
        self.memory_history = []
        self.phase_memory = {}
        
        logger.info(f"Memory Manager initialized:")
        logger.info(f"  Total RAM: {self.total_memory_gb:.1f} GB")
        logger.info(f"  Memory limit: {self.memory_limit_gb:.1f} GB")
        logger.info(f"  Auto cleanup: {self.auto_cleanup}")
        logger.info(f"  Aggressive GC: {self.aggressive_gc}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'rss_gb': mem_info.rss / (1024**3),
            'vms_gb': mem_info.vms / (1024**3),
            'percent': self.process.memory_percent(),
            'available_gb': virtual_mem.available / (1024**3),
            'system_percent': virtual_mem.percent
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        usage = self.get_memory_usage()
        return usage['rss_gb'] > (self.memory_limit_gb * self.cleanup_threshold)
    
    def force_cleanup(self, level: int = 1):
        """
        Force memory cleanup.
        
        Args:
            level: Cleanup aggressiveness (1=light, 2=medium, 3=aggressive)
        """
        logger.info(f"Forcing memory cleanup (level {level})")
        
        # Level 1: Basic garbage collection
        gc.collect()
        
        if level >= 2:
            # Level 2: Full garbage collection
            gc.collect(2)
            
            # Clear matplotlib cache if exists
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except ImportError:
                pass
        
        if level >= 3:
            # Level 3: Aggressive cleanup
            # Force collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Clear any dataframe caches
            pd.set_option('mode.chained_assignment', None)
            
            # Trim memory pools
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass
        
        # Log memory after cleanup
        usage_after = self.get_memory_usage()
        logger.info(f"Memory after cleanup: {usage_after['rss_gb']:.2f} GB ({usage_after['percent']:.1f}%)")
    
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring memory during an operation."""
        class MemoryMonitor:
            def __init__(self, manager, name):
                self.manager = manager
                self.name = name
                self.start_memory = None
            
            def __enter__(self):
                self.start_memory = self.manager.get_memory_usage()
                logger.debug(f"Starting {self.name} - Memory: {self.start_memory['rss_gb']:.2f} GB")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_memory = self.manager.get_memory_usage()
                delta = end_memory['rss_gb'] - self.start_memory['rss_gb']
                
                logger.debug(
                    f"Finished {self.name} - Memory: {end_memory['rss_gb']:.2f} GB (delta {delta:+.2f} GB)"
                )
                
                # Check if cleanup needed
                if self.manager.auto_cleanup and self.manager.check_memory_pressure():
                    logger.warning(f"High memory usage detected after {self.name}")
                    self.manager.force_cleanup(level=2)
        
        return MemoryMonitor(self, operation_name)
    
    def optimize_dataframe(self, df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            deep: Whether to do deep optimization (slower but more effective)
        """
        if df is None or df.empty:
            return df
            
        initial_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Optimize numeric columns
        for col in df.columns:
            if col not in df.columns:  # Skip if column was deleted
                continue
                
            col_type = df[col].dtype
            
            # Skip non-numeric types for numeric optimization
            if col_type == 'object' or str(col_type) == 'category':
                continue
            
            if str(col_type).startswith('int') or str(col_type).startswith('float'):
                try:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    # Integer optimization
                    if str(col_type)[:3] == 'int':
                        if pd.isna(c_min) or pd.isna(c_max):
                            continue
                            
                        if c_min >= 0:  # Unsigned (non-negative) integers
                            # Use uint16 as the minimum to prevent overflow in downstream ops
                            # (avoids "Python integer XXXX out of bounds for uint8")
                            if c_max < 65536:
                                df[col] = df[col].astype(np.uint16)
                            elif c_max < 4294967296:
                                df[col] = df[col].astype(np.uint32)
                        else:  # Signed integers
                            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                                df[col] = df[col].astype(np.int8)
                            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                                df[col] = df[col].astype(np.int16)
                            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                                df[col] = df[col].astype(np.int32)
                    
                    # Float optimization
                    elif str(col_type)[:5] == 'float':
                        if pd.isna(c_min) or pd.isna(c_max):
                            continue
                            
                        # Skip float16 as it causes issues with pandas operations
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                            
                except (TypeError, ValueError) as e:
                    logger.debug(f"Could not optimize column {col}: {e}")
                    continue
        
        # Optimize string columns to categories if low cardinality
        if deep:
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Skip columns that contain lists or other complex objects
                    sample_val = df[col].iloc[0] if len(df[col]) > 0 else None
                    if sample_val is not None and (isinstance(sample_val, (list, dict, tuple))):
                        continue
                        
                    num_unique = df[col].nunique()
                    num_total = len(df[col])
                    
                    # Convert to category if less than 50% unique values
                    if num_unique / num_total < 0.5 and num_unique < 1000:
                        df[col] = df[col].astype('category')
                except:
                    continue
        
        final_memory = df.memory_usage(deep=True).sum() / (1024**2)
        reduction_pct = 100 * (initial_memory - final_memory) / initial_memory if initial_memory > 0 else 0
        
        if reduction_pct > 0:
            logger.info(f"DataFrame optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction_pct:.1f}% reduction)")
        
        return df
    
    def track_phase_memory(self, phase: str, memory_gb: float):
        """Track memory usage for a specific phase."""
        self.phase_memory[phase] = memory_gb
        self.memory_history.append({
            'phase': phase,
            'memory_gb': memory_gb,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_memory_report(self) -> str:
        """Generate a memory usage report."""
        current = self.get_memory_usage()
        
        report = [
            "\n" + "="*60,
            "MEMORY USAGE REPORT",
            "="*60,
            f"Current Memory: {current['rss_gb']:.2f} GB ({current['percent']:.1f}%)",
            f"Available: {current['available_gb']:.2f} GB",
            f"Memory Limit: {self.memory_limit_gb:.2f} GB",
            f"System Usage: {current['system_percent']:.1f}%",
            ""
        ]
        
        if self.phase_memory:
            report.append("Phase Memory Usage:")
            for phase, memory in self.phase_memory.items():
                report.append(f"  {phase}: {memory:.2f} GB")
        
        return "\n".join(report)
    
    def check_can_continue(self, estimated_next_gb: float = 0) -> bool:
        """
        Check if there's enough memory to continue.
        
        Args:
            estimated_next_gb: Estimated memory needed for next operation
        """
        current = self.get_memory_usage()
        projected = current['rss_gb'] + estimated_next_gb
        
        if projected > self.memory_limit_gb:
            logger.error(f"Insufficient memory: Current {current['rss_gb']:.2f} GB + "
                        f"Estimated {estimated_next_gb:.2f} GB > Limit {self.memory_limit_gb:.2f} GB")
            
            # Try aggressive cleanup
            if self.auto_cleanup:
                logger.info("Attempting aggressive memory cleanup...")
                self.force_cleanup(level=3)
                
                # Check again
                current = self.get_memory_usage()
                projected = current['rss_gb'] + estimated_next_gb
                
                if projected > self.memory_limit_gb:
                    logger.error("Still insufficient memory after cleanup")
                    return False
                else:
                    logger.info("Memory cleanup successful, continuing...")
                    return True
            
            return False
        
        return True


def memory_efficient(estimated_gb: float = 0.1):
    """
    Decorator to make functions memory-efficient.
    
    Args:
        estimated_gb: Estimated memory usage of the function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create global memory manager
            if not hasattr(wrapper, '_memory_manager'):
                wrapper._memory_manager = MemoryManager()
            
            manager = wrapper._memory_manager
            
            # Check if we can run
            if not manager.check_can_continue(estimated_gb):
                raise MemoryError(f"Insufficient memory to run {func.__name__}")
            
            # Monitor the operation
            with manager.monitor_operation(func.__name__):
                result = func(*args, **kwargs)
            
            # Cleanup if needed
            if manager.check_memory_pressure():
                manager.force_cleanup(level=1)
            
            return result
        
        return wrapper
    return decorator


def clear_dataframe_list(df_list: list):
    """Clear a list of DataFrames from memory."""
    for df in df_list:
        if isinstance(df, pd.DataFrame):
            del df
    df_list.clear()
    gc.collect()


def estimate_dataframe_memory(rows: int, cols: int, dtype_mix: Dict[str, float] = None) -> float:
    """
    Estimate memory usage for a DataFrame.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        dtype_mix: Dictionary of dtype -> percentage (default assumes mixed types)
    """
    if dtype_mix is None:
        dtype_mix = {
            'float64': 0.3,
            'int64': 0.2,
            'object': 0.3,
            'category': 0.2
        }
    
    bytes_per_element = {
        'float64': 8,
        'float32': 4,
        'float16': 2,
        'int64': 8,
        'int32': 4,
        'int16': 2,
        'int8': 1,
        'uint8': 1,
        'object': 50,  # Estimated average for strings
        'category': 2   # Estimated for categorical
    }
    
    total_bytes = 0
    for dtype, percentage in dtype_mix.items():
        if dtype in bytes_per_element:
            total_bytes += rows * (cols * percentage) * bytes_per_element[dtype]
    
    return total_bytes / (1024**3)  # Convert to GB


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(memory_limit_gb: Optional[float] = None) -> MemoryManager:
    """Get or create the global memory manager."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(memory_limit_gb=memory_limit_gb)
    
    return _global_memory_manager