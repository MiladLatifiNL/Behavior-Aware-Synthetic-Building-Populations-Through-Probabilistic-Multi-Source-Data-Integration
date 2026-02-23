"""
Logging setup for PUMS Enrichment Pipeline.

This module provides centralized logging configuration with support for:
- Phase-specific log files
- Performance metrics logging
- Memory usage tracking
- Log rotation
"""

import logging
import logging.handlers
import os
import sys
import psutil
import time
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, Any
import io

from .config_loader import get_config


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record):
        """Add memory usage and process info to log record."""
        process = psutil.Process()
        record.memory_mb = process.memory_info().rss / 1024 / 1024
        record.cpu_percent = process.cpu_percent(interval=0.1)
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class SafeConsoleHandler(logging.StreamHandler):
    """Stream handler that avoids UnicodeEncodeError on Windows consoles.

    If the console cannot encode certain characters (e.g., emojis), this handler
    will replace unencodable characters rather than raising an exception.
    """

    def __init__(self, stream=None):
        # Default to stdout
        if stream is None:
            stream = sys.stdout
        # Wrap the underlying buffer with the same encoding but errors='replace'
        # to prevent crashes on cp1252 consoles.
        try:
            encoding = getattr(stream, "encoding", None) or "utf-8"
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                safe_stream = io.TextIOWrapper(buffer, encoding=encoding, errors="replace")
                super().__init__(safe_stream)
            else:
                # Fallback: use parent implementation; we'll handle replacement in emit
                super().__init__(stream)
        except Exception:
            super().__init__(stream)

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                # Replace message with a safe-encoded version
                msg = self.format(record)
                stream = self.stream
                encoding = getattr(stream, "encoding", None) or "utf-8"
                # Encode with replacement then decode back so write() accepts it
                safe_msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_msg + getattr(self, 'terminator', "\n"))
            except Exception:
                # Last resort: drop the message silently to avoid crashing
                pass


def setup_logging(phase: str = "main", console: bool = True) -> logging.Logger:
    """
    Set up logging for a specific phase.
    
    Args:
        phase: Phase name for log file (e.g., 'phase1', 'main')
        console: Whether to add console handler
        
    Returns:
        Logger instance
    """
    config = get_config()
    log_config = config.get_logging_config()
    
    # Create logs directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"pums_enrichment.{phase}")
    logger.setLevel(getattr(logging, log_config['level']))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with rotation
    log_file = log_dir / f"{phase}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config['max_bytes'],
        backupCount=log_config['backup_count'],
        encoding="utf-8"
    )
    
    # Format with performance metrics if enabled
    if log_config.get('log_performance', True):
        file_handler.addFilter(PerformanceFilter())
        file_format = "%(asctime)s - %(name)s - %(levelname)s - [Mem: %(memory_mb).1fMB, CPU: %(cpu_percent).1f%%] - %(message)s"
    else:
        file_format = log_config['format']
    
    file_formatter = logging.Formatter(file_format, datefmt=log_config['datefmt'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = SafeConsoleHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        
        # Use colored output if terminal supports it
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance (uses function's module logger if None)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            logger.info(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"Error in {func.__name__} after {elapsed_time:.2f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_memory_usage(logger: Optional[logging.Logger] = None):
    """
    Decorator to log memory usage before and after function execution.
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_diff = mem_after - mem_before
            
            logger.info(
                f"{func.__name__} memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB "
                f"(delta {mem_diff:+.1f}MB)"
            )
            
            return result
        
        return wrapper
    return decorator


def log_phase_progress(phase: str, step: str, current: int, total: int, 
                      logger: Optional[logging.Logger] = None):
    """
    Log progress for a phase processing step.
    
    Args:
        phase: Phase name
        step: Step description
        current: Current item number
        total: Total items
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger(f"pums_enrichment.{phase}")
    
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"{phase} - {step}: {current}/{total} ({percentage:.1f}%)")


def create_performance_summary(phase: str, metrics: Dict[str, Any], 
                             logger: Optional[logging.Logger] = None):
    """
    Log a performance summary for a phase.
    
    Args:
        phase: Phase name
        metrics: Dictionary of performance metrics
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger(f"pums_enrichment.{phase}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{phase.upper()} PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info(f"{'='*60}\n")


# Matching-specific logging utilities
def setup_matching_logger(phase: str) -> logging.Logger:
    """
    Set up specialized logger for matching diagnostics.
    
    Args:
        phase: Phase name (e.g., 'phase2', 'phase3')
        
    Returns:
        Logger for matching diagnostics
    """
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs" / "matching"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"matching.{phase}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # Detailed file handler for matching diagnostics
    log_file = log_dir / f"{phase}_matching_diagnostics.log"
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_matching_metrics(logger: logging.Logger, iteration: int, 
                        m_probs: Dict[str, float], u_probs: Dict[str, float],
                        convergence_metric: float):
    """
    Log matching algorithm metrics.
    
    Args:
        logger: Logger instance
        iteration: Current iteration number
        m_probs: Match probabilities
        u_probs: Non-match probabilities
        convergence_metric: Convergence metric value
    """
    logger.debug(f"\nIteration {iteration}:")
    logger.debug(f"Convergence metric: {convergence_metric:.6f}")
    logger.debug("M-probabilities:")
    for field, prob in sorted(m_probs.items()):
        logger.debug(f"  {field}: {prob:.4f}")
    logger.debug("U-probabilities:")
    for field, prob in sorted(u_probs.items()):
        logger.debug(f"  {field}: {prob:.4f}")


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test decorators
    @log_execution_time(logger)
    @log_memory_usage(logger)
    def test_function():
        import time
        time.sleep(0.5)
        # Allocate some memory
        data = [0] * 1000000
        return len(data)
    
    result = test_function()
    logger.info(f"Test function returned: {result}")
    
    # Test performance summary
    metrics = {
        "Total records processed": 1000,
        "Processing time (seconds)": 45.23,
        "Memory usage (MB)": 256.7,
        "Match rate": 0.956
    }
    create_performance_summary("test", metrics, logger)