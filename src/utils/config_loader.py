"""
Configuration loader for PUMS Enrichment Pipeline.

This module provides a singleton configuration loader that reads settings
from config.yaml and provides easy access throughout the application.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import threading


class ConfigurationError(Exception):
    """Raised when there's an issue with configuration."""
    pass


class Config:
    """Thread-safe singleton configuration loader."""
    
    _instance = None
    _config = None
    _lock = threading.Lock()
    
    def __new__(cls):
        # Double-checked locking pattern for thread safety
        if cls._instance is None:
            with cls._lock:
                # Check again after acquiring lock
                if cls._instance is None:
                    cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration loader."""
        # Use lock to ensure thread-safe initialization
        with self._lock:
            if self._config is None:
                self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file with comprehensive error handling."""
        # Find config file relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}")
        except PermissionError as e:
            raise ConfigurationError(f"Permission denied reading configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected error loading configuration: {e}")
        
        # Check if config is empty or None
        if not self._config:
            raise ConfigurationError("Configuration file is empty or invalid")
        
        # Validate required sections
        required_sections = ['data_paths', 'processing', 'phase1', 'matching', 'logging', 'validation']
        missing_sections = [s for s in required_sections if s not in self._config]
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")
        
        # Convert relative paths to absolute paths
        self._resolve_paths()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _resolve_paths(self):
        """Convert relative paths to absolute paths based on project root."""
        project_root = Path(__file__).parent.parent.parent
        
        # Resolve data paths
        for key, value in self._config['data_paths'].items():
            if isinstance(value, str) and not os.path.isabs(value):
                self._config['data_paths'][key] = str(project_root / value)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides for configuration values."""
        # Override sample size if environment variable is set
        if 'PUMS_SAMPLE_SIZE' in os.environ:
            try:
                self._config['processing']['default_sample_size'] = int(os.environ['PUMS_SAMPLE_SIZE'])
            except ValueError:
                logging.warning(f"Invalid PUMS_SAMPLE_SIZE environment variable: {os.environ['PUMS_SAMPLE_SIZE']}")
        
        # Override random seed if set
        if 'PUMS_RANDOM_SEED' in os.environ:
            try:
                self._config['processing']['random_seed'] = int(os.environ['PUMS_RANDOM_SEED'])
            except ValueError:
                logging.warning(f"Invalid PUMS_RANDOM_SEED environment variable: {os.environ['PUMS_RANDOM_SEED']}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'processing.sample_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_data_path(self, path_key: str) -> str:
        """
        Get a data path from configuration.
        
        Args:
            path_key: Key for the data path (e.g., 'pums_household')
            
        Returns:
            Absolute path to the data file
            
        Raises:
            ConfigurationError: If path key not found
        """
        if path_key not in self._config['data_paths']:
            raise ConfigurationError(f"Data path not found in configuration: {path_key}")
        
        return self._config['data_paths'][path_key]
    
    def get_phase1_columns(self, data_type: str) -> list:
        """
        Get column list for Phase 1 data.
        
        Args:
            data_type: Either 'household' or 'person'
            
        Returns:
            List of column names to keep
            
        Raises:
            ValueError: If invalid data_type
        """
        if data_type not in ['household', 'person']:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'household' or 'person'")
        
        return self._config['phase1'][f'{data_type}_columns']
    
    def get_sample_size(self, override: Optional[int] = None) -> Optional[int]:
        """
        Get sample size for processing.
        
        Args:
            override: Optional override value
            
        Returns:
            Sample size or None for full data
        """
        if override is not None:
            return override
        
        return self._config['processing']['default_sample_size']
    
    def get_random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return self._config['processing']['random_seed']
    
    def get_n_jobs(self) -> Optional[int]:
        """Get number of jobs for parallel processing."""
        n_jobs = self._config['processing']['n_jobs']
        if n_jobs is None:
            # Use all available cores
            import multiprocessing
            return multiprocessing.cpu_count()
        return n_jobs
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config['logging']
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self._config['validation']
    
    def get_matching_config(self) -> Dict[str, Any]:
        """Get matching configuration for Phase 2-3."""
        return self._config['matching']
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()


def get_config() -> Config:
    """
    Get the singleton configuration instance.
    
    Returns:
        Config instance
    """
    return Config()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("Configuration loaded successfully!")
        print(f"Sample size: {config.get_sample_size()}")
        print(f"Random seed: {config.get_random_seed()}")
        print(f"PUMS household data: {config.get_data_path('pums_household')}")
        print(f"Number of household columns: {len(config.get_phase1_columns('household'))}")
        print(f"Number of person columns: {len(config.get_phase1_columns('person'))}")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")