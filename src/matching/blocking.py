"""
Blocking strategies for efficient record linkage.

This module implements blocking techniques to reduce the computational
complexity of record linkage from O(n*m) to manageable levels while
maintaining high pair completeness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union
import logging
from collections import defaultdict
import jellyfish

logger = logging.getLogger(__name__)


class BlockingStrategy:
    """Base class for blocking strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.block_statistics = {}
    
    def create_blocks(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
        """
        Create blocks of record indices.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        raise NotImplementedError


class ExactBlockingStrategy(BlockingStrategy):
    """Blocking based on exact matching of specified fields."""
    
    def __init__(self, blocking_fields: List[str], name: str = "exact"):
        super().__init__(name)
        self.blocking_fields = blocking_fields
    
    def create_blocks(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
        """Create blocks using exact field matching."""
        blocks = defaultdict(lambda: {'df1': [], 'df2': []})
        
        # Create blocking keys for df1
        for idx, row in df1.iterrows():
            block_key = self._create_block_key(row)
            if block_key:
                blocks[block_key]['df1'].append(idx)
        
        # Create blocking keys for df2
        for idx, row in df2.iterrows():
            block_key = self._create_block_key(row)
            if block_key:
                blocks[block_key]['df2'].append(idx)
        
        # Remove blocks with records from only one dataset
        valid_blocks = {k: v for k, v in blocks.items() 
                       if len(v['df1']) > 0 and len(v['df2']) > 0}
        
        # Calculate statistics
        self._calculate_statistics(valid_blocks)
        
        return valid_blocks
    
    def _create_block_key(self, row: pd.Series) -> Optional[str]:
        """Create blocking key from row data."""
        key_parts = []
        for field in self.blocking_fields:
            if field in row and pd.notna(row[field]):
                key_parts.append(str(row[field]).upper())
            else:
                return None  # Skip if any field is missing
        
        return '|'.join(key_parts)
    
    def _calculate_statistics(self, blocks: Dict):
        """Calculate blocking statistics."""
        total_df1 = sum(len(v['df1']) for v in blocks.values())
        total_df2 = sum(len(v['df2']) for v in blocks.values())
        total_pairs = sum(len(v['df1']) * len(v['df2']) for v in blocks.values())
        
        self.block_statistics = {
            'num_blocks': len(blocks),
            'total_df1_records': total_df1,
            'total_df2_records': total_df2,
            'total_candidate_pairs': total_pairs,
            'avg_block_size': total_pairs / len(blocks) if blocks else 0
        }


class SoundexBlockingStrategy(BlockingStrategy):
    """Blocking based on phonetic encoding of name fields."""
    
    def __init__(self, name_field: str, additional_fields: Optional[List[str]] = None):
        super().__init__("soundex")
        self.name_field = name_field
        self.additional_fields = additional_fields or []
    
    def create_blocks(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
        """Create blocks using soundex encoding."""
        blocks = defaultdict(lambda: {'df1': [], 'df2': []})
        
        # Process df1
        for idx, row in df1.iterrows():
            block_key = self._create_soundex_key(row)
            if block_key:
                blocks[block_key]['df1'].append(idx)
        
        # Process df2
        for idx, row in df2.iterrows():
            block_key = self._create_soundex_key(row)
            if block_key:
                blocks[block_key]['df2'].append(idx)
        
        # Filter valid blocks
        valid_blocks = {k: v for k, v in blocks.items() 
                       if len(v['df1']) > 0 and len(v['df2']) > 0}
        
        self._calculate_statistics(valid_blocks)
        
        return valid_blocks
    
    def _create_soundex_key(self, row: pd.Series) -> Optional[str]:
        """Create soundex-based blocking key."""
        if self.name_field not in row or pd.isna(row[self.name_field]):
            return None
        
        try:
            # Get soundex of name
            soundex_code = jellyfish.soundex(str(row[self.name_field]))
            
            # Add additional fields if specified
            key_parts = [soundex_code]
            for field in self.additional_fields:
                if field in row and pd.notna(row[field]):
                    key_parts.append(str(row[field]).upper())
            
            return '|'.join(key_parts)
        except:
            return None
    
    def _calculate_statistics(self, blocks: Dict):
        """Calculate blocking statistics."""
        total_pairs = sum(len(v['df1']) * len(v['df2']) for v in blocks.values())
        
        self.block_statistics = {
            'num_blocks': len(blocks),
            'total_candidate_pairs': total_pairs,
            'avg_block_size': total_pairs / len(blocks) if blocks else 0
        }


class MultipassBlockingStrategy(BlockingStrategy):
    """Combine multiple blocking strategies to improve coverage."""
    
    def __init__(self, strategies: List[BlockingStrategy]):
        super().__init__("multipass")
        self.strategies = strategies
    
    def create_blocks(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
        """Create blocks using multiple strategies."""
        all_pairs = set()
        combined_blocks = {}
        
        for i, strategy in enumerate(self.strategies):
            blocks = strategy.create_blocks(df1, df2)
            
            if blocks:
                logger.debug(f"Strategy {i} ({strategy.name}) produced {len(blocks)} blocks")
            else:
                logger.debug(f"Strategy {i} ({strategy.name}) produced no blocks")
            
            # Collect all pairs from this strategy
            for block_key, indices in blocks.items():
                # Create unique key for combined blocks
                combined_key = f"{strategy.name}_{i}_{block_key}"
                combined_blocks[combined_key] = indices
                
                # Track unique pairs
                for idx1 in indices['df1']:
                    for idx2 in indices['df2']:
                        all_pairs.add((idx1, idx2))
        
        # Calculate overall statistics
        self.block_statistics = {
            'num_strategies': len(self.strategies),
            'total_unique_pairs': len(all_pairs),
            'total_blocks': len(combined_blocks),
            'strategy_stats': {s.name: s.block_statistics for s in self.strategies}
        }
        
        return combined_blocks


def create_standard_blocks(df1: pd.DataFrame, df2: pd.DataFrame, 
                          config: Dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Create standard blocking configuration for PUMS-RECS matching.
    
    Args:
        df1: PUMS buildings DataFrame
        df2: RECS templates DataFrame
        config: Blocking configuration
        
    Returns:
        Dictionary of blocks
    """
    # Log diagnostic information about data distribution
    logger.info("Creating blocking strategies for PUMS-RECS matching")
    logger.info(f"PUMS data: {len(df1)} records")
    logger.info(f"RECS data: {len(df2)} records")
    
    # Use config if provided, otherwise use defaults
    if config and 'strategies' in config:
        strategies = []
        for strategy_config in config['strategies']:
            if 'fields' in strategy_config:
                fields = strategy_config['fields']
                # Only use fields that exist in both datasets
                valid_fields = [f for f in fields if f in df1.columns and f in df2.columns]
                if valid_fields:
                    strategies.append(ExactBlockingStrategy(valid_fields))
                else:
                    logger.warning(f"No valid fields found for blocking strategy: {fields}")
    else:
        # Enhanced multi-level blocking strategies with diagnostic logging
        strategies = []
        
        # Track which strategies are attempted and why they fail
        attempted_strategies = []
        
        # Level 1: Climate + socioeconomic blocking (most important for energy)
        if 'climate_zone' in df1.columns and 'climate_zone' in df2.columns:
            if 'income_tercile' in df1.columns and 'income_tercile' in df2.columns:
                strategies.append(ExactBlockingStrategy(['climate_zone', 'income_tercile']))
                attempted_strategies.append('climate_zone + income_tercile')
            if 'ses_category' in df1.columns and 'ses_category' in df2.columns:
                strategies.append(ExactBlockingStrategy(['climate_zone', 'ses_category']))
                attempted_strategies.append('climate_zone + ses_category')
        
        # Level 2: Geographic + household characteristics
        if 'REGION' in df1.columns and 'REGION' in df2.columns:
            # Check if values overlap
            pums_regions = set(df1['REGION'].dropna().unique())
            recs_regions = set(df2['REGION'].dropna().unique())
            overlap = pums_regions & recs_regions
            if overlap:
                logger.info(f"REGION overlap: {overlap} (PUMS: {pums_regions}, RECS: {recs_regions})")
                strategies.append(ExactBlockingStrategy(['REGION', 'income_quintile']))
                attempted_strategies.append('REGION + income_quintile')
                strategies.append(ExactBlockingStrategy(['REGION', 'hh_size_3cat']))
                attempted_strategies.append('REGION + hh_size_3cat')
                if 'age_3cat' in df1.columns and 'age_3cat' in df2.columns:
                    strategies.append(ExactBlockingStrategy(['REGION', 'age_3cat']))
                    attempted_strategies.append('REGION + age_3cat')
            else:
                logger.warning(f"No REGION overlap between PUMS {pums_regions} and RECS {recs_regions}")
        
        if 'DIVISION' in df1.columns and 'DIVISION' in df2.columns:
            # Check if values overlap
            pums_divs = set(df1['DIVISION'].dropna().unique())
            recs_divs = set(df2['DIVISION'].dropna().unique())
            overlap = pums_divs & recs_divs
            if overlap:
                logger.info(f"DIVISION overlap: {overlap} (PUMS: {pums_divs}, RECS: {recs_divs})")
                strategies.append(ExactBlockingStrategy(['DIVISION', 'income_tercile']))
                attempted_strategies.append('DIVISION + income_tercile')
                strategies.append(ExactBlockingStrategy(['DIVISION', 'household_size_cat']))
                attempted_strategies.append('DIVISION + household_size_cat')
            else:
                logger.warning(f"No DIVISION overlap between PUMS {pums_divs} and RECS {recs_divs}")
        
        # Level 3: Building characteristics blocking
        if 'age_3cat' in df1.columns and 'age_3cat' in df2.columns:
            if 'tenure_type' in df1.columns and 'tenure_type' in df2.columns:
                strategies.append(ExactBlockingStrategy(['age_3cat', 'tenure_type']))
                attempted_strategies.append('age_3cat + tenure_type')
            if 'rooms_3cat' in df1.columns and 'rooms_3cat' in df2.columns:
                strategies.append(ExactBlockingStrategy(['age_3cat', 'rooms_3cat']))
                attempted_strategies.append('age_3cat + rooms_3cat')
        
        # Level 4: Energy vulnerability blocking
        if 'high_energy_vulnerable' in df1.columns and 'high_energy_vulnerable' in df2.columns:
            strategies.append(ExactBlockingStrategy(['high_energy_vulnerable', 'income_tercile']))
            attempted_strategies.append('high_energy_vulnerable + income_tercile')
        
        # Level 5: Non-geographic fallback strategies (IMPORTANT - these should work!)
        if 'income_quintile' in df1.columns and 'income_quintile' in df2.columns:
            strategies.append(ExactBlockingStrategy(['income_quintile']))
            attempted_strategies.append('income_quintile only')
        if 'household_size_cat' in df1.columns and 'household_size_cat' in df2.columns:
            strategies.append(ExactBlockingStrategy(['household_size_cat']))
            attempted_strategies.append('household_size_cat only')
        if 'income_tercile' in df1.columns and 'income_tercile' in df2.columns:
            strategies.append(ExactBlockingStrategy(['income_tercile']))
            attempted_strategies.append('income_tercile only')
        if 'hh_size_3cat' in df1.columns and 'hh_size_3cat' in df2.columns:
            strategies.append(ExactBlockingStrategy(['hh_size_3cat']))
            attempted_strategies.append('hh_size_3cat only')
        
        # Level 6: Urban/rural with other features  
        if 'urban_rural' in df1.columns and 'urban_rural' in df2.columns:
            if 'building_type_simple' in df1.columns and 'building_type_simple' in df2.columns:
                strategies.append(ExactBlockingStrategy(['urban_rural', 'building_type_simple']))
            strategies.append(ExactBlockingStrategy(['urban_rural']))
        
        # Level 7: Efficiency-based blocking (for energy matching)
        if 'efficiency_proxy' in df1.columns and 'efficiency_proxy' in df2.columns:
            if 'climate_zone' in df1.columns and 'climate_zone' in df2.columns:
                strategies.append(ExactBlockingStrategy(['efficiency_proxy', 'climate_zone']))
            strategies.append(ExactBlockingStrategy(['efficiency_proxy']))
            attempted_strategies.append('efficiency_proxy only')
        
        logger.info(f"Attempting {len(attempted_strategies)} blocking strategies: {attempted_strategies}")
    
    if not strategies:
        logger.error("No valid blocking strategies could be created!")
        return {}
    
    # Combine strategies
    multipass = MultipassBlockingStrategy(strategies)
    
    return multipass.create_blocks(df1, df2)


def evaluate_blocking_coverage(blocks: Dict, true_matches: Optional[pd.DataFrame] = None,
                             df1_size: int = None, df2_size: int = None) -> Dict[str, float]:
    """
    Evaluate blocking performance metrics.
    
    Args:
        blocks: Dictionary of blocks
        true_matches: Optional DataFrame with known matches for evaluation
        df1_size: Size of first dataset
        df2_size: Size of second dataset
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate total candidate pairs
    total_pairs = sum(len(v['df1']) * len(v['df2']) for v in blocks.values())
    
    # Calculate reduction ratio
    if df1_size and df2_size:
        total_possible_pairs = df1_size * df2_size
        reduction_ratio = 1 - (total_pairs / total_possible_pairs)
    else:
        reduction_ratio = None
    
    # Calculate pairs completeness if true matches provided
    pairs_completeness = None
    if true_matches is not None:
        covered_matches = 0
        for _, match in true_matches.iterrows():
            idx1, idx2 = match['idx1'], match['idx2']
            
            # Check if this true match is in any block
            for block_indices in blocks.values():
                if idx1 in block_indices['df1'] and idx2 in block_indices['df2']:
                    covered_matches += 1
                    break
        
        pairs_completeness = covered_matches / len(true_matches) if len(true_matches) > 0 else 0
    
    # Calculate block size statistics
    block_sizes = [len(v['df1']) * len(v['df2']) for v in blocks.values()]
    
    metrics = {
        'num_blocks': len(blocks),
        'total_candidate_pairs': total_pairs,
        'reduction_ratio': reduction_ratio,
        'pairs_completeness': pairs_completeness,
        'avg_block_size': np.mean(block_sizes) if block_sizes else 0,
        'median_block_size': np.median(block_sizes) if block_sizes else 0,
        'max_block_size': max(block_sizes) if block_sizes else 0,
        'min_block_size': min(block_sizes) if block_sizes else 0
    }
    
    return metrics


def create_blocking_report(blocks: Dict, df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    """
    Generate a text report about blocking results.
    
    Args:
        blocks: Dictionary of blocks
        df1: First DataFrame
        df2: Second DataFrame
        
    Returns:
        Report string
    """
    metrics = evaluate_blocking_coverage(blocks, df1_size=len(df1), df2_size=len(df2))
    
    report = []
    report.append("=" * 60)
    report.append("BLOCKING STRATEGY REPORT")
    report.append("=" * 60)
    report.append(f"Dataset 1 size: {len(df1):,}")
    report.append(f"Dataset 2 size: {len(df2):,}")
    report.append(f"Total possible pairs: {len(df1) * len(df2):,}")
    report.append("")
    report.append("Blocking Results:")
    report.append(f"  Number of blocks: {metrics['num_blocks']:,}")
    report.append(f"  Candidate pairs: {metrics['total_candidate_pairs']:,}")
    report.append(f"  Reduction ratio: {metrics['reduction_ratio']:.2%}" if metrics['reduction_ratio'] else "  Reduction ratio: N/A")
    report.append("")
    report.append("Block Size Statistics:")
    report.append(f"  Average: {metrics['avg_block_size']:.1f}")
    report.append(f"  Median: {metrics['median_block_size']:.1f}")
    report.append(f"  Min: {metrics['min_block_size']:,}")
    report.append(f"  Max: {metrics['max_block_size']:,}")
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test blocking strategies
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Create sample data
    df1 = pd.DataFrame({
        'id': range(100),
        'STATE': np.random.choice(['01', '02', '03'], 100),
        'income_quintile': np.random.choice(['q1', 'q2', 'q3', 'q4', 'q5'], 100),
        'household_size_cat': np.random.choice(['single', 'couple', 'family'], 100),
        'name': [f'Person{i}' for i in range(100)]
    })
    
    df2 = pd.DataFrame({
        'id': range(50),
        'STATE': np.random.choice(['01', '02', '03'], 50),
        'income_quintile': np.random.choice(['q1', 'q2', 'q3', 'q4', 'q5'], 50),
        'household_size_cat': np.random.choice(['single', 'couple', 'family'], 50),
        'name': [f'Template{i}' for i in range(50)]
    })
    
    print("Testing Blocking Strategies\n")
    
    # Test exact blocking
    print("1. Exact Blocking (STATE + income_quintile):")
    exact_strategy = ExactBlockingStrategy(['STATE', 'income_quintile'])
    exact_blocks = exact_strategy.create_blocks(df1, df2)
    print(f"   Created {len(exact_blocks)} blocks")
    print(f"   Statistics: {exact_strategy.block_statistics}")
    
    # Test multipass blocking
    print("\n2. Multipass Blocking:")
    strategies = [
        ExactBlockingStrategy(['STATE', 'income_quintile']),
        ExactBlockingStrategy(['STATE', 'household_size_cat']),
        ExactBlockingStrategy(['household_size_cat'])
    ]
    multipass = MultipassBlockingStrategy(strategies)
    multi_blocks = multipass.create_blocks(df1, df2)
    print(f"   Total blocks: {multipass.block_statistics['total_blocks']}")
    print(f"   Unique pairs: {multipass.block_statistics['total_unique_pairs']}")
    
    # Generate report
    print("\n3. Blocking Report:")
    print(create_blocking_report(multi_blocks, df1, df2))