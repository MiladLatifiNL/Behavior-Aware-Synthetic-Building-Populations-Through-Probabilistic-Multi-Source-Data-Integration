"""
Phase 4: Weather Integration Module.

This module integrates weather data with buildings from Phase 3, aligning
1-minute resolution weather data with detailed ATUS activity sequences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import time
from collections import defaultdict

from ..utils.config_loader import get_config
from ..utils.logging_setup import setup_logging, log_execution_time, log_memory_usage
from ..validation.data_validator import validate_dataframe
from ..data_loading.weather_loader import (
    load_and_process_state_weather,
    align_weather_with_activity,
    create_weather_summary,
    FIPS_TO_STATE
)
from ..data_loading.atus_loader import load_atus_activity_data

logger = logging.getLogger(__name__)


class Phase4WeatherIntegrator:
    """Phase 4 Weather Integration with activity-level alignment."""
    
    def __init__(self, config: Optional[Dict] = None, cache_size_mb: int = 500, use_references: bool = False):
        """
        Initialize Phase 4 processor.
        
        Args:
            config: Configuration dictionary
            cache_size_mb: Maximum cache size in MB
            use_references: Use weather references instead of embedding
        """
        self.config = config or get_config()
        self.cache_size_mb = cache_size_mb
        self.use_references = use_references
        
        # Get output directories - improved path handling
        phase4_output_path = Path(self.config.get_data_path('phase4_output'))
        
        # Better logic to determine if path is file or directory
        # Check if it's an existing file or has a clear file extension
        if phase4_output_path.exists() and phase4_output_path.is_file():
            self.output_dir = phase4_output_path.parent
        elif phase4_output_path.suffix in ['.pkl', '.pickle', '.csv', '.json']:
            self.output_dir = phase4_output_path.parent
        else:
            # Assume it's a directory path
            self.output_dir = phase4_output_path
            
        # Handle validation directory with same improved logic
        try:
            phase4_validation_path = Path(self.config.get_data_path('phase4_validation'))
            if phase4_validation_path.exists() and phase4_validation_path.is_file():
                self.validation_dir = phase4_validation_path.parent
            elif phase4_validation_path.suffix in ['.html', '.json', '.txt', '.md']:
                self.validation_dir = phase4_validation_path.parent
            else:
                self.validation_dir = phase4_validation_path
        except:
            # Fallback if phase4_validation not in config
            self.validation_dir = Path('data/validation')
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 4 specific paths
        self.phase4_output_path = self.output_dir / 'phase4_final_integrated_buildings.pkl'
        self.phase4_metadata_path = self.output_dir / 'phase4_metadata.json'
        
        # Create weather cache directory if it's configured
        try:
            cache_dir_path = self.config.get('phase4.caching.cache_dir', 'data/processed/weather_cache')
            self.cache_dir = Path(cache_dir_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Weather cache directory ensured at: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Could not create weather cache directory: {e}")
            self.cache_dir = None
        
        # Cache for weather data
        self.weather_cache = {}
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'buildings_count': 0,
            'persons_count': 0,
            'activities_count': 0,
            'states_processed': set(),
            'weather_alignments': 0,
            'cache_hits': 0,
            'buildings_with_weather': 0
        }
    
    @log_execution_time(logger)
    @log_memory_usage(logger)
    def run(self, sample_size: Optional[int] = None, 
            use_cached_weather: bool = True,
            date: Optional[datetime] = None,
            streaming_mode: bool = False,
            batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Execute Phase 4 weather integration.
        
        Args:
            sample_size: Number of buildings to process (None for all)
            use_cached_weather: Whether to use cached interpolated weather data
            date: Specific date to process (default: January 1, 2023)
            streaming_mode: Whether to use memory-efficient streaming
            batch_size: Batch size for streaming mode
            
        Returns:
            DataFrame with weather-integrated buildings
        """
        self.stats['start_time'] = datetime.now()
        logger.info("Starting Phase 4: Weather Integration with Activity Alignment")
        
        # Default to January 1, 2023 if no date specified
        if date is None:
            date = datetime(2023, 1, 1)
        logger.info(f"Processing weather for date: {date.strftime('%Y-%m-%d')}")
        
        try:
            # Streaming path: detect Phase 3 shards and process chunk-by-chunk
            phase3_shards_dir = self.output_dir / 'phase3_shards'
            phase3_manifest = phase3_shards_dir / 'manifest.json'

            if streaming_mode and phase3_shards_dir.exists() and phase3_manifest.exists():
                logger.info("Detected Phase 3 shards - Phase 4 will stream over shards")
                # Load ATUS activities for lookups once
                self.atus_activities = self._load_atus_activities()

                # Prepare shard output directory and manifest
                phase4_shards_dir = self.output_dir / 'phase4_shards'
                phase4_shards_dir.mkdir(parents=True, exist_ok=True)
                phase4_manifest = phase4_shards_dir / 'manifest.json'
                out_files: List[str] = []

                with open(phase3_manifest, 'r') as f:
                    info = json.load(f)
                    files = info.get('files', [])

                processed_total = 0
                max_total = sample_size or float('inf')
                shard_idx = 0
                sample_parts: List[pd.DataFrame] = []

                for fp in files:
                    if processed_total >= max_total:
                        break
                    try:
                        buildings_chunk = pd.read_pickle(fp)
                    except Exception as e:
                        logger.warning(f"Skipping unreadable Phase 3 shard {fp}: {e}")
                        continue

                    remaining = max_total - processed_total
                    if remaining != float('inf') and len(buildings_chunk) > remaining:
                        buildings_chunk = buildings_chunk.head(remaining)

                    # Group by state for efficient weather loading
                    buildings_by_state = self._group_buildings_by_state(buildings_chunk)
                    processed_buildings = []

                    for state_fips, state_buildings in buildings_by_state.items():
                        logger.info(f"[P4 shard {shard_idx}] Processing {len(state_buildings)} buildings in state {state_fips}")
                        # Load weather for this state
                        weather_1min = self._load_state_weather(state_fips, date, use_cached_weather)

                        # Process buildings in this state
                        for idx, building in state_buildings.iterrows():
                            building_with_weather = self._integrate_weather_with_building(
                                building, weather_1min, date
                            )
                            processed_buildings.append(building_with_weather)

                            if batch_size and len(processed_buildings) >= batch_size:
                                # Optional micro-batching hook; we keep per-shard writes for simplicity
                                pass

                    shard_df = pd.DataFrame(processed_buildings) if processed_buildings else pd.DataFrame()
                    out_path = phase4_shards_dir / f'phase4_part_{shard_idx:05d}.pkl'
                    shard_df.to_pickle(out_path)
                    out_files.append(str(out_path))

                    if len(sample_parts) < 3:
                        sample_parts.append(shard_df.head(4000))

                    processed_total += len(shard_df)
                    shard_idx += 1

                # Write manifest and sample
                with open(phase4_manifest, 'w') as f:
                    json.dump({'n_shards': shard_idx, 'files': out_files, 'total_buildings': int(processed_total)}, f, indent=2)

                sample_out = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
                sample_out.to_pickle(self.phase4_output_path)
                with open(self.phase4_metadata_path, 'w') as f:
                    json.dump({'end_time': datetime.now().isoformat(), 'buildings_processed': int(processed_total), 'sharded': True}, f, indent=2)

                logger.info(f"Phase 4 streaming completed over {shard_idx} shards with {processed_total} buildings")
                return sample_out
            else:
                # Standard small-run path: load Phase 3 output at once
                buildings = self._load_phase3_output(sample_size)
                # Load ATUS activities
                self.atus_activities = self._load_atus_activities()

                # Process buildings by state for efficient weather loading
                buildings_by_state = self._group_buildings_by_state(buildings)
                processed_buildings = []
                partial_outputs: List[pd.DataFrame] = []
                for state_fips, state_buildings in buildings_by_state.items():
                    logger.info(f"Processing {len(state_buildings)} buildings in state {state_fips}")
                    weather_1min = self._load_state_weather(state_fips, date, use_cached_weather)
                    for idx, building in state_buildings.iterrows():
                        building_with_weather = self._integrate_weather_with_building(
                            building, weather_1min, date
                        )
                        processed_buildings.append(building_with_weather)

                        if streaming_mode and batch_size and len(processed_buildings) >= batch_size:
                            partial_df = pd.DataFrame(processed_buildings)
                            partial_outputs.append(partial_df)
                            processed_buildings = []
                            if len(self.weather_cache) > 8:
                                try:
                                    self.weather_cache.pop(next(iter(self.weather_cache)))
                                except Exception:
                                    pass

                if processed_buildings:
                    partial_outputs.append(pd.DataFrame(processed_buildings))
                buildings_with_weather = pd.concat(partial_outputs, ignore_index=True) if partial_outputs else pd.DataFrame()

                # Save results
                self.stats['end_time'] = datetime.now()
                self._save_results(buildings_with_weather)

                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
                logger.info(f"Phase 4 completed in {duration:.2f} seconds")
                logger.info(f"Processed {self.stats['buildings_count']} buildings with "
                           f"{self.stats['persons_count']} persons and "
                           f"{self.stats['activities_count']} activities")
                logger.info(f"Weather aligned for {self.stats['weather_alignments']} activities")

                return buildings_with_weather
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {str(e)}")
            raise
    
    def _load_phase3_output(self, sample_size: Optional[int]) -> pd.DataFrame:
        """Load Phase 3 output data.
        If shards are present, concatenate only as much as needed to satisfy sample_size.
        """
        # Primary path
        phase3_path = self.output_dir.parent / 'processed' / 'phase3_pums_recs_atus_buildings.pkl'
        shards_dir = self.output_dir / 'phase3_shards'
        manifest = shards_dir / 'manifest.json'

        # If shards exist, stitch just enough for sample_size
        if shards_dir.exists() and manifest.exists():
            logger.info(f"Detected Phase 3 shards at {shards_dir}; assembling input")
            try:
                with open(manifest, 'r') as f:
                    m = json.load(f)
                files = m.get('files', [])
            except Exception as e:
                logger.warning(f"Failed to read Phase 3 manifest; falling back to single Phase 3 pickle: {e}")
                files = []

            parts: List[pd.DataFrame] = []
            loaded = 0
            cap = sample_size or float('inf')
            for fp in files:
                try:
                    part = pd.read_pickle(fp)
                except Exception as e:
                    logger.warning(f"Skipping unreadable Phase 3 shard {fp}: {e}")
                    continue
                if cap != float('inf'):
                    take = min(len(part), int(cap - loaded))
                    part = part.head(take)
                    loaded += take
                parts.append(part)
                if loaded >= cap:
                    break

            assembled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            logger.info(f"Assembled {len(assembled)} buildings from Phase 3 shards")
            return assembled

        # Fallback: single Phase 3 output file
        if phase3_path.exists():
            df = pd.read_pickle(phase3_path)
            if sample_size is not None and sample_size < len(df):
                df = df.head(sample_size)
            logger.info(f"Loaded {len(df)} buildings from Phase 3 output")
            return df

        raise FileNotFoundError("Phase 3 output not found: no shards manifest and no pickle at " + str(phase3_path))
    
    def _integrate_weather_with_building(self, building: pd.Series, 
                                        weather_1min: pd.DataFrame,
                                        date: datetime) -> Dict:
        """Integrate weather data with a single building and its occupants."""
        building_dict = building.to_dict()
        # Track building processed
        try:
            self.stats['buildings_count'] += 1
        except Exception:
            pass
        
        # Add building-level weather summary
        if not weather_1min.empty:
            building_dict['weather_summary'] = create_weather_summary(weather_1min)
            building_dict['weather_date'] = date.strftime('%Y-%m-%d')
        else:
            building_dict['weather_summary'] = {}
            building_dict['weather_date'] = None
        
        # Process each person's activities
        if 'persons' in building_dict and isinstance(building_dict['persons'], list):
            updated_persons = []
            
            for person in building_dict['persons']:
                if not isinstance(person, dict):
                    updated_persons.append(person)
                    continue
                
                self.stats['persons_count'] += 1
                
                # Check if person has activity sequence
                if 'activity_sequence' in person and person['activity_sequence']:
                    # Activities are embedded from updated Phase 3
                    person = self._align_weather_with_person_activities(
                        person, weather_1min
                    )
                elif 'atus_case_id' in person and not self.atus_activities.empty:
                    # Retrieve activities using case_id
                    person = self._retrieve_and_align_activities(
                        person, weather_1min
                    )
                else:
                    # No detailed activities available
                    person['weather_aligned'] = False
                
                updated_persons.append(person)
            
            building_dict['persons'] = updated_persons
        
        building_dict['has_weather'] = not weather_1min.empty
        if building_dict['has_weather']:
            try:
                self.stats['buildings_with_weather'] = self.stats.get('buildings_with_weather', 0) + 1
            except Exception:
                pass
        
        return building_dict

    def _group_buildings_by_state(self, buildings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group buildings by STATE FIPS code (zero-padded)."""
        if buildings is None or len(buildings) == 0:
            return {}
        df = buildings.copy()
        if 'STATE' not in df.columns:
            logger.warning("No STATE column found; processing all buildings as unknown state")
            return {'XX': df}
        # Normalize state to zero-padded string
        df['STATE'] = df['STATE'].astype(str).str.zfill(2)
        grouped = {state: grp for state, grp in df.groupby('STATE')}
        return grouped

    def _load_atus_activities(self) -> pd.DataFrame:
        """Load ATUS activity-level data; return empty DataFrame on failure."""
        try:
            df = load_atus_activity_data()
            # basic sanity
            if 'case_id' not in df.columns:
                logger.warning("ATUS activities missing case_id column; downstream lookup may be limited")
            return df
        except Exception as e:
            logger.warning(f"Could not load ATUS activity data: {e}")
            return pd.DataFrame()

    def _load_state_weather(self, state_fips: str, date: datetime, use_cached_weather: bool) -> pd.DataFrame:
        """Load 1-min weather for a state with simple in-process caching + file cache."""
        try:
            date_key = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)
            cache_key = (str(state_fips).zfill(2), date_key if date_key else 'ALL')

            # In-memory cache
            if cache_key in self.weather_cache:
                self.stats['cache_hits'] = self.stats.get('cache_hits', 0) + 1
                return self.weather_cache[cache_key]

            # File-level cache per weather_loader
            use_cache = bool(use_cached_weather)
            weather_1min = load_and_process_state_weather(
                str(state_fips).zfill(2), date, cache=use_cache
            )

            # Record state processed
            try:
                self.stats['states_processed'].add(str(state_fips).zfill(2))
            except Exception:
                pass

            # Store in-memory cache
            if use_cache:
                self.weather_cache[cache_key] = weather_1min

            state_name = FIPS_TO_STATE.get(str(state_fips).zfill(2), f"FIPS_{state_fips}")
            logger.info(f"Loaded weather for {state_name}: {len(weather_1min)} 1-minute records")
            return weather_1min
        except Exception as e:
            logger.error(f"Failed to load weather for state {state_fips}: {e}")
            return pd.DataFrame()
    
    def _align_weather_with_person_activities(self, person: Dict, 
                                             weather_1min: pd.DataFrame) -> Dict:
        """Align weather with a person's activity sequence."""
        if weather_1min.empty:
            person['weather_aligned'] = False
            return person
        
        activities = person.get('activity_sequence', [])
        aligned_activities = []
        
        for activity in activities:
            self.stats['activities_count'] += 1
            
            # Align weather with this activity
            activity_with_weather = align_weather_with_activity(
                activity.copy(), weather_1min
            )
            aligned_activities.append(activity_with_weather)
            
            if 'weather' in activity_with_weather:
                self.stats['weather_alignments'] += 1
        
        person['activity_sequence'] = aligned_activities
        person['weather_aligned'] = True
        
        # Calculate person-level weather exposure
        person['weather_exposure'] = self._calculate_weather_exposure(aligned_activities)
        
        return person
    
    def _retrieve_and_align_activities(self, person: Dict, 
                                      weather_1min: pd.DataFrame) -> Dict:
        """Retrieve activities using case_id and align with weather."""
        case_id = person.get('atus_case_id')
        
        if not case_id or self.atus_activities.empty:
            person['weather_aligned'] = False
            return person
        
        # Retrieve activities for this person
        person_activities = self.atus_activities[
            self.atus_activities['case_id'] == case_id
        ].copy()
        
        if person_activities.empty:
            person['weather_aligned'] = False
            return person
        
        # Sort by activity number
        person_activities = person_activities.sort_values('activity_number')
        
        # Create activity sequence
        activity_sequence = []
        for _, act in person_activities.iterrows():
            activity = {
                'activity_num': int(act.get('activity_number', 0)),
                'start_time': str(act.get('start_time', '')),
                'stop_time': str(act.get('stop_time', '')),
                'duration_minutes': int(act.get('duration_minutes', 0)),
                'activity_code': str(act.get('activity_tier1', '')) + 
                                str(act.get('activity_tier2', '')) + 
                                str(act.get('activity_tier3', '')),
                'location': str(act.get('TEWHERE', -1))
            }
            
            # Align weather
            if not weather_1min.empty:
                activity = align_weather_with_activity(activity, weather_1min)
                self.stats['weather_alignments'] += 1
            
            activity_sequence.append(activity)
            self.stats['activities_count'] += 1
        
        person['activity_sequence'] = activity_sequence
        person['weather_aligned'] = True
        person['weather_exposure'] = self._calculate_weather_exposure(activity_sequence)
        
        return person
    
    def _calculate_weather_exposure(self, activities: List[Dict]) -> Dict:
        """Calculate summary of weather exposure across activities."""
        if not activities:
            return {}
        
        # Aggregate weather exposure
        outdoor_minutes = 0
        temp_exposures = []
        solar_exposure = 0
        
        for activity in activities:
            duration = activity.get('duration_minutes', 0)
            location = activity.get('location', '-1')
            
            # Check if outdoor activity (location codes for outdoor)
            # ATUS location codes: 
            # - Outdoors away from home = 2
            # - Outdoors at home = 3
            # - Various outdoor locations > 10
            if location in ['2', '3'] or (location.isdigit() and int(location) > 10):
                outdoor_minutes += duration
                
                if 'weather' in activity:
                    weather = activity['weather']
                    if 'temp_mean' in weather and weather['temp_mean'] is not None:
                        temp_exposures.append(weather['temp_mean'])
                    if 'solar_total' in weather and weather['solar_total'] is not None:
                        solar_exposure += weather['solar_total']
        
        exposure_summary = {
            'outdoor_minutes': outdoor_minutes,
            'outdoor_hours': outdoor_minutes / 60,
            'avg_outdoor_temp': np.mean(temp_exposures) if temp_exposures else None,
            'total_solar_exposure': solar_exposure,
            'indoor_minutes': sum(a.get('duration_minutes', 0) for a in activities) - outdoor_minutes
        }
        
        return exposure_summary
    
    def _save_results(self, buildings: pd.DataFrame):
        """Save Phase 4 results with metadata and generate validation report."""
        logger.info("Saving Phase 4 results")
        
        # Save main output
        buildings.to_pickle(self.phase4_output_path)
        
        # Save metadata
        metadata = {
            'phase': 'phase4_weather_integration',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            'statistics': {
                'buildings_count': self.stats['buildings_count'],
                'persons_count': self.stats['persons_count'],
                'activities_count': self.stats['activities_count'],
                'states_processed': list(self.stats['states_processed']),
                'weather_alignments': self.stats['weather_alignments'],
                'cache_hits': self.stats['cache_hits'],
                'weather_coverage_rate': (self.stats.get('buildings_with_weather', 0) / 
                                         max(self.stats['buildings_count'], 1)) * 100
            },
            'weather_resolution': '1-minute interpolated from 30-minute NSRDB data',
            'activity_resolution': '1-minute ATUS activities',
            'data_source': 'NSRDB 2023 weather data',
            'note': 'Weather aligned with individual activity periods'
        }
        
        # Use the tracked count from stats
        metadata['statistics']['buildings_with_weather'] = self.stats.get('buildings_with_weather', 0)
        
        with open(self.phase4_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved results to {self.phase4_output_path}")
        logger.info(f"Saved metadata to {self.phase4_metadata_path}")
        
        # Generate validation report
        try:
            from ..validation.phase_validators import validate_phase4_output
            from ..validation.report_generator import generate_phase4_report
            
            logger.info("Running Phase 4 validation")
            validation_results = validate_phase4_output(buildings)
            
            logger.info("Generating Phase 4 validation report")
            report_path = generate_phase4_report(buildings, metadata, validation_results)
            logger.info(f"Validation report saved to {report_path}")
            
            # Log validation summary
            if validation_results.get('valid', False):
                logger.info("Phase 4 validation PASSED")
            else:
                logger.warning("Phase 4 validation completed with issues:")
                for error in validation_results.get('errors', []):
                    logger.error(f"  - ERROR: {error}")
                for warning in validation_results.get('warnings', []):
                    logger.warning(f"  - WARNING: {warning}")
                    
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            # Don't fail the whole process if report generation fails
            pass


def run_phase4(sample_size: Optional[int] = None,
               use_cached_weather: bool = True,
               date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Convenience function to run Phase 4 weather integration.
    
    Args:
        sample_size: Number of buildings to process
        use_cached_weather: Whether to use cached weather data
        date: Date to process (default: January 1, 2023)
        
    Returns:
        DataFrame with weather-integrated buildings
    """
    integrator = Phase4WeatherIntegrator()
    return integrator.run(sample_size, use_cached_weather, date)