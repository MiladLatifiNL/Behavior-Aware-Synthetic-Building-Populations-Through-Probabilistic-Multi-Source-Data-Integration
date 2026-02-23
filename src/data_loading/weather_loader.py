"""
Weather data loader for Phase 4.

This module loads NSRDB weather data from state-level CSV files and interpolates
from 30-minute resolution to 1-minute resolution for alignment with ATUS activities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


# FIPS code to state name mapping for weather data files
FIPS_TO_STATE = {
    '01': 'alabama', '02': 'alaska', '04': 'arizona', '05': 'arkansas',
    '06': 'california', '08': 'colorado', '09': 'connecticut', '10': 'delaware',
    '11': 'district_of_columbia', '12': 'florida', '13': 'georgia', '15': 'hawaii',
    '16': 'idaho', '17': 'illinois', '18': 'indiana', '19': 'iowa',
    '20': 'kansas', '21': 'kentucky', '22': 'louisiana', '23': 'maine',
    '24': 'maryland', '25': 'massachusetts', '26': 'michigan', '27': 'minnesota',
    '28': 'mississippi', '29': 'missouri', '30': 'montana', '31': 'nebraska',
    '32': 'nevada', '33': 'new_hampshire', '34': 'new_jersey', '35': 'new_mexico',
    '36': 'new_york', '37': 'north_carolina', '38': 'north_dakota', '39': 'ohio',
    '40': 'oklahoma', '41': 'oregon', '42': 'pennsylvania', '44': 'rhode_island',
    '45': 'south_carolina', '46': 'south_dakota', '47': 'tennessee', '48': 'texas',
    '49': 'utah', '50': 'vermont', '51': 'virginia', '53': 'washington',
    '54': 'west_virginia', '55': 'wisconsin', '56': 'wyoming'
}


def load_state_weather_data(state_fips: str, weather_path: Path = None) -> pd.DataFrame:
    """
    Load weather data for a specific state.
    
    Args:
        state_fips: FIPS code for the state (e.g., '01' for Alabama)
        weather_path: Path to weather data directory (default: data/raw/weather/2023)
        
    Returns:
        DataFrame with weather data for the state
        
    Raises:
        FileNotFoundError: If weather data file not found for the state
        ValueError: If FIPS code is invalid
    """
    if weather_path is None:
        weather_path = Path("data/raw/weather/2023")
    
    # Map FIPS code to state name
    if state_fips not in FIPS_TO_STATE:
        raise ValueError(f"Invalid FIPS code: {state_fips}")
    
    state_name = FIPS_TO_STATE[state_fips]
    weather_file = weather_path / state_name / f"{state_name}.csv"
    
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather data not found for {state_name}: {weather_file}")
    
    logger.info(f"Loading weather data for {state_name} from {weather_file}")
    
    # Read the CSV file - first two rows are metadata
    # Try different encodings as weather data might have degree symbols
    try:
        df = pd.read_csv(weather_file, skiprows=2, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(weather_file, skiprows=2, low_memory=False, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(weather_file, skiprows=2, low_memory=False, encoding='cp1252')
    
    # Create datetime index from Year, Month, Day, Hour, Minute columns
    df['datetime'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour', 'Minute']].assign(Second=0)
    )
    df = df.set_index('datetime')
    
    logger.info(f"Loaded {len(df)} weather records for {state_name}")
    
    return df


def interpolate_weather_to_minutes(weather_30min: pd.DataFrame, 
                                  date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Interpolate weather data from 30-minute to 1-minute resolution.
    
    Args:
        weather_30min: Weather data at 30-minute resolution
        date: Specific date to interpolate (default: all data)
        
    Returns:
        DataFrame with 1-minute resolution weather data
    """
    logger.info("Interpolating weather data from 30-min to 1-min resolution")
    
    # Filter to specific date if provided
    if date:
        date_str = date.strftime('%Y-%m-%d')
        weather_30min = weather_30min[date_str:date_str]
    
    # Select key weather variables for interpolation
    weather_vars = {
        'Temperature': 'linear',  # Linear interpolation for temperature
        'Relative Humidity': 'linear',  # Linear for humidity
        'Pressure': 'linear',  # Linear for pressure
        'Wind Speed': 'linear',  # Linear for wind speed
        'Wind Direction': 'linear',  # Linear for wind direction (with circular handling)
        'GHI': 'linear',  # Global Horizontal Irradiance - linear
        'DHI': 'linear',  # Diffuse Horizontal Irradiance - linear
        'DNI': 'linear',  # Direct Normal Irradiance - linear
        'Cloud Type': 'pad',  # Forward fill for categorical cloud type
        'Dew Point': 'linear'  # Linear for dew point
    }
    
    # Select available columns
    available_cols = [col for col in weather_vars.keys() if col in weather_30min.columns]
    weather_subset = weather_30min[available_cols].copy()
    
    # Create 1-minute index for the date range
    start_time = weather_subset.index[0]
    end_time = weather_subset.index[-1]
    minute_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Reindex to 1-minute frequency
    weather_1min = weather_subset.reindex(minute_index)
    
    # Apply appropriate interpolation for each variable
    for col in available_cols:
        method = weather_vars.get(col, 'linear')
        
        if col == 'Wind Direction':
            # Special handling for circular wind direction
            weather_1min[col] = interpolate_circular(weather_1min[col])
        elif col == 'Cloud Type':
            # Forward fill for categorical data
            weather_1min[col] = weather_1min[col].ffill()
        else:
            # Standard interpolation
            weather_1min[col] = weather_1min[col].interpolate(method=method)
    
    # Fill any remaining NaN values at boundaries
    weather_1min = weather_1min.ffill().bfill()
    
    logger.info(f"Interpolated to {len(weather_1min)} 1-minute records")
    
    return weather_1min


def calculate_heat_index(temperature: pd.Series, humidity: pd.Series) -> pd.Series:
    """
    Calculate heat index from temperature (Celsius) and relative humidity.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in percent
        
    Returns:
        Heat index in Celsius
    """
    # Convert to Fahrenheit for calculation
    temp_f = temperature * 9/5 + 32
    
    # Simplified heat index formula
    heat_index_f = np.where(
        temp_f >= 80,
        -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity 
        - 0.22475541 * temp_f * humidity - 0.00683783 * temp_f**2 
        - 0.05481717 * humidity**2 + 0.00122874 * temp_f**2 * humidity 
        + 0.00085282 * temp_f * humidity**2 - 0.00000199 * temp_f**2 * humidity**2,
        temp_f  # Use actual temperature if below threshold
    )
    
    # Convert back to Celsius
    heat_index_c = (heat_index_f - 32) * 5/9
    return pd.Series(heat_index_c, index=temperature.index)


def calculate_hvac_demand_indicator(weather_1min: pd.DataFrame) -> float:
    """
    Calculate HVAC demand indicator based on weather conditions.
    
    Args:
        weather_1min: Weather data with temperature and humidity
        
    Returns:
        HVAC demand score (0-100)
    """
    score = 0
    
    if 'Temperature' in weather_1min.columns:
        temp = weather_1min['Temperature']
        # Heating demand (below 15°C)
        heating_demand = np.maximum(15 - temp, 0).mean() * 3
        # Cooling demand (above 25°C)
        cooling_demand = np.maximum(temp - 25, 0).mean() * 3
        score = min(heating_demand + cooling_demand, 100)
    
    return score


def calculate_peak_demand_risk(weather_1min: pd.DataFrame) -> str:
    """
    Calculate risk of peak energy demand based on weather extremes.
    
    Args:
        weather_1min: Weather data
        
    Returns:
        Risk level: 'Low', 'Medium', 'High', or 'Extreme'
    """
    risk_score = 0
    
    if 'Temperature' in weather_1min.columns:
        temp = weather_1min['Temperature']
        # Check for extreme temperatures
        if (temp < -10).any() or (temp > 35).any():
            risk_score += 3
        elif (temp < 0).any() or (temp > 30).any():
            risk_score += 2
        elif (temp < 10).any() or (temp > 25).any():
            risk_score += 1
    
    if 'Relative Humidity' in weather_1min.columns:
        humidity = weather_1min['Relative Humidity']
        # High humidity increases cooling demand
        if (humidity > 80).mean() > 0.5:
            risk_score += 1
    
    if risk_score >= 4:
        return 'Extreme'
    elif risk_score >= 3:
        return 'High'
    elif risk_score >= 2:
        return 'Medium'
    else:
        return 'Low'


def interpolate_circular(series: pd.Series) -> pd.Series:
    """
    Interpolate circular data (e.g., wind direction in degrees).
    
    Args:
        series: Series with circular data (0-360 degrees)
        
    Returns:
        Interpolated series handling circular wraparound
    """
    # Convert to radians for circular interpolation
    radians = np.deg2rad(series)
    
    # Convert to x,y components
    x = np.cos(radians)
    y = np.sin(radians)
    
    # Interpolate x and y separately
    x_interp = pd.Series(x).interpolate(method='linear')
    y_interp = pd.Series(y).interpolate(method='linear')
    
    # Convert back to degrees
    result = np.rad2deg(np.arctan2(y_interp, x_interp))
    
    # Ensure 0-360 range
    result = result % 360
    
    return pd.Series(result, index=series.index)


def calculate_degree_days(weather_1min: pd.DataFrame, base_temp: float = 65.0) -> Dict[str, float]:
    """
    Calculate heating and cooling degree days from temperature data.
    
    Args:
        weather_1min: Weather data with Temperature column
        base_temp: Base temperature in Fahrenheit (default: 65F)
        
    Returns:
        Dictionary with HDD and CDD values
    """
    if 'Temperature' not in weather_1min.columns:
        return {'HDD': 0.0, 'CDD': 0.0}
    
    temp = weather_1min['Temperature']
    
    # Calculate degree-minutes, then convert to degree-days
    heating_degree_minutes = np.maximum(base_temp - temp, 0).sum()
    cooling_degree_minutes = np.maximum(temp - base_temp, 0).sum()
    
    # Convert minutes to days (1440 minutes per day)
    hdd = heating_degree_minutes / 1440
    cdd = cooling_degree_minutes / 1440
    
    return {'HDD': hdd, 'CDD': cdd}


def create_weather_summary(weather_1min: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Create comprehensive summary statistics from 1-minute weather data.
    
    Args:
        weather_1min: Weather data at 1-minute resolution
        
    Returns:
        Dictionary with weather summary statistics including energy indicators
    """
    summary = {}
    
    # Temperature statistics
    if 'Temperature' in weather_1min.columns:
        temp = weather_1min['Temperature']
        summary['temp_mean'] = temp.mean()
        summary['temp_min'] = temp.min()
        summary['temp_max'] = temp.max()
        summary['temp_std'] = temp.std()
        summary['temp_range'] = temp.max() - temp.min()
        
        # Temperature percentiles for comfort analysis
        summary['temp_p10'] = temp.quantile(0.1)
        summary['temp_p90'] = temp.quantile(0.9)
        
        # Extreme temperature indicators
        summary['minutes_below_0C'] = (temp < 0).sum()
        summary['minutes_above_30C'] = (temp > 30).sum()
        summary['minutes_above_35C'] = (temp > 35).sum()
    
    # Humidity statistics and comfort indices
    if 'Relative Humidity' in weather_1min.columns:
        humidity = weather_1min['Relative Humidity']
        summary['humidity_mean'] = humidity.mean()
        summary['humidity_min'] = humidity.min()
        summary['humidity_max'] = humidity.max()
        
        # Humidity comfort zones
        summary['minutes_humid'] = (humidity > 70).sum()  # High humidity minutes
        summary['minutes_dry'] = (humidity < 30).sum()    # Low humidity minutes
        
        # Calculate heat index if temperature available
        if 'Temperature' in weather_1min.columns:
            heat_index = calculate_heat_index(weather_1min['Temperature'], humidity)
            summary['heat_index_mean'] = heat_index.mean()
            summary['heat_index_max'] = heat_index.max()
            summary['minutes_heat_stress'] = (heat_index > 32).sum()  # Heat stress threshold
    
    # Solar radiation totals and peak values
    if 'GHI' in weather_1min.columns:
        ghi = weather_1min['GHI']
        # Convert W/m2 minutes to Wh/m2 for daily total
        summary['solar_ghi_total'] = ghi.sum() / 60
        summary['solar_ghi_peak'] = ghi.max()
        summary['solar_ghi_mean'] = ghi.mean()
        
        # Solar availability metrics
        summary['minutes_sunny'] = (ghi > 200).sum()  # Minutes with significant solar
        summary['peak_solar_hour'] = ghi.idxmax().hour if len(ghi) > 0 else None
        
    if 'DHI' in weather_1min.columns:
        summary['solar_dhi_total'] = weather_1min['DHI'].sum() / 60
        summary['solar_dhi_peak'] = weather_1min['DHI'].max()
        
    if 'DNI' in weather_1min.columns:
        summary['solar_dni_total'] = weather_1min['DNI'].sum() / 60
        summary['solar_dni_peak'] = weather_1min['DNI'].max()
    
    # Wind statistics and energy potential
    if 'Wind Speed' in weather_1min.columns:
        wind = weather_1min['Wind Speed']
        summary['wind_speed_mean'] = wind.mean()
        summary['wind_speed_max'] = wind.max()
        summary['wind_speed_std'] = wind.std()
        
        # Wind energy potential (proportional to v^3)
        summary['wind_energy_potential'] = (wind ** 3).mean()
        summary['minutes_calm'] = (wind < 0.5).sum()  # Calm conditions
        summary['minutes_windy'] = (wind > 10).sum()   # Windy conditions
        
    # Enhanced degree days with hourly resolution
    degree_days = calculate_degree_days(weather_1min)
    summary.update(degree_days)
    
    # Add degree-minutes for finer resolution
    if 'Temperature' in weather_1min.columns:
        base_temp = 18.3  # 65°F in Celsius
        summary['heating_degree_minutes'] = np.maximum(base_temp - weather_1min['Temperature'], 0).sum()
        summary['cooling_degree_minutes'] = np.maximum(weather_1min['Temperature'] - base_temp, 0).sum()
    
    # Pressure and weather stability
    if 'Pressure' in weather_1min.columns:
        pressure = weather_1min['Pressure']
        summary['pressure_mean'] = pressure.mean()
        summary['pressure_min'] = pressure.min()
        summary['pressure_max'] = pressure.max()
        summary['pressure_change'] = pressure.iloc[-1] - pressure.iloc[0] if len(pressure) > 1 else 0
    
    # Dew point and condensation risk
    if 'Dew Point' in weather_1min.columns:
        dew_point = weather_1min['Dew Point']
        summary['dew_point_mean'] = dew_point.mean()
        
        # Condensation risk when dew point close to temperature
        if 'Temperature' in weather_1min.columns:
            temp_dew_diff = weather_1min['Temperature'] - dew_point
            summary['condensation_risk_minutes'] = (temp_dew_diff < 2).sum()
    
    # Cloud cover impact on solar
    if 'Cloud Type' in weather_1min.columns:
        # Estimate cloud impact (simplified)
        cloud_values = weather_1min['Cloud Type'].value_counts()
        summary['minutes_clear'] = cloud_values.get(0, 0) if 0 in cloud_values.index else 0
        summary['minutes_cloudy'] = len(weather_1min) - summary['minutes_clear']
    
    # Energy demand indicators
    summary['hvac_demand_indicator'] = calculate_hvac_demand_indicator(weather_1min)
    summary['peak_demand_risk'] = calculate_peak_demand_risk(weather_1min)
    
    # Weather volatility metrics
    if 'Temperature' in weather_1min.columns:
        temp_changes = weather_1min['Temperature'].diff().abs()
        summary['temp_volatility'] = temp_changes.mean()
        summary['max_temp_change'] = temp_changes.max()
    
    return summary


def load_and_process_state_weather(state_fips: str, 
                                   date: Optional[datetime] = None,
                                   cache: bool = True) -> pd.DataFrame:
    """
    Load and process weather data for a state, with caching.
    
    Args:
        state_fips: FIPS code for the state
        date: Specific date to process (default: all 2023)
        cache: Whether to cache processed data
        
    Returns:
        DataFrame with 1-minute resolution weather data
    """
    # Check cache first
    cache_dir = Path("data/processed/weather_cache")
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{state_fips}_1min.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached weather data for state {state_fips}")
            weather_1min = pd.read_pickle(cache_file)
            
            # Filter to specific date if requested
            if date:
                date_str = date.strftime('%Y-%m-%d')
                weather_1min = weather_1min[date_str:date_str]
            
            return weather_1min
    
    # Load raw data
    weather_30min = load_state_weather_data(state_fips)
    
    # Interpolate to 1-minute
    weather_1min = interpolate_weather_to_minutes(weather_30min, date)
    
    # Cache if requested
    if cache and not date:  # Only cache full year data
        cache_file = cache_dir / f"{state_fips}_1min.pkl"
        weather_1min.to_pickle(cache_file)
        logger.info(f"Cached weather data to {cache_file}")
    
    return weather_1min


def align_weather_with_activity(activity: Dict, weather_1min: pd.DataFrame,
                              base_date: Optional[datetime] = None) -> Dict:
    """
    Align weather data with a specific activity period, handling midnight crossings.
    
    Args:
        activity: Activity dictionary with start_time and stop_time
        weather_1min: Weather data at 1-minute resolution
        base_date: Base date for the activity (defaults to weather data date)
        
    Returns:
        Activity dictionary enhanced with weather data
    """
    # Parse activity times
    start_time = activity.get('start_time', '')
    stop_time = activity.get('stop_time', '')
    
    # Handle missing or empty times
    if not start_time or not stop_time:
        activity['weather'] = {'error': 'Missing start or stop time'}
        return activity
    
    # Use provided base date or default to weather data's date
    if base_date is None:
        base_date = weather_1min.index[0].date() if len(weather_1min) > 0 else datetime.now().date()
    
    # Parse time strings and create datetime objects
    try:
        # Handle both HH:MM and HH:MM:SS formats
        start_parts = start_time.split(':')
        stop_parts = stop_time.split(':')
        
        start_hour = int(start_parts[0])
        start_min = int(start_parts[1]) if len(start_parts) > 1 else 0
        start_sec = int(start_parts[2]) if len(start_parts) > 2 else 0
        
        stop_hour = int(stop_parts[0])
        stop_min = int(stop_parts[1]) if len(stop_parts) > 1 else 0
        stop_sec = int(stop_parts[2]) if len(stop_parts) > 2 else 0
        
    except (ValueError, IndexError) as e:
        activity['weather'] = {'error': f'Invalid time format: {e}'}
        return activity
    
    # Create datetime objects
    start_dt = datetime.combine(base_date, datetime.min.time().replace(
        hour=start_hour, minute=start_min, second=start_sec))
    stop_dt = datetime.combine(base_date, datetime.min.time().replace(
        hour=stop_hour, minute=stop_min, second=stop_sec))
    
    # Handle midnight crossing - ATUS activities can span midnight
    # Check if this looks like a midnight crossing (e.g., 23:00 to 02:00)
    if stop_dt <= start_dt:
        # This is a midnight crossing activity
        stop_dt += timedelta(days=1)
        activity['crosses_midnight'] = True
        
        # For multi-day weather data, we need to handle this properly
        # Check if we have weather data for the next day
        next_day_start = datetime.combine(base_date + timedelta(days=1), datetime.min.time())
        
        if len(weather_1min) > 0 and weather_1min.index[-1] >= stop_dt:
            # We have weather data for the full activity period
            pass
        else:
            # We need to handle partial weather data
            # For now, we'll use what we have and note the limitation
            activity['weather_partial'] = True
    
    # Extract weather for activity period
    try:
        # Handle case where activity extends beyond available weather data
        weather_start = max(start_dt, weather_1min.index[0]) if len(weather_1min) > 0 else start_dt
        weather_end = min(stop_dt, weather_1min.index[-1]) if len(weather_1min) > 0 else stop_dt
        
        if weather_start <= weather_end:
            activity_weather = weather_1min[weather_start:weather_end]
        else:
            activity_weather = pd.DataFrame()  # Empty DataFrame if no overlap
        
        if len(activity_weather) > 0:
            # Calculate comprehensive weather statistics for the activity
            weather_stats = {
                'samples': len(activity_weather),
                'duration_minutes': activity.get('duration_minutes', 0),
                'weather_coverage': len(activity_weather) / max(activity.get('duration_minutes', 1), 1)
            }
            
            # Temperature statistics
            if 'Temperature' in activity_weather.columns:
                temp = activity_weather['Temperature']
                weather_stats.update({
                    'temp_mean': temp.mean(),
                    'temp_min': temp.min(),
                    'temp_max': temp.max(),
                    'temp_std': temp.std() if len(temp) > 1 else 0
                })
                
                # Activity-specific comfort metrics
                weather_stats['minutes_comfortable'] = ((temp >= 18) & (temp <= 24)).sum()
                weather_stats['minutes_hot'] = (temp > 28).sum()
                weather_stats['minutes_cold'] = (temp < 15).sum()
            
            # Humidity statistics
            if 'Relative Humidity' in activity_weather.columns:
                humidity = activity_weather['Relative Humidity']
                weather_stats.update({
                    'humidity_mean': humidity.mean(),
                    'humidity_min': humidity.min(),
                    'humidity_max': humidity.max()
                })
            
            # Solar radiation (important for outdoor activities)
            if 'GHI' in activity_weather.columns:
                ghi = activity_weather['GHI']
                weather_stats.update({
                    'solar_total': ghi.sum() / 60,  # Wh/m2
                    'solar_mean': ghi.mean(),
                    'solar_max': ghi.max(),
                    'uv_exposure': estimate_uv_index(ghi.mean())  # Simplified UV estimate
                })
            
            # Wind statistics
            if 'Wind Speed' in activity_weather.columns:
                wind = activity_weather['Wind Speed']
                weather_stats.update({
                    'wind_speed_mean': wind.mean(),
                    'wind_speed_max': wind.max(),
                    'wind_chill_effect': calculate_wind_chill_minutes(
                        activity_weather) if 'Temperature' in activity_weather.columns else 0
                })
            
            # Pressure (can affect comfort and health)
            if 'Pressure' in activity_weather.columns:
                weather_stats['pressure_mean'] = activity_weather['Pressure'].mean()
            
            # Precipitation (if available)
            if 'Precipitation' in activity_weather.columns:
                precip = activity_weather['Precipitation']
                weather_stats.update({
                    'precipitation_total': precip.sum(),
                    'minutes_raining': (precip > 0).sum()
                })
            
            # Activity-specific energy impact
            location = activity.get('location', '-1')
            if location in ['2', '3'] or (location.isdigit() and int(location) > 10):
                # Outdoor activity - weather has direct impact
                weather_stats['exposure_type'] = 'outdoor'
                weather_stats['comfort_score'] = calculate_outdoor_comfort_score(activity_weather)
            else:
                # Indoor activity - weather affects building energy use
                weather_stats['exposure_type'] = 'indoor'
                weather_stats['hvac_load'] = estimate_hvac_load(activity_weather)
            
            activity['weather'] = weather_stats
            
        else:
            activity['weather'] = {
                'error': 'No weather data overlap with activity period',
                'requested_start': str(start_dt),
                'requested_end': str(stop_dt),
                'weather_available_start': str(weather_1min.index[0]) if len(weather_1min) > 0 else 'None',
                'weather_available_end': str(weather_1min.index[-1]) if len(weather_1min) > 0 else 'None'
            }
            
    except Exception as e:
        logger.warning(f"Error aligning weather for activity {start_time}-{stop_time}: {e}")
        activity['weather'] = {'error': str(e)}
    
    return activity


def estimate_uv_index(ghi_mean: float) -> float:
    """Estimate UV index from solar radiation (simplified)."""
    # Very simplified estimation - actual UV depends on many factors
    return min(ghi_mean * 0.025, 11)  # Cap at UV index 11


def calculate_wind_chill_minutes(weather_data: pd.DataFrame) -> int:
    """Calculate minutes with wind chill effect."""
    if 'Temperature' not in weather_data.columns or 'Wind Speed' not in weather_data.columns:
        return 0
    
    # Wind chill applies when temp < 10°C and wind > 4.8 km/h
    temp = weather_data['Temperature']
    wind = weather_data['Wind Speed']
    
    # Convert wind speed to km/h if needed (assuming m/s input)
    wind_kmh = wind * 3.6
    
    wind_chill_conditions = (temp < 10) & (wind_kmh > 4.8)
    return wind_chill_conditions.sum()


def calculate_outdoor_comfort_score(weather_data: pd.DataFrame) -> float:
    """Calculate comfort score for outdoor activities (0-100)."""
    score = 100
    
    if 'Temperature' in weather_data.columns:
        temp = weather_data['Temperature'].mean()
        # Optimal outdoor temperature range: 18-25°C
        if temp < 10 or temp > 32:
            score -= 30
        elif temp < 15 or temp > 28:
            score -= 15
        elif temp < 18 or temp > 25:
            score -= 5
    
    if 'Relative Humidity' in weather_data.columns:
        humidity = weather_data['Relative Humidity'].mean()
        if humidity > 80 or humidity < 20:
            score -= 20
        elif humidity > 70 or humidity < 30:
            score -= 10
    
    if 'Wind Speed' in weather_data.columns:
        wind = weather_data['Wind Speed'].mean()
        if wind > 10:  # m/s - strong wind
            score -= 15
        elif wind > 7:  # moderate wind
            score -= 5
    
    return max(score, 0)


def estimate_hvac_load(weather_data: pd.DataFrame) -> float:
    """Estimate HVAC load impact for indoor activities."""
    load = 0
    
    if 'Temperature' in weather_data.columns:
        temp = weather_data['Temperature'].mean()
        # Heating load (below 18°C)
        if temp < 18:
            load += (18 - temp) * 2
        # Cooling load (above 24°C)
        elif temp > 24:
            load += (temp - 24) * 2.5  # Cooling typically requires more energy
    
    return min(load, 100)  # Cap at 100