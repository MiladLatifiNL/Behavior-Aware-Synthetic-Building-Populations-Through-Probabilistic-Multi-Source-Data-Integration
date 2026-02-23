"""
Weather Data Visualizations.

Creates detailed visualizations of weather patterns, conditions, and temporal alignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("coolwarm")

class WeatherVisualizer:
    """Creates visualizations for weather data and patterns."""
    
    def __init__(self, output_dir: str = "results/visualizations/weather"):
        """Initialize the weather visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_weather_from_buildings(self, buildings_df: pd.DataFrame) -> pd.DataFrame:
        """Extract weather data from buildings."""
        weather_records = []
        
        for idx, building in buildings_df.iterrows():
            if 'weather_data' in building and isinstance(building['weather_data'], dict):
                weather = building['weather_data']
                weather['building_id'] = building.get('building_id', f'B{idx}')
                weather['state'] = building.get('STATE', 'Unknown')
                weather_records.append(weather)
            elif 'temperature' in building:  # Alternative format
                weather_records.append({
                    'building_id': building.get('building_id', f'B{idx}'),
                    'temperature': building.get('temperature'),
                    'humidity': building.get('humidity'),
                    'wind_speed': building.get('wind_speed'),
                    'solar_radiation': building.get('solar_radiation'),
                    'precipitation': building.get('precipitation'),
                    'state': building.get('STATE', 'Unknown')
                })
        
        return pd.DataFrame(weather_records)
    
    def create_temperature_profiles(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create temperature profile visualizations.
        
        Args:
            buildings_df: Buildings dataframe with weather data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Temperature Profiles and Patterns", fontsize=16, fontweight='bold')
        
        weather_df = self.extract_weather_from_buildings(buildings_df)
        
        # 1. Daily temperature curve (simulated 24-hour)
        ax = axes[0, 0]
        hours = np.arange(24)
        # Simulate typical daily temperature pattern
        temp_base = weather_df['temperature'].mean() if 'temperature' in weather_df.columns else 70
        daily_temps = temp_base + 10 * np.sin((hours - 6) * np.pi / 12)
        
        ax.plot(hours, daily_temps, 'r-', linewidth=2, label='Average')
        ax.fill_between(hours, daily_temps - 5, daily_temps + 5, alpha=0.3, color='red')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Temperature (°F)")
        ax.set_title("24-Hour Temperature Profile", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Temperature distribution
        ax = axes[0, 1]
        if 'temperature' in weather_df.columns and len(weather_df) > 0:
            ax.hist(weather_df['temperature'].dropna(), bins=30, 
                   color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(weather_df['temperature'].mean(), color='red', 
                      linestyle='--', label=f"Mean: {weather_df['temperature'].mean():.1f}°F")
            ax.set_xlabel("Temperature (°F)")
            ax.set_ylabel("Count")
            ax.set_title("Temperature Distribution", fontweight='bold')
            ax.legend()
        
        # 3. Temperature by state/region
        ax = axes[1, 0]
        if 'state' in weather_df.columns and 'temperature' in weather_df.columns:
            state_temps = weather_df.groupby('state')['temperature'].mean().sort_values().tail(10)
            ax.barh(state_temps.index, state_temps.values, color='orange')
            ax.set_xlabel("Average Temperature (°F)")
            ax.set_title("Temperature by State (Top 10)", fontweight='bold')
        
        # 4. Temperature vs time of year (simulated seasonal)
        ax = axes[1, 1]
        months = np.arange(1, 13)
        seasonal_temps = 70 + 20 * np.sin((months - 4) * np.pi / 6)
        ax.plot(months, seasonal_temps, 'b-', linewidth=2, marker='o')
        ax.fill_between(months, seasonal_temps - 10, seasonal_temps + 10, 
                        alpha=0.3, color='blue')
        ax.set_xlabel("Month")
        ax.set_ylabel("Temperature (°F)")
        ax.set_title("Seasonal Temperature Pattern", fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "temperature_profiles.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temperature profiles to {output_path}")
        
        return fig
    
    def create_weather_conditions_dashboard(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create comprehensive weather conditions dashboard.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Weather Conditions Dashboard", fontsize=16, fontweight='bold')
        
        weather_df = self.extract_weather_from_buildings(buildings_df)
        
        # 1. Humidity distribution
        ax = axes[0, 0]
        if 'humidity' in weather_df.columns:
            ax.hist(weather_df['humidity'].dropna(), bins=20, 
                   color='blue', alpha=0.6, edgecolor='black')
            ax.set_xlabel("Humidity (%)")
            ax.set_ylabel("Count")
            ax.set_title("Humidity Distribution", fontweight='bold')
        
        # 2. Wind speed
        ax = axes[0, 1]
        if 'wind_speed' in weather_df.columns:
            ax.hist(weather_df['wind_speed'].dropna(), bins=20,
                   color='green', alpha=0.6, edgecolor='black')
            ax.set_xlabel("Wind Speed (mph)")
            ax.set_ylabel("Count")
            ax.set_title("Wind Speed Distribution", fontweight='bold')
        
        # 3. Solar radiation
        ax = axes[0, 2]
        if 'solar_radiation' in weather_df.columns:
            ax.hist(weather_df['solar_radiation'].dropna(), bins=20,
                   color='yellow', alpha=0.6, edgecolor='black')
            ax.set_xlabel("Solar Radiation (W/m²)")
            ax.set_ylabel("Count")
            ax.set_title("Solar Radiation", fontweight='bold')
        
        # 4. Temperature vs Humidity scatter
        ax = axes[1, 0]
        if 'temperature' in weather_df.columns and 'humidity' in weather_df.columns:
            ax.scatter(weather_df['temperature'], weather_df['humidity'],
                      alpha=0.5, s=20, c=weather_df['wind_speed'] if 'wind_speed' in weather_df.columns else 'blue')
            ax.set_xlabel("Temperature (°F)")
            ax.set_ylabel("Humidity (%)")
            ax.set_title("Temperature vs Humidity", fontweight='bold')
        
        # 5. Precipitation patterns
        ax = axes[1, 1]
        if 'precipitation' in weather_df.columns:
            precip_counts = pd.cut(weather_df['precipitation'], 
                                  bins=[0, 0.01, 0.1, 0.5, 1.0, 10],
                                  labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme']).value_counts()
            ax.pie(precip_counts.values, labels=precip_counts.index, autopct='%1.1f%%',
                  colors=['lightblue', 'blue', 'darkblue', 'purple', 'darkred'])
            ax.set_title("Precipitation Categories", fontweight='bold')
        
        # 6. Climate zones
        ax = axes[1, 2]
        if 'climate_zone' in buildings_df.columns:
            climate_zones = buildings_df['climate_zone'].value_counts()
            ax.bar(climate_zones.index[:5], climate_zones.values[:5], color='teal')
            ax.set_xlabel("Climate Zone")
            ax.set_ylabel("Count")
            ax.set_title("Climate Zone Distribution", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "weather_conditions_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved weather dashboard to {output_path}")
        
        return fig
    
    def create_weather_activity_alignment(self, buildings_df: pd.DataFrame) -> go.Figure:
        """
        Create visualization showing weather-activity alignment.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Activity Patterns by Weather Conditions",
                          "Energy Use vs Weather"),
            row_heights=[0.6, 0.4]
        )
        
        # Simulate activity-weather relationships
        weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
        outdoor_activities = [80, 60, 20, 10]
        indoor_activities = [20, 40, 80, 90]
        
        # Activity patterns by weather
        fig.add_trace(
            go.Bar(name='Outdoor Activities', x=weather_conditions, y=outdoor_activities,
                  marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Indoor Activities', x=weather_conditions, y=indoor_activities,
                  marker_color='blue'),
            row=1, col=1
        )
        
        # Energy use vs temperature
        temps = np.linspace(20, 100, 50)
        heating = np.maximum(0, 65 - temps) * 2
        cooling = np.maximum(0, temps - 75) * 1.5
        
        fig.add_trace(
            go.Scatter(x=temps, y=heating, name='Heating Load',
                      line=dict(color='red', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=temps, y=cooling, name='Cooling Load',
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Weather Condition", row=1, col=1)
        fig.update_xaxes(title_text="Temperature (°F)", row=2, col=1)
        fig.update_yaxes(title_text="Activity Level (%)", row=1, col=1)
        fig.update_yaxes(title_text="Energy Load (kW)", row=2, col=1)
        
        fig.update_layout(
            title_text="Weather-Activity-Energy Alignment",
            height=800,
            showlegend=True
        )
        
        # Save
        output_path = self.output_dir / "weather_activity_alignment.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved weather-activity alignment to {output_path}")
        
        return fig
    
    def create_extreme_weather_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create extreme weather analysis visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Extreme Weather Analysis", fontsize=16, fontweight='bold')
        
        weather_df = self.extract_weather_from_buildings(buildings_df)
        
        # 1. Temperature extremes
        ax = axes[0, 0]
        if 'temperature' in weather_df.columns and len(weather_df) > 0:
            temps = weather_df['temperature'].dropna()
            extreme_cold = len(temps[temps < 32]) / len(temps) * 100
            extreme_hot = len(temps[temps > 95]) / len(temps) * 100
            normal = 100 - extreme_cold - extreme_hot
            
            ax.pie([extreme_cold, normal, extreme_hot], 
                  labels=['Freezing (<32°F)', 'Normal', 'Extreme Heat (>95°F)'],
                  colors=['lightblue', 'lightgreen', 'red'],
                  autopct='%1.1f%%')
            ax.set_title("Temperature Extremes", fontweight='bold')
        
        # 2. High wind events
        ax = axes[0, 1]
        if 'wind_speed' in weather_df.columns:
            winds = weather_df['wind_speed'].dropna()
            wind_categories = pd.cut(winds, bins=[0, 10, 25, 40, 100],
                                    labels=['Calm', 'Moderate', 'Strong', 'Extreme'])
            wind_counts = wind_categories.value_counts()
            ax.bar(wind_counts.index, wind_counts.values, 
                  color=['green', 'yellow', 'orange', 'red'])
            ax.set_xlabel("Wind Category")
            ax.set_ylabel("Count")
            ax.set_title("Wind Speed Categories", fontweight='bold')
        
        # 3. Precipitation intensity
        ax = axes[1, 0]
        if 'precipitation' in weather_df.columns:
            precip = weather_df['precipitation'].dropna()
            ax.hist(precip[precip > 0], bins=30, color='blue', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Precipitation (inches)")
            ax.set_ylabel("Count")
            ax.set_title("Precipitation Events", fontweight='bold')
            ax.axvline(1.0, color='red', linestyle='--', label='Heavy Rain (>1")')
            ax.legend()
        
        # 4. Extreme weather frequency by state
        ax = axes[1, 1]
        if 'state' in weather_df.columns and 'temperature' in weather_df.columns:
            # Count extreme events by state
            extreme_counts = {}
            for state in weather_df['state'].unique():
                state_data = weather_df[weather_df['state'] == state]
                if 'temperature' in state_data.columns:
                    temps = state_data['temperature'].dropna()
                    extremes = len(temps[(temps < 32) | (temps > 95)])
                    extreme_counts[state] = extremes
            
            if extreme_counts:
                top_states = dict(sorted(extreme_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
                ax.barh(list(top_states.keys()), list(top_states.values()), color='darkred')
                ax.set_xlabel("Extreme Weather Events")
                ax.set_title("States with Most Extreme Weather", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "extreme_weather_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved extreme weather analysis to {output_path}")
        
        return fig
    
    def create_temporal_weather_patterns(self, buildings_df: pd.DataFrame) -> go.Figure:
        """
        Create temporal weather pattern visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Plotly figure
        """
        # Create hourly weather patterns
        hours = list(range(24))
        
        # Simulate typical daily patterns
        temp_pattern = [65 + 15 * np.sin((h - 6) * np.pi / 12) for h in hours]
        humidity_pattern = [70 - 20 * np.sin((h - 6) * np.pi / 12) for h in hours]
        solar_pattern = [max(0, 500 * np.sin((h - 6) * np.pi / 12)) for h in hours]
        
        # Create plotly figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add temperature trace
        fig.add_trace(
            go.Scatter(x=hours, y=temp_pattern, name="Temperature",
                      line=dict(color="red", width=2)),
            secondary_y=False,
        )
        
        # Add humidity trace
        fig.add_trace(
            go.Scatter(x=hours, y=humidity_pattern, name="Humidity",
                      line=dict(color="blue", width=2)),
            secondary_y=True,
        )
        
        # Add solar radiation as area
        fig.add_trace(
            go.Scatter(x=hours, y=solar_pattern, name="Solar Radiation",
                      fill='tozeroy', line=dict(color="orange", width=1),
                      opacity=0.3),
            secondary_y=False,
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Hour of Day")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Temperature (°F) / Solar (W/m²)", secondary_y=False)
        fig.update_yaxes(title_text="Humidity (%)", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title_text="24-Hour Weather Pattern",
            height=500,
            hovermode='x unified'
        )
        
        # Save
        output_path = self.output_dir / "temporal_weather_patterns.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved temporal patterns to {output_path}")
        
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all weather visualizations."""
        self.create_temperature_profiles(buildings_df)
        self.create_weather_conditions_dashboard(buildings_df)
        self.create_weather_activity_alignment(buildings_df)
        self.create_extreme_weather_analysis(buildings_df)
        self.create_temporal_weather_patterns(buildings_df)
        
        logger.info(f"Created all weather visualizations in {self.output_dir}")