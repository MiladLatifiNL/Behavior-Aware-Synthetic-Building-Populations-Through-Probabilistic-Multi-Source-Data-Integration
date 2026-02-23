"""
Building Characteristics Visualizations.

Creates detailed visualizations of building properties, types, and distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

class BuildingVisualizer:
    """Creates visualizations for building characteristics."""
    
    def __init__(self, output_dir: str = "results/visualizations/buildings"):
        """Initialize the building visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_building_type_distribution(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create building type distribution visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Building Characteristics Distribution", fontsize=16, fontweight='bold')
        
        # Building type pie chart
        ax = axes[0, 0]
        if 'building_type_simple' in buildings_df.columns:
            building_types = buildings_df['building_type_simple'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(building_types)))
            wedges, texts, autotexts = ax.pie(building_types.values, 
                                              labels=building_types.index,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              explode=[0.05] * len(building_types))
            ax.set_title("Building Types", fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_color('white')
        
        # Building age distribution
        ax = axes[0, 1]
        if 'building_age' in buildings_df.columns:
            ax.hist(buildings_df['building_age'].dropna(), bins=20, 
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Building Age (years)")
            ax.set_ylabel("Count")
            ax.set_title("Building Age Distribution", fontweight='bold')
            ax.axvline(buildings_df['building_age'].median(), 
                      color='red', linestyle='--', label=f'Median: {buildings_df["building_age"].median():.0f}')
            ax.legend()
        
        # Household size distribution
        ax = axes[1, 0]
        if 'household_size' in buildings_df.columns:
            sizes = buildings_df['household_size'].value_counts().sort_index()
            ax.bar(sizes.index, sizes.values, color='coral', edgecolor='black')
            ax.set_xlabel("Household Size (persons)")
            ax.set_ylabel("Count")
            ax.set_title("Household Size Distribution", fontweight='bold')
            ax.set_xticks(sizes.index)
        
        # Number of bedrooms
        ax = axes[1, 1]
        if 'num_bedrooms' in buildings_df.columns:
            bedrooms = buildings_df['num_bedrooms'].value_counts().sort_index()
            ax.bar(bedrooms.index, bedrooms.values, color='lightgreen', edgecolor='black')
            ax.set_xlabel("Number of Bedrooms")
            ax.set_ylabel("Count")
            ax.set_title("Bedroom Count Distribution", fontweight='bold')
            ax.set_xticks(bedrooms.index)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "building_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved building type distribution to {output_path}")
        plt.close(fig)
        return fig
    
    def create_geographic_distribution(self, buildings_df: pd.DataFrame) -> go.Figure:
        """
        Create geographic distribution visualization.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Plotly figure
        """
        if 'STATE' not in buildings_df.columns:
            logger.warning("No STATE column found for geographic visualization")
            return None
        
        # Count buildings by state
        state_counts = buildings_df['STATE'].value_counts().reset_index()
        state_counts.columns = ['STATE', 'count']
        
        # Map state codes to names (simplified - you'd want a complete mapping)
        state_map = {
            '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
            '06': 'California', '08': 'Colorado', '09': 'Connecticut',
            '10': 'Delaware', '11': 'DC', '12': 'Florida', '13': 'Georgia'
        }
        state_counts['state_name'] = state_counts['STATE'].map(state_map).fillna('Other')
        
        # Create choropleth map
        fig = px.choropleth(
            state_counts,
            locations='state_name',
            locationmode='USA-states',
            color='count',
            color_continuous_scale='Viridis',
            scope='usa',
            title='Building Distribution by State',
            labels={'count': 'Number of Buildings'}
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='albers usa'
            ),
            height=600,
            width=1000
        )
        
        # Save
        output_path = self.output_dir / "geographic_distribution.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved geographic distribution to {output_path}")
        
        return fig
    
    def create_energy_characteristics(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create energy characteristics visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Building Energy Characteristics", fontsize=16, fontweight='bold')
        
        # Heating fuel type
        ax = axes[0, 0]
        if 'heating_fuel' in buildings_df.columns:
            fuel_types = buildings_df['heating_fuel'].value_counts()
            ax.barh(fuel_types.index[:10], fuel_types.values[:10], color='orange')
            ax.set_xlabel("Count")
            ax.set_title("Heating Fuel Types", fontweight='bold')
        
        # Energy intensity
        ax = axes[0, 1]
        if 'energy_intensity' in buildings_df.columns:
            ax.hist(buildings_df['energy_intensity'].dropna(), bins=30,
                   color='red', alpha=0.6, edgecolor='black')
            ax.set_xlabel("Energy Intensity")
            ax.set_ylabel("Count")
            ax.set_title("Energy Intensity Distribution", fontweight='bold')
        
        # Building age vs energy intensity
        ax = axes[0, 2]
        if 'building_age' in buildings_df.columns and 'energy_intensity' in buildings_df.columns:
            ax.scatter(buildings_df['building_age'], buildings_df['energy_intensity'],
                      alpha=0.5, s=20, c=buildings_df['household_size'], cmap='viridis')
            ax.set_xlabel("Building Age (years)")
            ax.set_ylabel("Energy Intensity")
            ax.set_title("Age vs Energy Intensity", fontweight='bold')
            plt.colorbar(ax.collections[0], ax=ax, label='Household Size')
        
        # RECS match quality
        ax = axes[1, 0]
        if 'recs_match_weight' in buildings_df.columns:
            ax.hist(buildings_df['recs_match_weight'].dropna(), bins=30,
                   color='green', alpha=0.6, edgecolor='black')
            ax.set_xlabel("RECS Match Weight")
            ax.set_ylabel("Count")
            ax.set_title("RECS Matching Quality", fontweight='bold')
            ax.axvline(0, color='red', linestyle='--', label='Match Threshold')
            ax.legend()
        
        # Climate zones
        ax = axes[1, 1]
        if 'climate_zone' in buildings_df.columns:
            climate = buildings_df['climate_zone'].value_counts()
            ax.pie(climate.values, labels=climate.index, autopct='%1.1f%%',
                  colors=plt.cm.coolwarm(np.linspace(0, 1, len(climate))))
            ax.set_title("Climate Zone Distribution", fontweight='bold')
        
        # Energy burden
        ax = axes[1, 2]
        if 'energy_burden' in buildings_df.columns:
            ax.hist(buildings_df['energy_burden'].dropna(), bins=30,
                   color='purple', alpha=0.6, edgecolor='black')
            ax.set_xlabel("Energy Burden (%)")
            ax.set_ylabel("Count")
            ax.set_title("Energy Burden Distribution", fontweight='bold')
            ax.axvline(6, color='red', linestyle='--', label='High Burden (>6%)')
            ax.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "energy_characteristics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved energy characteristics to {output_path}")
        plt.close(fig)
        return fig
    
    def create_building_occupancy_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create building occupancy analysis visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Building Occupancy Analysis", fontsize=16, fontweight='bold')
        
        # Occupancy density
        ax = axes[0, 0]
        if 'occupancy_intensity' in buildings_df.columns:
            ax.hist(buildings_df['occupancy_intensity'].dropna(), bins=30,
                   color='teal', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Occupancy Intensity (persons/room)")
            ax.set_ylabel("Count")
            ax.set_title("Occupancy Density", fontweight='bold')
        
        # Rooms per person
        ax = axes[0, 1]
        if 'rooms_per_person' in buildings_df.columns:
            ax.hist(buildings_df['rooms_per_person'].dropna(), bins=30,
                   color='navy', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Rooms per Person")
            ax.set_ylabel("Count")
            ax.set_title("Space per Person", fontweight='bold')
            ax.axvline(2, color='green', linestyle='--', label='Adequate (>2)')
            ax.axvline(1, color='red', linestyle='--', label='Crowded (<1)')
            ax.legend()
        
        # Building size vs occupancy
        ax = axes[1, 0]
        if 'num_rooms' in buildings_df.columns and 'household_size' in buildings_df.columns:
            ax.scatter(buildings_df['num_rooms'], buildings_df['household_size'],
                      alpha=0.5, s=30)
            ax.set_xlabel("Number of Rooms")
            ax.set_ylabel("Household Size")
            ax.set_title("Building Size vs Occupancy", fontweight='bold')
            
            # Add trend line
            z = np.polyfit(buildings_df['num_rooms'].dropna(), 
                          buildings_df.loc[buildings_df['num_rooms'].notna(), 'household_size'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(buildings_df['num_rooms'].min(), 
                                buildings_df['num_rooms'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend')
            ax.legend()
        
        # Tenure type
        ax = axes[1, 1]
        if 'tenure_type' in buildings_df.columns:
            tenure = buildings_df['tenure_type'].value_counts()
            colors = ['gold', 'silver', '#CD7F32', 'gray'][:len(tenure)]  # bronze as hex color
            explode = [0.05] * len(tenure)
            ax.pie(tenure.values, labels=tenure.index, autopct='%1.1f%%',
                  colors=colors, explode=explode)
            ax.set_title("Tenure Type (Owner vs Renter)", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "building_occupancy_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved occupancy analysis to {output_path}")
        plt.close(fig)
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all building visualizations."""
        self.create_building_type_distribution(buildings_df)
        self.create_geographic_distribution(buildings_df)
        self.create_energy_characteristics(buildings_df)
        self.create_building_occupancy_analysis(buildings_df)
        logger.info(f"Created all building visualizations in {self.output_dir}")