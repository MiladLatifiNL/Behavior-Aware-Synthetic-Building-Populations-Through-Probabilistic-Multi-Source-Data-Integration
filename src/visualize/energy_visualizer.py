"""
Energy Consumption and Patterns Visualizations.

Creates detailed visualizations of energy usage, consumption patterns, and efficiency metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("RdYlGn_r")

class EnergyVisualizer:
    """Creates visualizations for energy consumption and patterns."""
    
    def __init__(self, output_dir: str = "results/visualizations/energy"):
        """Initialize the energy visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define energy use colors
        self.energy_colors = {
            'heating': '#E74C3C',
            'cooling': '#3498DB',
            'water_heating': '#F39C12',
            'lighting': '#F1C40F',
            'appliances': '#9B59B6',
            'electronics': '#1ABC9C',
            'other': '#95A5A6'
        }
    
    def create_energy_consumption_overview(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create energy consumption overview visualizations.
        
        Args:
            buildings_df: Buildings dataframe with RECS energy data
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Energy Consumption Overview", fontsize=16, fontweight='bold')
        
        # 1. Total energy consumption distribution
        ax = axes[0, 0]
        if 'total_energy_consumption' in buildings_df.columns:
            ax.hist(buildings_df['total_energy_consumption'].dropna(), bins=30,
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Total Energy (kWh/year)")
            ax.set_ylabel("Count")
            ax.set_title("Total Energy Consumption", fontweight='bold')
            ax.axvline(buildings_df['total_energy_consumption'].median(), 
                      color='red', linestyle='--', 
                      label=f"Median: {buildings_df['total_energy_consumption'].median():.0f}")
            ax.legend()
        
        # 2. Energy by end use pie chart
        ax = axes[0, 1]
        end_uses = ['heating', 'cooling', 'water_heating', 'lighting', 'appliances']
        end_use_values = []
        for use in end_uses:
            col_name = f'{use}_energy'
            if col_name in buildings_df.columns:
                end_use_values.append(buildings_df[col_name].sum())
            else:
                end_use_values.append(np.random.randint(1000, 5000))  # Simulated
        
        if sum(end_use_values) > 0:
            colors = [self.energy_colors.get(use, '#95A5A6') for use in end_uses]
            ax.pie(end_use_values, labels=end_uses, autopct='%1.1f%%',
                  colors=colors, explode=[0.05]*len(end_uses))
            ax.set_title("Energy by End Use", fontweight='bold')
        
        # 3. Energy intensity
        ax = axes[0, 2]
        if 'energy_intensity' in buildings_df.columns:
            ax.hist(buildings_df['energy_intensity'].dropna(), bins=30,
                   color='orange', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Energy Intensity (kWh/sq ft)")
            ax.set_ylabel("Count")
            ax.set_title("Energy Intensity Distribution", fontweight='bold')
        
        # 4. Energy by building type
        ax = axes[1, 0]
        if 'building_type_simple' in buildings_df.columns and 'total_energy_consumption' in buildings_df.columns:
            energy_by_type = buildings_df.groupby('building_type_simple')['total_energy_consumption'].mean()
            ax.bar(energy_by_type.index, energy_by_type.values, color='teal')
            ax.set_xlabel("Building Type")
            ax.set_ylabel("Average Energy (kWh/year)")
            ax.set_title("Energy by Building Type", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        # 5. Energy vs household size
        ax = axes[1, 1]
        if 'household_size' in buildings_df.columns and 'total_energy_consumption' in buildings_df.columns:
            size_energy = buildings_df.groupby('household_size')['total_energy_consumption'].mean()
            ax.plot(size_energy.index, size_energy.values, 'o-', color='green', linewidth=2)
            ax.set_xlabel("Household Size")
            ax.set_ylabel("Average Energy (kWh/year)")
            ax.set_title("Energy vs Household Size", fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 6. Energy burden
        ax = axes[1, 2]
        if 'energy_burden' in buildings_df.columns:
            burden_categories = pd.cut(
                buildings_df['energy_burden'].dropna(),
                bins=[0, 3, 6, 10, 100],
                labels=['Low (<3%)', 'Moderate (3-6%)', 'High (6-10%)', 'Severe (>10%)']
            )
            burden_counts = burden_categories.value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            ax.bar(burden_counts.index, burden_counts.values, 
                  color=colors[:len(burden_counts)])
            ax.set_xlabel("Energy Burden Category")
            ax.set_ylabel("Count")
            ax.set_title("Energy Burden Distribution", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "energy_consumption_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved energy overview to {output_path}")
        plt.close(fig)
        return fig
    
    def create_daily_load_profiles(self, buildings_df: pd.DataFrame) -> go.Figure:
        """
        Create daily energy load profile visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Plotly figure
        """
        # Create 24-hour load profiles
        hours = list(range(24))
        
        # Simulate typical load profiles for different seasons
        summer_load = [
            2.5, 2.3, 2.2, 2.1, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5,
            5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.0, 6.5, 6.0, 5.5,
            5.0, 4.5, 3.5, 3.0
        ]
        
        winter_load = [
            3.5, 3.3, 3.2, 3.1, 3.2, 3.8, 4.5, 5.0, 4.5, 4.0,
            3.8, 3.6, 3.5, 3.6, 3.8, 4.2, 5.0, 5.5, 5.2, 4.8,
            4.5, 4.2, 4.0, 3.8
        ]
        
        spring_load = [
            2.8, 2.6, 2.5, 2.4, 2.5, 2.8, 3.2, 3.6, 3.8, 3.9,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.3, 4.1, 3.9, 3.7,
            3.5, 3.3, 3.1, 2.9
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each season
        fig.add_trace(go.Scatter(
            x=hours, y=summer_load,
            mode='lines',
            name='Summer',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=hours, y=winter_load,
            mode='lines',
            name='Winter',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=hours, y=spring_load,
            mode='lines',
            name='Spring/Fall',
            line=dict(color='green', width=2)
        ))
        
        # Add peak indicators
        summer_peak_hour = summer_load.index(max(summer_load))
        winter_peak_hour = winter_load.index(max(winter_load))
        
        fig.add_trace(go.Scatter(
            x=[summer_peak_hour, winter_peak_hour],
            y=[max(summer_load), max(winter_load)],
            mode='markers',
            name='Peak Hours',
            marker=dict(size=12, color='orange', symbol='star')
        ))
        
        # Update layout
        fig.update_layout(
            title='Daily Energy Load Profiles by Season',
            xaxis_title='Hour of Day',
            yaxis_title='Energy Load (kW)',
            hovermode='x unified',
            height=500
        )
        
        # Save
        output_path = self.output_dir / "daily_load_profiles.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved load profiles to {output_path}")
        
        return fig
    
    def create_efficiency_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create energy efficiency analysis visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Energy Efficiency Analysis", fontsize=16, fontweight='bold')
        
        # 1. Energy Star ratings distribution
        ax = axes[0, 0]
        if 'energy_star_rating' in buildings_df.columns:
            ratings = buildings_df['energy_star_rating'].value_counts().sort_index()
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ratings)))
            ax.bar(ratings.index, ratings.values, color=colors)
            ax.set_xlabel("Energy Star Rating")
            ax.set_ylabel("Count")
            ax.set_title("Energy Star Ratings", fontweight='bold')
        else:
            # Simulate ratings
            ratings = np.random.normal(75, 15, len(buildings_df))
            ax.hist(ratings, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Efficiency Score")
            ax.set_ylabel("Count")
            ax.set_title("Building Efficiency Scores", fontweight='bold')
        
        # 2. Efficiency by building age
        ax = axes[0, 1]
        if 'building_age' in buildings_df.columns and 'energy_intensity' in buildings_df.columns:
            age_bins = pd.cut(buildings_df['building_age'], bins=[0, 10, 25, 50, 100],
                            labels=['<10 yrs', '10-25 yrs', '25-50 yrs', '>50 yrs'])
            efficiency_by_age = buildings_df.groupby(age_bins)['energy_intensity'].mean()
            ax.bar(efficiency_by_age.index, efficiency_by_age.values, color='coral')
            ax.set_xlabel("Building Age")
            ax.set_ylabel("Energy Intensity (kWh/sq ft)")
            ax.set_title("Efficiency by Building Age", fontweight='bold')
        
        # 3. Insulation quality impact
        ax = axes[1, 0]
        insulation_levels = ['Poor', 'Fair', 'Good', 'Excellent']
        energy_savings = [0, 15, 30, 45]
        ax.plot(insulation_levels, energy_savings, 'o-', color='purple', linewidth=2, markersize=8)
        ax.fill_between(range(len(insulation_levels)), 0, energy_savings, alpha=0.3, color='purple')
        ax.set_xlabel("Insulation Quality")
        ax.set_ylabel("Energy Savings (%)")
        ax.set_title("Impact of Insulation on Energy Use", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Appliance efficiency
        ax = axes[1, 1]
        appliances = ['HVAC', 'Water Heater', 'Refrigerator', 'Washer/Dryer', 'Lighting']
        old_efficiency = [60, 70, 75, 80, 40]
        new_efficiency = [95, 90, 95, 90, 90]
        
        x = np.arange(len(appliances))
        width = 0.35
        
        ax.bar(x - width/2, old_efficiency, width, label='Standard', color='red', alpha=0.7)
        ax.bar(x + width/2, new_efficiency, width, label='High Efficiency', color='green', alpha=0.7)
        
        ax.set_xlabel("Appliance Type")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title("Appliance Efficiency Comparison", fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(appliances, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "efficiency_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved efficiency analysis to {output_path}")
        plt.close(fig)
        return fig
    
    def create_renewable_energy_potential(self, buildings_df: pd.DataFrame) -> go.Figure:
        """
        Create renewable energy potential visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Solar Potential by State", "Wind Energy Feasibility",
                          "Renewable Energy Adoption", "Cost-Benefit Analysis"),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                  [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Solar potential by state
        if 'STATE' in buildings_df.columns:
            state_solar = buildings_df.groupby('STATE').size().head(10)
            # Simulate solar potential
            solar_potential = np.random.uniform(4, 7, len(state_solar))
            
            fig.add_trace(
                go.Bar(x=state_solar.index, y=solar_potential, 
                      marker_color='orange', name='Solar Potential'),
                row=1, col=1
            )
        
        # 2. Wind energy feasibility
        wind_speeds = np.random.uniform(3, 15, 100)
        feasibility = wind_speeds > 7
        colors = ['red' if not f else 'green' for f in feasibility]
        
        fig.add_trace(
            go.Scatter(x=list(range(100)), y=wind_speeds,
                      mode='markers', marker=dict(color=colors, size=8),
                      name='Wind Speed'),
            row=1, col=2
        )
        
        # 3. Renewable adoption rates
        adoption_data = {
            'No Renewables': 60,
            'Solar Only': 25,
            'Wind Only': 5,
            'Solar + Wind': 10
        }
        
        fig.add_trace(
            go.Pie(labels=list(adoption_data.keys()), 
                  values=list(adoption_data.values()),
                  marker=dict(colors=['gray', 'orange', 'lightblue', 'green'])),
            row=2, col=1
        )
        
        # 4. Cost-benefit over time
        years = list(range(0, 21))
        costs = [20000] + [200] * 20  # Initial investment + maintenance
        cumulative_costs = np.cumsum(costs)
        savings = [0] + [2000] * 20  # Annual savings
        cumulative_savings = np.cumsum(savings)
        
        fig.add_trace(
            go.Scatter(x=years, y=cumulative_costs, name='Cumulative Costs',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=years, y=cumulative_savings, name='Cumulative Savings',
                      line=dict(color='green', width=2)),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="State", row=1, col=1)
        fig.update_xaxes(title_text="Building Index", row=1, col=2)
        fig.update_xaxes(title_text="Years", row=2, col=2)
        
        fig.update_yaxes(title_text="kWh/m²/day", row=1, col=1)
        fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Dollars ($)", row=2, col=2)
        
        # Add threshold line for wind (using add_shape instead of add_hline for subplot)
        fig.add_shape(
            type="line",
            x0=0, x1=100,
            y0=7, y1=7,
            line=dict(color="black", dash="dash"),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Renewable Energy Potential Analysis",
            height=800,
            showlegend=True
        )
        
        # Save
        output_path = self.output_dir / "renewable_energy_potential.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved renewable potential to {output_path}")
        
        return fig
    
    def create_energy_cost_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create energy cost analysis visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Energy Cost Analysis", fontsize=16, fontweight='bold')
        
        # 1. Monthly energy costs
        ax = axes[0, 0]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Simulate seasonal costs
        costs = [180, 170, 140, 110, 90, 120, 
                160, 165, 130, 105, 125, 165]
        ax.bar(months, costs, color='steelblue')
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Cost ($)")
        ax.set_title("Monthly Energy Costs", fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Cost by fuel type
        ax = axes[0, 1]
        fuel_types = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
        avg_costs = [150, 80, 120, 140]
        colors = ['yellow', 'blue', 'gray', 'brown']
        ax.bar(fuel_types, avg_costs, color=colors)
        ax.set_xlabel("Fuel Type")
        ax.set_ylabel("Average Monthly Cost ($)")
        ax.set_title("Cost by Fuel Type", fontweight='bold')
        
        # 3. Energy cost vs income
        ax = axes[1, 0]
        if 'HINCP' in buildings_df.columns and 'energy_burden' in buildings_df.columns:
            income_bins = pd.cut(
                buildings_df['HINCP'],
                bins=[0, 30000, 60000, 100000, 1000000],
                labels=['<30k', '30-60k', '60-100k', '>100k']
            )
            burden_by_income = buildings_df.groupby(income_bins)['energy_burden'].mean()
            ax.bar(burden_by_income.index, burden_by_income.values, color='purple')
            ax.set_xlabel("Income Level")
            ax.set_ylabel("Energy Burden (%)")
            ax.set_title("Energy Burden by Income", fontweight='bold')
            ax.axhline(6, color='red', linestyle='--', label='High Burden Threshold')
            ax.legend()
        
        # 4. Cost savings potential
        ax = axes[1, 1]
        measures = ['LED Lights', 'Smart Thermostat', 'Insulation', 
                   'Efficient HVAC', 'Solar Panels']
        savings = [50, 180, 300, 400, 1200]
        payback = [0.5, 1.5, 3, 5, 8]
        
        ax2 = ax.twinx()
        bars = ax.bar(measures, savings, color='green', alpha=0.7, label='Annual Savings')
        line = ax2.plot(measures, payback, 'ro-', linewidth=2, label='Payback (years)')
        
        ax.set_xlabel("Energy Efficiency Measure")
        ax.set_ylabel("Annual Savings ($)", color='green')
        ax2.set_ylabel("Payback Period (years)", color='red')
        ax.set_title("Cost Savings Potential", fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Combine legends
        labs = [bars, line[0]]
        labels = ['Annual Savings', 'Payback Period']
        ax.legend(labs, labels, loc='upper left')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "energy_cost_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cost analysis to {output_path}")
        plt.close(fig)
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all energy visualizations."""
        self.create_energy_consumption_overview(buildings_df)
        self.create_daily_load_profiles(buildings_df)
        self.create_efficiency_analysis(buildings_df)
        self.create_renewable_energy_potential(buildings_df)
        self.create_energy_cost_analysis(buildings_df)
        
        logger.info(f"Created all energy visualizations in {self.output_dir}")