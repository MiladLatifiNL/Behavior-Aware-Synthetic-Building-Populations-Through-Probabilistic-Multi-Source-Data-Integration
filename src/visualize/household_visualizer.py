"""
Household Dynamics and Interactions Visualizations.

Creates detailed visualizations of household dynamics, interactions, and coordinated behaviors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class HouseholdVisualizer:
    """Creates visualizations for household dynamics and interactions."""
    
    def __init__(self, output_dir: str = "results/visualizations/households"):
        """Initialize the household visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define activity colors for consistency
        self.activity_colors = {
            'sleep': '#2C3E50',
            'work': '#E74C3C', 
            'school': '#3498DB',
            'childcare': '#16A085',
            'household': '#F39C12',
            'leisure': '#27AE60',
            'eating': '#E67E22',
            'personal': '#8E44AD'
        }
    
    def create_household_structure_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create household structure analysis visualizations.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Household Structure Analysis", fontsize=16, fontweight='bold')
        
        # 1. Household size distribution
        ax = axes[0, 0]
        if 'household_size' in buildings_df.columns:
            sizes = buildings_df['household_size'].value_counts().sort_index()
            ax.bar(sizes.index, sizes.values, color='steelblue', edgecolor='black')
            ax.set_xlabel("Household Size")
            ax.set_ylabel("Count")
            ax.set_title("Household Size Distribution", fontweight='bold')
            ax.set_xticks(sizes.index)
        
        # 2. Family types
        ax = axes[0, 1]
        if 'household_composition' in buildings_df.columns:
            comp = buildings_df['household_composition'].value_counts()
            ax.pie(comp.values, labels=comp.index, autopct='%1.1f%%',
                  colors=plt.cm.Pastel1(np.linspace(0, 1, len(comp))))
            ax.set_title("Family Types", fontweight='bold')
        
        # 3. Number of workers per household
        ax = axes[0, 2]
        if 'num_workers' in buildings_df.columns:
            workers = buildings_df['num_workers'].value_counts().sort_index()
            ax.bar(workers.index, workers.values, color='green', alpha=0.7)
            ax.set_xlabel("Number of Workers")
            ax.set_ylabel("Count")
            ax.set_title("Workers per Household", fontweight='bold')
        
        # 4. Age diversity within households
        ax = axes[1, 0]
        age_diversity = []
        for _, building in buildings_df.iterrows():
            if 'persons' in building and isinstance(building['persons'], list):
                ages = [p.get('AGEP', 0) for p in building['persons'] if isinstance(p, dict)]
                if len(ages) > 1:
                    age_diversity.append(np.std(ages))
        
        if age_diversity:
            ax.hist(age_diversity, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Age Standard Deviation")
            ax.set_ylabel("Count")
            ax.set_title("Age Diversity in Households", fontweight='bold')
        
        # 5. Income distribution by household type
        ax = axes[1, 1]
        if 'household_composition' in buildings_df.columns and 'HINCP' in buildings_df.columns:
            comp_income = buildings_df.groupby('household_composition')['HINCP'].mean()
            ax.barh(comp_income.index, comp_income.values, color='gold')
            ax.set_xlabel("Average Income ($)")
            ax.set_title("Income by Household Type", fontweight='bold')
        
        # 6. Multigenerational households
        ax = axes[1, 2]
        if 'multigenerational' in buildings_df.columns:
            multi = buildings_df['multigenerational'].value_counts()
            labels = ['Single Generation', 'Multi-generational'][:len(multi)]
            colors = ['lightblue', 'coral'][:len(multi)]
            explode = [0] * len(multi)
            if len(multi) > 1:
                explode[1] = 0.1
            ax.pie(multi.values, labels=labels,
                  autopct='%1.1f%%', colors=colors, explode=explode)
            ax.set_title("Multi-generational Households", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "household_structure_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved household structure to {output_path}")
        
        return fig
    
    def create_coordinated_activities_timeline(self, buildings_df: pd.DataFrame,
                                             sample_household: int = 0) -> plt.Figure:
        """
        Create coordinated household activities timeline.
        
        Args:
            buildings_df: Buildings dataframe
            sample_household: Index of household to visualize
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        if sample_household >= len(buildings_df):
            ax1.text(0.5, 0.5, "Household index out of range", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        building = buildings_df.iloc[sample_household]
        
        if 'persons' not in building or not isinstance(building['persons'], list):
            ax1.text(0.5, 0.5, "No persons in selected household", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Main timeline plot
        persons = building['persons']
        n_persons = len(persons)
        
        # Track household-level activities
        household_activities = {
            'meals': [(420, 30), (720, 45), (1080, 60)],  # Breakfast, lunch, dinner
            'sleep': [(0, 360), (1320, 120)],  # Night sleep
            'family_time': [(1140, 120)]  # Evening family time
        }
        
        # Plot each person's activities
        for p_idx, person in enumerate(persons):
            if not isinstance(person, dict):
                continue
            
            y_pos = n_persons - p_idx - 1
            
            # Draw person activities
            if person.get('has_activities') and 'activity_sequence' in person:
                for activity in person['activity_sequence']:
                    if not isinstance(activity, dict):
                        continue
                    
                    start = activity.get('start_minute', 0)
                    duration = activity.get('duration_minutes', 20)
                    activity_type = activity.get('activity_category', 'other')
                    
                    color = self.activity_colors.get(activity_type, '#95A5A6')
                    
                    rect = Rectangle((start, y_pos - 0.4), duration, 0.8,
                                   facecolor=color, edgecolor='black', 
                                   linewidth=0.5, alpha=0.8)
                    ax1.add_patch(rect)
            
            # Add person label
            age = person.get('AGEP', 'Unknown')
            sex = 'M' if person.get('SEX') == 1 else 'F'
            employed = 'W' if person.get('is_employed') else ''
            label = f"P{p_idx+1} ({age}{sex}{employed})"
            ax1.text(-50, y_pos, label, ha='right', va='center', fontsize=10)
        
        # Highlight coordinated activities
        for activity_type, periods in household_activities.items():
            for start, duration in periods:
                if activity_type == 'meals':
                    ax1.axvspan(start, start + duration, alpha=0.1, color='orange')
                elif activity_type == 'sleep':
                    ax1.axvspan(start, start + duration, alpha=0.1, color='blue')
                elif activity_type == 'family_time':
                    ax1.axvspan(start, start + duration, alpha=0.1, color='green')
        
        # Format main plot
        ax1.set_xlim(0, 1440)
        ax1.set_ylim(-0.5, n_persons - 0.5)
        ax1.set_ylabel("Household Members")
        ax1.set_title(f"Coordinated Household Activities - Household {sample_household+1}", 
                     fontsize=14, fontweight='bold')
        
        # Set x-axis hours
        hour_ticks = np.arange(0, 1441, 120)
        hour_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]
        ax1.set_xticks(hour_ticks)
        ax1.set_xticklabels([])
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        legend_elements = [mpatches.Patch(color=color, label=activity.replace('_', ' ').title())
                          for activity, color in self.activity_colors.items()]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        
        # Lower plot: Household coordination score
        coordination_scores = []
        for minute in range(0, 1440, 30):  # Check every 30 minutes
            # Count how many people are doing same activity
            activities_at_time = []
            for person in persons:
                if isinstance(person, dict) and person.get('has_activities'):
                    if 'activity_sequence' in person:
                        for act in person['activity_sequence']:
                            if isinstance(act, dict):
                                start = act.get('start_minute', 0)
                                end = start + act.get('duration_minutes', 20)
                                if start <= minute < end:
                                    activities_at_time.append(act.get('activity_category', 'other'))
                                    break
            
            # Calculate coordination (% doing same activity)
            if activities_at_time:
                most_common = max(set(activities_at_time), key=activities_at_time.count)
                coordination = activities_at_time.count(most_common) / len(persons) * 100
            else:
                coordination = 0
            
            coordination_scores.append(coordination)
        
        minutes = list(range(0, 1440, 30))
        ax2.fill_between(minutes, coordination_scores, alpha=0.5, color='purple')
        ax2.plot(minutes, coordination_scores, color='purple', linewidth=1)
        ax2.set_xlim(0, 1440)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Time of Day")
        ax2.set_ylabel("Coordination %")
        ax2.set_title("Household Activity Coordination Level", fontsize=12)
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(hour_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.close(fig)
        
        # Save
        output_path = self.output_dir / "coordinated_activities_timeline.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved coordinated timeline to {output_path}")
        
        return fig
    
    def create_household_interaction_network(self, buildings_df: pd.DataFrame,
                                            sample_households: int = 5) -> go.Figure:
        """
        Create household member interaction network.
        
        Args:
            buildings_df: Buildings dataframe
            sample_households: Number of households to visualize
            
        Returns:
            Plotly figure
        """
        # Create network graph
        G = nx.Graph()
        
        node_colors = []
        node_sizes = []
        node_labels = []
        
        # Process sample households
        for h_idx in range(min(sample_households, len(buildings_df))):
            building = buildings_df.iloc[h_idx]
            
            if 'persons' not in building or not isinstance(building['persons'], list):
                continue
            
            household_id = f"H{h_idx}"
            
            # Add household node
            G.add_node(household_id)
            node_colors.append('red')
            node_sizes.append(30)
            node_labels.append(f"Household {h_idx+1}")
            
            # Add person nodes
            for p_idx, person in enumerate(building['persons']):
                if not isinstance(person, dict):
                    continue
                
                person_id = f"{household_id}_P{p_idx}"
                age = person.get('AGEP', 0)
                
                G.add_node(person_id)
                node_colors.append('lightblue')
                node_sizes.append(20)
                node_labels.append(f"Age {age}")
                
                # Connect to household
                G.add_edge(household_id, person_id, weight=2)
                
                # Connect persons within household
                if p_idx > 0:
                    prev_person_id = f"{household_id}_P{p_idx-1}"
                    G.add_edge(person_id, prev_person_id, weight=1)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Node Type',
                    xanchor='left'
                ),
                line_width=2
            ),
            text=node_labels,
            textposition="top center"
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Household Interaction Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700
                       ))
        
        # Save
        output_path = self.output_dir / "household_interaction_network.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved interaction network to {output_path}")
        
        return fig
    
    def create_resource_sharing_patterns(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create resource sharing patterns visualization.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Household Resource Sharing Patterns", fontsize=16, fontweight='bold')
        
        # 1. Vehicle sharing
        ax = axes[0, 0]
        if 'num_vehicles' in buildings_df.columns and 'num_workers' in buildings_df.columns:
            vehicles_per_worker = buildings_df['num_vehicles'] / buildings_df['num_workers'].replace(0, 1)
            categories = pd.cut(vehicles_per_worker, bins=[0, 0.5, 1, 1.5, 10],
                              labels=['High Sharing', 'Moderate Sharing', 'Low Sharing', 'No Sharing'])
            counts = categories.value_counts()
            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                  colors=['green', 'yellow', 'orange', 'red'])
            ax.set_title("Vehicle Sharing Patterns", fontweight='bold')
        
        # 2. Space utilization
        ax = axes[0, 1]
        if 'num_rooms' in buildings_df.columns and 'household_size' in buildings_df.columns:
            rooms_per_person = buildings_df['num_rooms'] / buildings_df['household_size']
            ax.hist(rooms_per_person.dropna(), bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Rooms per Person")
            ax.set_ylabel("Count")
            ax.set_title("Space Utilization", fontweight='bold')
            ax.axvline(2, color='green', linestyle='--', label='Comfortable (2+)')
            ax.axvline(1, color='red', linestyle='--', label='Crowded (<1)')
            ax.legend()
        
        # 3. Income pooling (simulated)
        ax = axes[1, 0]
        pooling_types = ['Full Pooling', 'Partial Pooling', 'Independent', 'Mixed']
        pooling_counts = [40, 30, 20, 10]
        ax.bar(pooling_types, pooling_counts, color='purple')
        ax.set_xlabel("Income Pooling Type")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Household Income Pooling", fontweight='bold')
        
        # 4. Shared activities frequency
        ax = axes[1, 1]
        activities = ['Meals', 'Entertainment', 'Chores', 'Transportation', 'Shopping']
        sharing_freq = [85, 60, 70, 45, 55]  # Simulated percentages
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(activities)))
        bars = ax.barh(activities, sharing_freq, color=colors)
        ax.set_xlabel("Sharing Frequency (%)")
        ax.set_title("Shared Activity Frequency", fontweight='bold')
        
        # Add value labels on bars
        for bar, freq in zip(bars, sharing_freq):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{freq}%', va='center')
        
        plt.tight_layout()
        plt.close(fig)
        
        # Save
        output_path = self.output_dir / "resource_sharing_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved resource sharing to {output_path}")
        
        return fig
    
    def create_lifecycle_stage_analysis(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create household lifecycle stage analysis.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Household Lifecycle Stage Analysis", fontsize=16, fontweight='bold')
        
        # Determine lifecycle stages
        lifecycle_stages = []
        for _, building in buildings_df.iterrows():
            if 'persons' not in building or not isinstance(building['persons'], list):
                lifecycle_stages.append('Unknown')
                continue
            
            ages = [p.get('AGEP', 0) for p in building['persons'] if isinstance(p, dict)]
            if not ages:
                lifecycle_stages.append('Unknown')
                continue
            
            min_age = min(ages)
            max_age = max(ages)
            n_persons = len(ages)
            
            if n_persons == 1:
                if max_age < 35:
                    stage = 'Young Single'
                elif max_age < 65:
                    stage = 'Middle-aged Single'
                else:
                    stage = 'Senior Single'
            elif n_persons == 2:
                if min_age < 18:
                    stage = 'Single Parent'
                elif max_age < 35:
                    stage = 'Young Couple'
                elif max_age < 65:
                    stage = 'Middle-aged Couple'
                else:
                    stage = 'Senior Couple'
            else:
                if min_age < 18:
                    stage = 'Family with Children'
                else:
                    stage = 'Extended Family'
            
            lifecycle_stages.append(stage)
        
        buildings_df['lifecycle_stage'] = lifecycle_stages
        
        # 1. Lifecycle stage distribution
        ax = axes[0, 0]
        stage_counts = pd.Series(lifecycle_stages).value_counts()
        ax.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%',
              colors=plt.cm.Set3(np.linspace(0, 1, len(stage_counts))))
        ax.set_title("Lifecycle Stage Distribution", fontweight='bold')
        
        # 2. Income by lifecycle stage
        ax = axes[0, 1]
        if 'HINCP' in buildings_df.columns:
            income_by_stage = buildings_df.groupby('lifecycle_stage')['HINCP'].mean()
            ax.barh(income_by_stage.index, income_by_stage.values, color='gold')
            ax.set_xlabel("Average Income ($)")
            ax.set_title("Income by Lifecycle Stage", fontweight='bold')
        
        # 3. Energy use by lifecycle stage
        ax = axes[1, 0]
        if 'total_energy_consumption' in buildings_df.columns:
            energy_by_stage = buildings_df.groupby('lifecycle_stage')['total_energy_consumption'].mean()
            ax.bar(range(len(energy_by_stage)), energy_by_stage.values, color='orange')
            ax.set_xticks(range(len(energy_by_stage)))
            ax.set_xticklabels(energy_by_stage.index, rotation=45, ha='right')
            ax.set_ylabel("Average Energy (kWh/year)")
            ax.set_title("Energy Use by Lifecycle Stage", fontweight='bold')
        
        # 4. Household size by lifecycle stage
        ax = axes[1, 1]
        size_by_stage = buildings_df.groupby('lifecycle_stage')['household_size'].mean()
        ax.scatter(range(len(size_by_stage)), size_by_stage.values, 
                  s=200, c=range(len(size_by_stage)), cmap='viridis')
        ax.set_xticks(range(len(size_by_stage)))
        ax.set_xticklabels(size_by_stage.index, rotation=45, ha='right')
        ax.set_ylabel("Average Household Size")
        ax.set_title("Household Size by Lifecycle Stage", fontweight='bold')
        
        # Add value labels
        for i, (stage, size) in enumerate(size_by_stage.items()):
            ax.text(i, size + 0.1, f'{size:.1f}', ha='center')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "lifecycle_stage_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved lifecycle analysis to {output_path}")
        
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all household visualizations."""
        self.create_household_structure_analysis(buildings_df)
        self.create_coordinated_activities_timeline(buildings_df)
        self.create_household_interaction_network(buildings_df)
        self.create_resource_sharing_patterns(buildings_df)
        self.create_lifecycle_stage_analysis(buildings_df)
        
        logger.info(f"Created all household visualizations in {self.output_dir}")