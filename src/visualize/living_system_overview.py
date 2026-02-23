"""
Living System Overview Visualizations.

Creates high-level visualizations showing the complete living system
data flow and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SystemOverviewVisualizer:
    """Creates overview visualizations of the living system."""
    
    def __init__(self, output_dir: str = "results/visualizations/overview"):
        """Initialize the overview visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_data_flow_sankey(self, metadata: Dict) -> go.Figure:
        """
        Create a Sankey diagram showing data flow through the 4 phases.
        
        Args:
            metadata: Dictionary with phase statistics
            
        Returns:
            Plotly figure object
        """
        # Define nodes
        labels = [
            "PUMS Households", "PUMS Persons",  # Sources
            "Phase 1: Buildings",  # Phase 1 output
            "RECS Templates",  # Phase 2 input
            "Phase 2: Buildings+RECS",  # Phase 2 output
            "ATUS Activities",  # Phase 3 input
            "Phase 3: Buildings+Activities",  # Phase 3 output
            "Weather Data",  # Phase 4 input
            "Phase 4: Complete Living System"  # Final output
        ]
        
        # Define links (source -> target with values)
        source = [0, 1, 2, 3, 4, 5, 6, 7]
        target = [2, 2, 4, 4, 6, 6, 8, 8]
        
        # Get actual values from metadata if available
        values = [
            metadata.get('households', 100),
            metadata.get('persons', 250),
            metadata.get('phase1_buildings', 100),
            metadata.get('recs_templates', 18000),
            metadata.get('phase2_buildings', 100),
            metadata.get('atus_respondents', 8548),
            metadata.get('phase3_buildings', 100),
            metadata.get('weather_records', 35040)
        ]
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            title="Living System Data Flow - 4 Phase Pipeline",
            font_size=12,
            height=600,
            width=1200
        )
        
        # Save
        output_path = self.output_dir / "data_flow_sankey.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved Sankey diagram to {output_path}")
        
        return fig
    
    def create_system_metrics_dashboard(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create a dashboard showing key system metrics.
        
        Args:
            buildings_df: Final integrated buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Living System Overview Dashboard", fontsize=20, fontweight='bold')
        
        # Calculate metrics
        total_buildings = len(buildings_df)
        total_persons = sum(len(b['persons']) if isinstance(b.get('persons', []), list) else 0 
                          for _, b in buildings_df.iterrows())
        
        total_activities = 0
        persons_with_activities = 0
        for _, building in buildings_df.iterrows():
            if isinstance(building.get('persons', []), list):
                for person in building['persons']:
                    if isinstance(person, dict) and person.get('has_activities'):
                        persons_with_activities += 1
                        if 'activity_sequence' in person:
                            total_activities += len(person['activity_sequence'])
        
        # Create metric cards
        ax1 = plt.subplot(2, 4, 1)
        self._create_metric_card(ax1, "Buildings", total_buildings, "#1f77b4")
        
        ax2 = plt.subplot(2, 4, 2)
        self._create_metric_card(ax2, "Persons", total_persons, "#ff7f0e")
        
        ax3 = plt.subplot(2, 4, 3)
        self._create_metric_card(ax3, "Activities", total_activities, "#2ca02c")
        
        ax4 = plt.subplot(2, 4, 4)
        coverage = (persons_with_activities / max(total_persons, 1)) * 100
        self._create_metric_card(ax4, "Coverage", f"{coverage:.1f}%", "#d62728")
        
        # Building type distribution
        ax5 = plt.subplot(2, 4, 5)
        if 'building_type' in buildings_df.columns:
            building_types = buildings_df['building_type'].value_counts()
            ax5.pie(building_types.values, labels=building_types.index, autopct='%1.1f%%')
            ax5.set_title("Building Types", fontweight='bold')
        else:
            ax5.text(0.5, 0.5, "No building type data", ha='center', va='center')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
        
        # Household size distribution
        ax6 = plt.subplot(2, 4, 6)
        if 'household_size' in buildings_df.columns:
            sizes = buildings_df['household_size'].value_counts().sort_index()
            ax6.bar(sizes.index, sizes.values, color='#9467bd')
            ax6.set_xlabel("Household Size")
            ax6.set_ylabel("Count")
            ax6.set_title("Household Sizes", fontweight='bold')
        
        # Geographic distribution
        ax7 = plt.subplot(2, 4, 7)
        if 'STATE' in buildings_df.columns:
            states = buildings_df['STATE'].value_counts().head(10)
            ax7.barh(states.index, states.values, color='#8c564b')
            ax7.set_xlabel("Count")
            ax7.set_title("Top 10 States", fontweight='bold')
        
        # Processing timeline
        ax8 = plt.subplot(2, 4, 8)
        phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
        times = [20, 3, 5, 8]  # Example times in seconds
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ax8.bar(phases, times, color=colors)
        ax8.set_ylabel("Time (seconds)")
        ax8.set_title("Processing Time by Phase", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "system_metrics_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dashboard to {output_path}")
        plt.close(fig)
        return fig
    
    def create_relationship_network(self, buildings_df: pd.DataFrame, sample_size: int = 5) -> plt.Figure:
        """
        Create a network graph showing Building-Person-Activity relationships.
        
        Args:
            buildings_df: Buildings dataframe
            sample_size: Number of buildings to visualize
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Sample buildings for clarity
        sample_df = buildings_df.head(sample_size)
        
        # Define positions
        building_y = 0.5
        person_y_start = 0.2
        person_y_end = 0.8
        activity_x = 0.8
        
        colors = plt.cm.Set3(np.linspace(0, 1, sample_size))
        
        for idx, (_, building) in enumerate(sample_df.iterrows()):
            building_id = building.get('building_id', f'B{idx}')
            
            # Draw building node
            building_x = 0.2
            building_node = Circle((building_x, building_y), 0.03, 
                                  color=colors[idx], alpha=0.7)
            ax.add_patch(building_node)
            ax.text(building_x, building_y - 0.05, building_id[:10], 
                   ha='center', fontsize=8)
            
            # Draw persons and activities
            if isinstance(building.get('persons', []), list):
                persons = building['persons']
                n_persons = len(persons)
                
                if n_persons > 0:
                    person_ys = np.linspace(person_y_start, person_y_end, n_persons)
                    
                    for p_idx, person in enumerate(persons):
                        if not isinstance(person, dict):
                            continue
                        
                        person_x = 0.5
                        person_y = person_ys[p_idx]
                        
                        # Draw person node
                        person_node = Circle((person_x, person_y), 0.02, 
                                           color=colors[idx], alpha=0.5)
                        ax.add_patch(person_node)
                        
                        # Connect building to person
                        arrow1 = FancyArrowPatch((building_x + 0.03, building_y),
                                               (person_x - 0.02, person_y),
                                               arrowstyle='->', lw=0.5,
                                               color=colors[idx], alpha=0.3)
                        ax.add_patch(arrow1)
                        
                        # Add person info
                        age = person.get('AGEP', 'Unknown')
                        ax.text(person_x, person_y - 0.03, f"Age:{age}", 
                               fontsize=6, ha='center')
                        
                        # Draw activities
                        if person.get('has_activities') and 'activity_sequence' in person:
                            activities = person['activity_sequence'][:3]  # First 3 activities
                            
                            for a_idx, activity in enumerate(activities):
                                if not isinstance(activity, dict):
                                    continue
                                
                                activity_y = person_y + (a_idx - 1) * 0.05
                                
                                # Draw activity node
                                activity_node = Circle((activity_x, activity_y), 0.015, 
                                                     color='#2ca02c', alpha=0.3)
                                ax.add_patch(activity_node)
                                
                                # Connect person to activity
                                arrow2 = FancyArrowPatch((person_x + 0.02, person_y),
                                                       (activity_x - 0.015, activity_y),
                                                       arrowstyle='->', lw=0.3,
                                                       color='gray', alpha=0.3)
                                ax.add_patch(arrow2)
        
        # Add labels
        ax.text(0.2, 0.9, "Buildings", fontsize=14, fontweight='bold', ha='center')
        ax.text(0.5, 0.9, "Persons", fontsize=14, fontweight='bold', ha='center')
        ax.text(0.8, 0.9, "Activities", fontsize=14, fontweight='bold', ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Living System Relationship Network", fontsize=16, fontweight='bold')
        
        # Save
        output_path = self.output_dir / "relationship_network.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network graph to {output_path}")
        plt.close(fig)
        return fig
    
    def _create_metric_card(self, ax, title: str, value, color: str):
        """Create a metric card visualization."""
        ax.add_patch(FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.3,
                                    edgecolor=color, linewidth=2))
        
        ax.text(0.5, 0.65, str(value), fontsize=24, fontweight='bold',
               ha='center', va='center')
        ax.text(0.5, 0.35, title, fontsize=12, ha='center', va='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


def create_system_overview(buildings_df: pd.DataFrame, metadata: Dict = None,
                          output_dir: str = "results/visualizations/overview"):
    """
    Create all system overview visualizations.
    
    Args:
        buildings_df: Final integrated buildings dataframe
        metadata: Optional metadata dictionary
        output_dir: Output directory for visualizations
    """
    visualizer = SystemOverviewVisualizer(output_dir)
    
    # Default metadata if not provided
    if metadata is None:
        metadata = {
            'households': len(buildings_df),
            'persons': sum(len(b.get('persons', [])) if isinstance(b.get('persons', []), list) else 0 
                         for _, b in buildings_df.iterrows()),
            'phase1_buildings': len(buildings_df),
            'recs_templates': 18000,
            'phase2_buildings': len(buildings_df),
            'atus_respondents': 8548,
            'phase3_buildings': len(buildings_df),
            'weather_records': 35040
        }
    
    # Create visualizations
    visualizer.create_data_flow_sankey(metadata)
    visualizer.create_system_metrics_dashboard(buildings_df)
    visualizer.create_relationship_network(buildings_df)
    
    logger.info(f"Created system overview visualizations in {output_dir}")