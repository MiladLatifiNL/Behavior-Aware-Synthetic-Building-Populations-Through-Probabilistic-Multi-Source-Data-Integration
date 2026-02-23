"""
Activity Pattern Visualizations.

Creates detailed visualizations of activity patterns, schedules, and daily routines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
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
sns.set_palette("husl")

class ActivityVisualizer:
    """Creates visualizations for activity patterns and schedules."""
    
    def __init__(self, output_dir: str = "results/visualizations/activities"):
        """Initialize the activity visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define activity colors for consistency
        self.activity_colors = {
            'sleep': '#2C3E50',
            'personal_care': '#8E44AD',
            'work': '#E74C3C',
            'education': '#3498DB',
            'household': '#F39C12',
            'caring': '#16A085',
            'leisure': '#27AE60',
            'eating': '#E67E22',
            'travel': '#95A5A6',
            'shopping': '#D35400',
            'other': '#BDC3C7'
        }
    
    def extract_activities_from_persons(self, buildings_df: pd.DataFrame) -> List[Dict]:
        """Extract all activities from all persons."""
        activities_list = []
        
        for b_idx, building in buildings_df.iterrows():
            if 'persons' not in building or not isinstance(building['persons'], list):
                continue
            
            for p_idx, person in enumerate(building['persons']):
                if not isinstance(person, dict):
                    continue
                
                if person.get('has_activities') and 'activity_sequence' in person:
                    person_id = f"B{b_idx}_P{p_idx}"
                    
                    for activity in person['activity_sequence']:
                        if isinstance(activity, dict):
                            activity_data = activity.copy()
                            activity_data['person_id'] = person_id
                            activity_data['building_idx'] = b_idx
                            activity_data['person_idx'] = p_idx
                            activity_data['person_age'] = person.get('AGEP', None)
                            activity_data['person_sex'] = person.get('SEX', None)
                            
                            # Convert start_time to start_minute if needed
                            if 'start_time' in activity_data and 'start_minute' not in activity_data:
                                time_parts = activity_data['start_time'].split(':')
                                if len(time_parts) >= 2:
                                    activity_data['start_minute'] = int(time_parts[0]) * 60 + int(time_parts[1])
                            
                            # Map activity code to category
                            code = activity_data.get('activity_code', '')
                            if code.startswith('01'):
                                activity_data['activity_category'] = 'sleep'
                            elif code.startswith('02'):
                                activity_data['activity_category'] = 'personal_care'
                            elif code.startswith('05'):
                                activity_data['activity_category'] = 'work'
                            elif code.startswith('03'):
                                activity_data['activity_category'] = 'caring'
                            elif code.startswith('11'):
                                activity_data['activity_category'] = 'eating'
                            elif code.startswith('12') or code.startswith('13'):
                                activity_data['activity_category'] = 'leisure'
                            elif code.startswith('18'):
                                activity_data['activity_category'] = 'travel'
                            else:
                                activity_data['activity_category'] = 'other'
                            
                            activities_list.append(activity_data)
        
        return activities_list
    
    def create_daily_activity_timeline(self, activities: List[Dict], 
                                      sample_persons: int = 10) -> plt.Figure:
        """
        Create a timeline showing daily activities for sample persons.
        
        Args:
            activities: List of activity dictionaries
            sample_persons: Number of persons to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Group activities by person
        activities_df = pd.DataFrame(activities)
        if len(activities_df) == 0:
            ax.text(0.5, 0.5, "No activity data available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Get unique persons
        unique_persons = activities_df['person_id'].unique()[:sample_persons]
        
        # Plot each person's activities
        for y_pos, person_id in enumerate(unique_persons):
            person_activities = activities_df[activities_df['person_id'] == person_id].sort_values('start_minute')
            
            for _, activity in person_activities.iterrows():
                # Get start time
                if 'start_minute' in activity:
                    start = activity['start_minute']
                elif 'start_time' in activity:
                    time_parts = activity['start_time'].split(':')
                    start = int(time_parts[0]) * 60 + int(time_parts[1]) if len(time_parts) >= 2 else 0
                else:
                    start = 0
                
                duration = activity.get('duration_minutes', 20)
                activity_type = activity.get('activity_category', 'other')
                
                # Map activity to color
                color = self.activity_colors.get(activity_type, self.activity_colors['other'])
                
                # Draw rectangle for activity
                rect = Rectangle((start, y_pos - 0.4), duration, 0.8,
                               facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Format axes
        ax.set_xlim(0, 1440)  # 24 hours in minutes
        ax.set_ylim(-0.5, len(unique_persons) - 0.5)
        
        # Set x-axis to show hours
        hour_ticks = np.arange(0, 1441, 120)  # Every 2 hours
        hour_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels, rotation=45)
        
        # Set y-axis
        ax.set_yticks(range(len(unique_persons)))
        ax.set_yticklabels([f"Person {i+1}" for i in range(len(unique_persons))])
        
        # Labels and title
        ax.set_xlabel("Time of Day", fontsize=12)
        ax.set_ylabel("Person", fontsize=12)
        ax.set_title("Daily Activity Timeline - Sample Persons", fontsize=16, fontweight='bold')
        
        # Add legend
        legend_patches = [mpatches.Patch(color=color, label=activity.replace('_', ' ').title()) 
                         for activity, color in self.activity_colors.items()]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "daily_activity_timeline.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved daily timeline to {output_path}")
        plt.close(fig)
        return fig
    
    def create_aggregate_activity_patterns(self, activities: List[Dict]) -> plt.Figure:
        """
        Create aggregate activity patterns across all persons.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Aggregate Activity Patterns", fontsize=16, fontweight='bold')
        
        activities_df = pd.DataFrame(activities)
        
        if len(activities_df) == 0:
            return fig
        
        # 1. Activity type distribution (pie chart)
        ax = axes[0, 0]
        if 'activity_category' in activities_df.columns:
            activity_dist = activities_df['activity_category'].value_counts()
            colors = [self.activity_colors.get(cat, self.activity_colors['other']) 
                     for cat in activity_dist.index]
            ax.pie(activity_dist.values, labels=activity_dist.index, autopct='%1.1f%%',
                  colors=colors)
            ax.set_title("Activity Type Distribution", fontweight='bold')
        
        # 2. Activity duration by type (box plot)
        ax = axes[0, 1]
        if 'activity_category' in activities_df.columns and 'duration_minutes' in activities_df.columns:
            top_activities = activities_df['activity_category'].value_counts().head(8).index
            data_for_box = []
            labels_for_box = []
            
            for activity in top_activities:
                durations = activities_df[activities_df['activity_category'] == activity]['duration_minutes']
                if len(durations) > 0:
                    data_for_box.append(durations.values)
                    labels_for_box.append(activity.replace('_', ' ').title())
            
            if data_for_box:
                bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
                for patch, activity in zip(bp['boxes'], top_activities):
                    patch.set_facecolor(self.activity_colors.get(activity, self.activity_colors['other']))
                
                ax.set_xlabel("Activity Type")
                ax.set_ylabel("Duration (minutes)")
                ax.set_title("Activity Duration Distribution", fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        
        # 3. Hourly activity distribution (stacked area chart)
        ax = axes[1, 0]
        if 'start_minute' in activities_df.columns and 'activity_category' in activities_df.columns:
            # Convert start_minute to hour
            activities_df['hour'] = activities_df['start_minute'] // 60
            
            # Count activities by hour and type
            hourly_activities = (
                activities_df.groupby(['hour', 'activity_category'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            
            # Plot stacked area
            hours = hourly_activities.index
            bottom = np.zeros(len(hours))
            
            for activity in hourly_activities.columns:
                color = self.activity_colors.get(activity, self.activity_colors['other'])
                ax.fill_between(hours, bottom, bottom + hourly_activities[activity].values,
                              label=activity.replace('_', ' ').title(), color=color, alpha=0.7)
                bottom += hourly_activities[activity].values
            
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Number of Activities")
            ax.set_title("24-Hour Activity Distribution", fontweight='bold')
            ax.set_xlim(0, 23)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        
        # 4. Peak activity times
        ax = axes[1, 1]
        if 'start_minute' in activities_df.columns:
            activities_df['hour'] = activities_df['start_minute'] // 60
            peak_hours = activities_df['hour'].value_counts().sort_index()
            
            ax.bar(peak_hours.index, peak_hours.values, color='steelblue', edgecolor='black')
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Activity Count")
            ax.set_title("Peak Activity Times", fontweight='bold')
            ax.set_xlim(-0.5, 23.5)
            
            # Highlight peak hours
            peak_hour = peak_hours.idxmax()
            ax.axvline(peak_hour, color='red', linestyle='--', 
                      label=f'Peak: {peak_hour}:00')
            ax.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "aggregate_activity_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved aggregate patterns to {output_path}")
        plt.close(fig)
        return fig
    
    def create_activity_transition_matrix(self, activities: List[Dict]) -> go.Figure:
        """
        Create activity transition matrix heatmap.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Plotly figure
        """
        activities_df = pd.DataFrame(activities)
        
        if len(activities_df) == 0 or 'activity_category' not in activities_df.columns:
            return None
        
        # Sort by person and time
        activities_df = activities_df.sort_values(['person_id', 'start_minute'])
        
        # Create transition matrix
        transitions = {}
        
        for person_id in activities_df['person_id'].unique():
            person_acts = activities_df[activities_df['person_id'] == person_id]['activity_category'].values
            
            for i in range(len(person_acts) - 1):
                from_act = person_acts[i]
                to_act = person_acts[i + 1]
                
                if from_act not in transitions:
                    transitions[from_act] = {}
                if to_act not in transitions[from_act]:
                    transitions[from_act][to_act] = 0
                transitions[from_act][to_act] += 1
        
        # Convert to matrix
        activities_list = list(set(activities_df['activity_category'].unique()))
        matrix = np.zeros((len(activities_list), len(activities_list)))
        
        for i, from_act in enumerate(activities_list):
            for j, to_act in enumerate(activities_list):
                if from_act in transitions and to_act in transitions[from_act]:
                    matrix[i, j] = transitions[from_act][to_act]
        
        # Normalize rows
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        matrix = np.nan_to_num(matrix)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[act.replace('_', ' ').title() for act in activities_list],
            y=[act.replace('_', ' ').title() for act in activities_list],
            colorscale='Viridis',
            text=[[f'{val:.2%}' for val in row] for row in matrix],
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='Activity Transition Probability Matrix',
            xaxis_title='To Activity',
            yaxis_title='From Activity',
            height=700,
            width=800
        )
        
        # Save
        output_path = self.output_dir / "activity_transition_matrix.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved transition matrix to {output_path}")
        # Plotly figure doesn't require plt.close
        return fig
    
    def create_activity_by_demographics(self, activities: List[Dict]) -> plt.Figure:
        """
        Create activity patterns by demographic groups.
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Activity Patterns by Demographics", fontsize=16, fontweight='bold')
        
        activities_df = pd.DataFrame(activities)
        
        if len(activities_df) == 0:
            return fig
        
        # 1. Activities by age group
        ax = axes[0, 0]
        if 'person_age' in activities_df.columns and 'activity_category' in activities_df.columns:
            activities_df['age_group'] = pd.cut(
                activities_df['person_age'],
                bins=[0, 18, 35, 50, 65, 100],
                labels=['<18', '18-34', '35-49', '50-64', '65+']
            )
            
            age_activities = (
                activities_df.groupby(['age_group', 'activity_category'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            age_activities.plot(kind='bar', stacked=True, ax=ax, 
                              color=[self.activity_colors.get(cat, self.activity_colors['other']) 
                                    for cat in age_activities.columns])
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Activity Count")
            ax.set_title("Activities by Age Group", fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.tick_params(axis='x', rotation=0)
        
        # 2. Activities by gender
        ax = axes[0, 1]
        if 'person_sex' in activities_df.columns and 'activity_category' in activities_df.columns:
            activities_df['gender'] = activities_df['person_sex'].map({1: 'Male', 2: 'Female'})
            
            gender_activities = (
                activities_df.groupby(['gender', 'activity_category'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            gender_activities.plot(kind='bar', ax=ax,
                                  color=[self.activity_colors.get(cat, self.activity_colors['other']) 
                                        for cat in gender_activities.columns])
            ax.set_xlabel("Gender")
            ax.set_ylabel("Activity Count")
            ax.set_title("Activities by Gender", fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.tick_params(axis='x', rotation=0)
        
        # 3. Weekend vs Weekday patterns (simulated)
        ax = axes[1, 0]
        if 'day_type' in activities_df.columns and 'activity_category' in activities_df.columns:
            day_activities = activities_df.groupby(['day_type', 'activity_category']).size().unstack(fill_value=0)
            day_activities.plot(kind='bar', ax=ax,
                               color=[self.activity_colors.get(cat, self.activity_colors['other']) 
                                     for cat in day_activities.columns])
            ax.set_xlabel("Day Type")
            ax.set_ylabel("Activity Count")
            ax.set_title("Weekend vs Weekday Activities", fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            # Simulate weekend/weekday split
            ax.text(0.5, 0.5, "Day type analysis not available", 
                   ha='center', va='center', fontsize=12)
        
        # 4. Average activity duration by age
        ax = axes[1, 1]
        if 'person_age' in activities_df.columns and 'duration_minutes' in activities_df.columns:
            activities_df['age_group'] = pd.cut(
                activities_df['person_age'],
                bins=[0, 18, 35, 50, 65, 100],
                labels=['<18', '18-34', '35-49', '50-64', '65+']
            )
            
            avg_duration = activities_df.groupby('age_group')['duration_minutes'].mean()
            ax.bar(avg_duration.index, avg_duration.values, color='teal', edgecolor='black')
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Average Duration (minutes)")
            ax.set_title("Average Activity Duration by Age", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "activity_by_demographics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved demographic patterns to {output_path}")
        plt.close(fig)
        return fig
    
    def create_household_activity_coordination(self, buildings_df: pd.DataFrame,
                                              sample_household: int = 0) -> plt.Figure:
        """
        Create visualization of household activity coordination.
        
        Args:
            buildings_df: Buildings dataframe
            sample_household: Index of household to visualize
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        if sample_household >= len(buildings_df):
            ax.text(0.5, 0.5, "Household index out of range", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        building = buildings_df.iloc[sample_household]
        
        if 'persons' not in building or not isinstance(building['persons'], list):
            ax.text(0.5, 0.5, "No persons in selected household", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Plot each person's activities
        for p_idx, person in enumerate(building['persons']):
            if not isinstance(person, dict) or not person.get('has_activities'):
                continue
            
            if 'activity_sequence' not in person:
                continue
            
            activities = person['activity_sequence']
            
            for activity in activities:
                if not isinstance(activity, dict):
                    continue
                
                # Get start time
                if 'start_minute' in activity:
                    start = activity['start_minute']
                elif 'start_time' in activity:
                    time_parts = activity['start_time'].split(':')
                    start = int(time_parts[0]) * 60 + int(time_parts[1]) if len(time_parts) >= 2 else 0
                else:
                    start = 0
                
                duration = activity.get('duration_minutes', 20)
                activity_type = activity.get('activity_category', 'other')
                
                # Map activity to color
                color = self.activity_colors.get(activity_type, self.activity_colors['other'])
                
                # Draw rectangle for activity
                rect = Rectangle((start, p_idx - 0.4), duration, 0.8,
                               facecolor=color, edgecolor='black', 
                               linewidth=0.5, alpha=0.8)
                ax.add_patch(rect)
        
        # Format axes
        ax.set_xlim(0, 1440)
        ax.set_ylim(-0.5, len(building['persons']) - 0.5)
        
        # Set x-axis to show hours
        hour_ticks = np.arange(0, 1441, 120)
        hour_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels, rotation=45)
        
        # Set y-axis
        ax.set_yticks(range(len(building['persons'])))
        ax.set_yticklabels([f"Person {i+1}\n(Age: {p.get('AGEP', 'N/A')})" 
                           for i, p in enumerate(building['persons'])])
        
        # Labels and title
        ax.set_xlabel("Time of Day", fontsize=12)
        ax.set_ylabel("Household Members", fontsize=12)
        ax.set_title(f"Household Activity Coordination - Household {sample_household+1}", 
                    fontsize=16, fontweight='bold')
        
        # Add legend
        legend_patches = [mpatches.Patch(color=color, label=activity.replace('_', ' ').title()) 
                         for activity, color in self.activity_colors.items()]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        
        # Highlight coordination times (meals, sleep)
        meal_times = [(420, 480), (720, 780), (1080, 1140)]  # Breakfast, lunch, dinner
        for start, end in meal_times:
            ax.axvspan(start, end, alpha=0.1, color='orange', zorder=0)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "household_activity_coordination.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved household coordination to {output_path}")
        plt.close(fig)
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all activity visualizations."""
        # Extract activities
        activities = self.extract_activities_from_persons(buildings_df)
        
        if len(activities) == 0:
            logger.warning("No activities found in buildings data")
            return
        
        # Create visualizations
        self.create_daily_activity_timeline(activities)
        self.create_aggregate_activity_patterns(activities)
        self.create_activity_transition_matrix(activities)
        self.create_activity_by_demographics(activities)
        self.create_household_activity_coordination(buildings_df)
        
        logger.info(f"Created all activity visualizations in {self.output_dir}")