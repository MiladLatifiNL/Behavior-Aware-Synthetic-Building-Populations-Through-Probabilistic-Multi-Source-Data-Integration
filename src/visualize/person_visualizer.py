"""
Person Demographics and Relationships Visualizations.

Creates detailed visualizations of person characteristics, demographics, and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class PersonVisualizer:
    """Creates visualizations for person demographics and relationships."""
    
    def __init__(self, output_dir: str = "results/visualizations/persons"):
        """Initialize the person visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_persons_from_buildings(self, buildings_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all persons from buildings into a flat dataframe."""
        persons_list = []
        
        for idx, building in buildings_df.iterrows():
            building_id = building.get('building_id', f'Building_{idx}')
            
            if 'persons' in building and isinstance(building['persons'], list):
                for p_idx, person in enumerate(building['persons']):
                    if isinstance(person, dict):
                        person_data = person.copy()
                        person_data['building_id'] = building_id
                        person_data['person_id'] = f"{building_id}_P{p_idx}"
                        persons_list.append(person_data)
        
        return pd.DataFrame(persons_list)
    
    def create_age_pyramid(self, persons_df: pd.DataFrame) -> plt.Figure:
        """
        Create population age pyramid by gender.
        
        Args:
            persons_df: Persons dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'AGEP' not in persons_df.columns or 'SEX' not in persons_df.columns:
            ax.text(0.5, 0.5, "Age or Sex data not available", 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create age bins
        age_bins = list(range(0, 101, 5))
        age_labels = [f"{i}-{i+4}" for i in age_bins[:-1]]
        persons_df['age_group'] = pd.cut(persons_df['AGEP'], bins=age_bins, 
                                        labels=age_labels, right=False)
        
        # Count by age and sex (1=Male, 2=Female in PUMS)
        male_counts = persons_df[persons_df['SEX'] == 1].groupby('age_group', observed=False).size()
        female_counts = persons_df[persons_df['SEX'] == 2].groupby('age_group', observed=False).size()
        
        # Ensure all age groups are present
        male_counts = male_counts.reindex(age_labels, fill_value=0)
        female_counts = female_counts.reindex(age_labels, fill_value=0)
        
        # Create the pyramid
        y_pos = np.arange(len(age_labels))
        
        # Males on the left (negative values)
        ax.barh(y_pos, -male_counts.values, color='steelblue', label='Male')
        # Females on the right (positive values)
        ax.barh(y_pos, female_counts.values, color='coral', label='Female')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(age_labels)
        ax.set_xlabel('Count')
        ax.set_ylabel('Age Group')
        ax.set_title('Population Age Pyramid', fontsize=16, fontweight='bold')
        ax.legend()
        
        # Make x-axis labels absolute values (set ticks explicitly to avoid warnings)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(abs(t))) for t in ticks])
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "age_pyramid.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved age pyramid to {output_path}")
        
        return fig
    
    def create_demographic_distributions(self, persons_df: pd.DataFrame) -> plt.Figure:
        """
        Create comprehensive demographic distributions.
        
        Args:
            persons_df: Persons dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Person Demographics Distribution", fontsize=16, fontweight='bold')
        
        # Age distribution
        ax = axes[0, 0]
        if 'AGEP' in persons_df.columns:
            ax.hist(persons_df['AGEP'].dropna(), bins=20, 
                   color='teal', edgecolor='black', alpha=0.7)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution", fontweight='bold')
            ax.axvline(persons_df['AGEP'].median(), color='red', 
                      linestyle='--', label=f'Median: {persons_df["AGEP"].median():.0f}')
            ax.legend()
        
        # Employment status
        ax = axes[0, 1]
        if 'is_employed' in persons_df.columns:
            emp_status = persons_df['is_employed'].value_counts()
            emp_labels = ['Employed', 'Not Employed'][:len(emp_status)]
            colors = ['green', 'gray'][:len(emp_status)]
            explode = [0.05] * len(emp_status)
            ax.pie(emp_status.values, labels=emp_labels, autopct='%1.1f%%',
                  colors=colors, explode=explode)
            ax.set_title("Employment Status", fontweight='bold')
        
        # Income distribution
        ax = axes[0, 2]
        if 'PINCP' in persons_df.columns:
            income = persons_df[persons_df['PINCP'] > 0]['PINCP']
            if len(income) > 0:
                ax.hist(income, bins=30, color='gold', 
                       edgecolor='black', alpha=0.7)
                ax.set_xlabel("Personal Income ($)")
                ax.set_ylabel("Count")
                ax.set_title("Income Distribution", fontweight='bold')
                ax.axvline(income.median(), color='red', 
                          linestyle='--', label=f'Median: ${income.median():,.0f}')
                ax.legend()
        
        # Education level
        ax = axes[1, 0]
        if 'education_level' in persons_df.columns:
            edu = persons_df['education_level'].value_counts().head(5)
            ax.barh(edu.index, edu.values, color='purple')
            ax.set_xlabel("Count")
            ax.set_title("Education Level", fontweight='bold')
        
        # Work from home
        ax = axes[1, 1]
        if 'works_from_home' in persons_df.columns:
            wfh = persons_df[persons_df['is_employed'] == 1]['works_from_home'].value_counts()
            labels = ['Office/On-site', 'Work from Home']
            if len(wfh) > 0:
                ax.pie(wfh.values, labels=labels[:len(wfh)], autopct='%1.1f%%',
                      colors=['lightblue', 'orange'])
                ax.set_title("Work Location (Employed Only)", fontweight='bold')
        
        # Age vs Income scatter
        ax = axes[1, 2]
        if 'AGEP' in persons_df.columns and 'PINCP' in persons_df.columns:
            mask = (persons_df['PINCP'] > 0) & (persons_df['PINCP'] < 500000)
            ax.scatter(persons_df.loc[mask, 'AGEP'], 
                      persons_df.loc[mask, 'PINCP'],
                      alpha=0.5, s=10, c=persons_df.loc[mask, 'SEX'], cmap='coolwarm')
            ax.set_xlabel("Age")
            ax.set_ylabel("Income ($)")
            ax.set_title("Age vs Income", fontweight='bold')
            ax.set_ylim(0, 200000)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "demographic_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved demographic distributions to {output_path}")
        
        return fig
    
    def create_household_composition(self, buildings_df: pd.DataFrame) -> plt.Figure:
        """
        Create household composition visualization.
        
        Args:
            buildings_df: Buildings dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Household Composition Analysis", fontsize=16, fontweight='bold')
        
        # Household types
        ax = axes[0, 0]
        if 'household_composition' in buildings_df.columns:
            comp = buildings_df['household_composition'].value_counts()
            ax.pie(comp.values, labels=comp.index, autopct='%1.1f%%',
                  colors=plt.cm.Pastel1(np.linspace(0, 1, len(comp))))
            ax.set_title("Household Types", fontweight='bold')
        
        # Number of children
        ax = axes[0, 1]
        if 'num_children' in buildings_df.columns:
            children = buildings_df['num_children'].value_counts().sort_index()
            ax.bar(children.index, children.values, color='lightgreen', edgecolor='black')
            ax.set_xlabel("Number of Children")
            ax.set_ylabel("Count")
            ax.set_title("Children per Household", fontweight='bold')
            ax.set_xticks(children.index)
        
        # Number of seniors
        ax = axes[1, 0]
        if 'num_seniors' in buildings_df.columns:
            seniors = buildings_df['num_seniors'].value_counts().sort_index()
            ax.bar(seniors.index, seniors.values, color='lightcoral', edgecolor='black')
            ax.set_xlabel("Number of Seniors (65+)")
            ax.set_ylabel("Count")
            ax.set_title("Seniors per Household", fontweight='bold')
            ax.set_xticks(seniors.index)
        
        # Multi-generational households
        ax = axes[1, 1]
        if 'multigenerational' in buildings_df.columns:
            multi = buildings_df['multigenerational'].value_counts()
            labels = ['Single Generation', 'Multi-generational'][:len(multi)]
            colors = ['skyblue', 'salmon', 'lightgreen', 'coral'][:len(multi)]
            explode = [0] * len(multi)
            if len(multi) > 1:
                explode[0] = 0.1
            ax.pie(multi.values, labels=labels, autopct='%1.1f%%',
                  colors=colors, explode=explode if len(explode) > 0 else None)
            ax.set_title("Multi-generational Households", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "household_composition.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved household composition to {output_path}")
        
        return fig
    
    def create_relationship_network(self, buildings_df: pd.DataFrame, 
                                  sample_buildings: int = 3) -> go.Figure:
        """
        Create interactive relationship network within households.
        
        Args:
            buildings_df: Buildings dataframe
            sample_buildings: Number of buildings to visualize
            
        Returns:
            Plotly figure
        """
        # Sample buildings
        sample_df = buildings_df.head(sample_buildings)
        
        # Create network data
        nodes = []
        edges = []
        node_id = 0
        
        for b_idx, building in sample_df.iterrows():
            building_id = building.get('building_id', f'B{b_idx}')
            
            # Add building node
            building_node_id = node_id
            nodes.append({
                'id': node_id,
                'label': f"Building {b_idx+1}",
                'type': 'building',
                'size': 30
            })
            node_id += 1
            
            if 'persons' in building and isinstance(building['persons'], list):
                person_node_ids = []
                
                for p_idx, person in enumerate(building['persons']):
                    if not isinstance(person, dict):
                        continue
                    
                    # Add person node
                    age = person.get('AGEP', 'Unknown')
                    sex = 'M' if person.get('SEX') == 1 else 'F'
                    employed = 'Employed' if person.get('is_employed') else 'Not Employed'
                    
                    nodes.append({
                        'id': node_id,
                        'label': f"Person {p_idx+1}\nAge: {age}, {sex}\n{employed}",
                        'type': 'person',
                        'size': 20
                    })
                    
                    # Connect to building
                    edges.append({
                        'source': building_node_id,
                        'target': node_id
                    })
                    
                    person_node_ids.append(node_id)
                    node_id += 1
                
                # Connect persons within household (simplified - could use actual relationships)
                if len(person_node_ids) > 1:
                    for i in range(len(person_node_ids) - 1):
                        edges.append({
                            'source': person_node_ids[i],
                            'target': person_node_ids[i + 1]
                        })
        
        # Create Plotly network graph
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
        
        node_trace = go.Scatter(
            x=[], y=[], mode='markers+text',
            hovermode='closest',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Type',
                    xanchor='left'
                )
            ),
            text=[],
            textposition="top center"
        )
        
        # Simple layout for demonstration
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([n['id'] for n in nodes])
        G.add_edges_from([(e['source'], e['target']) for e in edges])
        pos = nx.spring_layout(G)
        
        # Add positions to traces
        for edge in edges:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        for node in nodes:
            x, y = pos[node['id']]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node['label'],)
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Household Relationship Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=800
                       ))
        
        # Save
        output_path = self.output_dir / "relationship_network.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved relationship network to {output_path}")
        
        return fig
    
    def create_employment_analysis(self, persons_df: pd.DataFrame) -> plt.Figure:
        """
        Create employment analysis visualizations.
        
        Args:
            persons_df: Persons dataframe
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Employment Analysis", fontsize=16, fontweight='bold')
        
        # Employment by age group
        ax = axes[0, 0]
        if 'AGEP' in persons_df.columns and 'is_employed' in persons_df.columns:
            age_groups = pd.cut(persons_df['AGEP'], bins=[0, 25, 35, 45, 55, 65, 100],
                               labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])
            emp_by_age = persons_df.groupby([age_groups, 'is_employed']).size().unstack(fill_value=0)
            emp_by_age.plot(kind='bar', stacked=True, ax=ax, color=['gray', 'green'])
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Count")
            ax.set_title("Employment by Age Group", fontweight='bold')
            ax.legend(['Not Employed', 'Employed'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Income by employment
        ax = axes[0, 1]
        if 'PINCP' in persons_df.columns and 'is_employed' in persons_df.columns:
            employed_income = persons_df[persons_df['is_employed'] == 1]['PINCP'].dropna()
            unemployed_income = persons_df[persons_df['is_employed'] == 0]['PINCP'].dropna()
            
            bp = ax.boxplot([unemployed_income[unemployed_income > 0], 
                            employed_income[employed_income > 0]], 
                           labels=['Not Employed', 'Employed'],
                           patch_artist=True)
            
            for patch, color in zip(bp['boxes'], ['lightgray', 'lightgreen']):
                patch.set_facecolor(color)
            
            ax.set_ylabel("Income ($)")
            ax.set_title("Income by Employment Status", fontweight='bold')
            ax.set_ylim(0, 150000)
        
        # Work hours distribution
        ax = axes[1, 0]
        if 'WKWN' in persons_df.columns:  # Weeks worked
            work_weeks = persons_df[persons_df['WKWN'] > 0]['WKWN']
            if len(work_weeks) > 0:
                ax.hist(work_weeks, bins=20, color='blue', alpha=0.7, edgecolor='black')
                ax.set_xlabel("Weeks Worked per Year")
                ax.set_ylabel("Count")
                ax.set_title("Work Weeks Distribution", fontweight='bold')
                ax.axvline(52, color='red', linestyle='--', label='Full Year (52 weeks)')
                ax.legend()
        
        # Occupation categories (simplified)
        ax = axes[1, 1]
        if 'occupation_category' in persons_df.columns:
            occ = persons_df[persons_df['is_employed'] == 1]['occupation_category'].value_counts().head(10)
            ax.barh(occ.index, occ.values, color='steelblue')
            ax.set_xlabel("Count")
            ax.set_title("Top 10 Occupations", fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "employment_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved employment analysis to {output_path}")
        
        return fig
    
    def create_all_visualizations(self, buildings_df: pd.DataFrame):
        """Create all person visualizations."""
        # Extract persons
        persons_df = self.extract_persons_from_buildings(buildings_df)
        
        if len(persons_df) == 0:
            logger.warning("No persons found in buildings data")
            return
        
        # Create visualizations
        self.create_age_pyramid(persons_df)
        self.create_demographic_distributions(persons_df)
        self.create_household_composition(buildings_df)
        self.create_relationship_network(buildings_df)
        self.create_employment_analysis(persons_df)
        
        logger.info(f"Created all person visualizations in {self.output_dir}")