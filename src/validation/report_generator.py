"""
Report generator for PUMS Enrichment Pipeline validation.

This module creates HTML validation reports with visualizations and metrics
for each phase of the pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
from jinja2 import Template
import logging
import base64
from io import BytesIO

from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)

# HTML template for validation reports
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .summary {
            background-color: {% if validation_results.valid %}#d4edda{% else %}#f8d7da{% endif %};
            border: 1px solid {% if validation_results.valid %}#c3e6cb{% else %}#f5c6cb{% endif %};
            color: {% if validation_results.valid %}#155724{% else %}#721c24{% endif %};
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
        }
        .warning {
            color: #ffc107;
        }
        .success {
            color: #28a745;
        }
        .plot {
            margin: 20px 0;
            text-align: center;
        }
        .plot img {
            max-width: 100%;
            height: auto;
        }
        .timestamp {
            color: #6c757d;
            font-size: 14px;
        }
        .checks {
            margin: 20px 0;
        }
        .check-item {
            padding: 5px 0;
        }
        .check-passed {
            color: #28a745;
        }
        .check-failed {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p class="timestamp">Generated: {{ timestamp }}</p>
        
        <div class="summary">
            <h2>Validation Summary</h2>
            <p><strong>Status:</strong> {{ validation_results.summary }}</p>
            <p><strong>Errors:</strong> {{ validation_results.errors|length }}</p>
            <p><strong>Warnings:</strong> {{ validation_results.warnings|length }}</p>
        </div>
        
        <h2>Key Metrics</h2>
        <div class="metrics">
            {% for key, value in validation_results.metrics.items() %}
            <div class="metric-card">
                <div class="metric-value">
                    {% if value is number %}
                        {% if value < 1 %}
                            {{ "{:.2f}".format(value) }}
                        {% elif value > 1000000 %}
                            {{ "{:,.0f}".format(value) }}
                        {% else %}
                            {{ "{:,.0f}".format(value) }}
                        {% endif %}
                    {% else %}
                        {{ value }}
                    {% endif %}
                </div>
                <div class="metric-label">{{ key|replace('_', ' ')|title }}</div>
            </div>
            {% endfor %}
        </div>
        
        <h2>Validation Checks</h2>
        <div class="checks">
            <h3>Passed Checks ({{ validation_results.checks_passed|length }})</h3>
            {% for check in validation_results.checks_passed %}
            <div class="check-item check-passed"> {{ check|replace('_', ' ')|title }}</div>
            {% endfor %}
            
            {% if validation_results.checks_failed %}
            <h3>Failed Checks ({{ validation_results.checks_failed|length }})</h3>
            {% for check in validation_results.checks_failed %}
            <div class="check-item check-failed"> {{ check|replace('_', ' ')|title }}</div>
            {% endfor %}
            {% endif %}
        </div>
        
        {% if validation_results.errors %}
        <h2>Errors</h2>
        <ul class="error">
            {% for error in validation_results.errors %}
            <li>{{ error }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if validation_results.warnings %}
        <h2>Warnings</h2>
        <ul class="warning">
            {% for warning in validation_results.warnings %}
            <li>{{ warning }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if plots %}
        <h2>Data Visualizations</h2>
        {% for plot_title, plot_data in plots.items() %}
        <div class="plot">
            <h3>{{ plot_title }}</h3>
            <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_title }}">
        </div>
        {% endfor %}
        {% endif %}
        
        <h2>Processing Metadata</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            {% for key, value in metadata.items() %}
            {% if key not in ['processing_steps', 'validation_results', 'config'] %}
            <tr>
                <td>{{ key|replace('_', ' ')|title }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endif %}
            {% endfor %}
        </table>
        
        {% if metadata.processing_steps %}
        <h2>Processing Steps</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Records</th>
                <th>Timestamp</th>
            </tr>
            {% for step in metadata.processing_steps %}
            <tr>
                <td>{{ step.step|replace('_', ' ')|title }}</td>
                <td>{{ step.get('records', step.get('features_created', 'N/A')) }}</td>
                <td>{{ step.timestamp }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
</body>
</html>
"""


def create_plot_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return plot_data


def generate_phase1_plots(buildings: pd.DataFrame) -> Dict[str, str]:
    """Generate plots for Phase 1 validation report."""
    plots = {}
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Household size distribution
    if 'household_size_cat' in buildings.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        buildings['household_size_cat'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title('Household Size Distribution')
        ax.set_xlabel('Household Size Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plots['Household Size Distribution'] = create_plot_base64(fig)
    
    # Plot 2: Income distribution
    if 'income_quintile' in buildings.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        income_order = ['q1_lowest', 'q2_low', 'q3_medium', 'q4_high', 'q5_highest']
        income_counts = buildings['income_quintile'].value_counts()
        income_counts = income_counts.reindex(income_order, fill_value=0)
        income_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Income Quintile Distribution')
        ax.set_xlabel('Income Quintile')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plots['Income Distribution'] = create_plot_base64(fig)
    
    # Plot 3: Building type distribution
    if 'building_type_simple' in buildings.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        building_types = buildings['building_type_simple'].value_counts()
        building_types.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Building Type Distribution')
        ax.set_ylabel('')
        plots['Building Type Distribution'] = create_plot_base64(fig)
    
    # Plot 4: Person count vs household size
    if all(col in buildings.columns for col in ['NP', 'actual_person_count']):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(buildings['NP'], buildings['actual_person_count'], alpha=0.5)
        ax.plot([0, buildings['NP'].max()], [0, buildings['NP'].max()], 'r--', label='Perfect match')
        ax.set_xlabel('Expected Person Count (NP)')
        ax.set_ylabel('Actual Person Count')
        ax.set_title('Person Count Completeness')
        ax.legend()
        plots['Person Count Completeness'] = create_plot_base64(fig)
    
    # Plot 5: Geographic distribution
    if 'STATE' in buildings.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        state_counts = buildings['STATE'].value_counts().head(20)
        state_counts.plot(kind='bar', ax=ax)
        ax.set_title('Top 20 States by Building Count')
        ax.set_xlabel('State')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plots['Geographic Distribution'] = create_plot_base64(fig)
    
    # Plot 6: Energy intensity distribution
    if 'energy_intensity_cat' in buildings.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        energy_order = ['very_low', 'low', 'moderate', 'high', 'very_high']
        energy_counts = buildings['energy_intensity_cat'].value_counts()
        energy_counts = energy_counts.reindex(energy_order, fill_value=0)
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        energy_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Energy Intensity Distribution')
        ax.set_xlabel('Energy Intensity Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plots['Energy Intensity Distribution'] = create_plot_base64(fig)
    
    return plots


def generate_phase1_report(buildings: pd.DataFrame, metadata: Dict[str, Any], 
                         validation_results: Dict[str, Any]) -> Path:
    """
    Generate HTML validation report for Phase 1.
    
    Args:
        buildings: Building DataFrame
        metadata: Processing metadata
        validation_results: Validation results
        
    Returns:
        Path to generated report
    """
    logger.info("Generating Phase 1 validation report")
    
    config = get_config()
    report_path = Path(config.get_data_path('phase1_validation'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plots = generate_phase1_plots(buildings)
    
    # Create report data
    report_data = {
        'title': 'Phase 1 Validation Report - PUMS Household-Person Integration',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_results': validation_results,
        'metadata': metadata,
        'plots': plots
    }
    
    # Render template
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**report_data)
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Validation report saved to: {report_path}")
    
    return report_path


def generate_phase2_report(buildings: pd.DataFrame, metadata: Dict[str, Any],
                         validation_results: Dict[str, Any]) -> Path:
    """Generate HTML validation report for Phase 2."""
    logger.info("Generating Phase 2 validation report")
    
    config = get_config()
    report_path = Path(config.get_data_path('phase2_validation'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Phase 2 specific plots would include match weight distributions, etc.
    plots = {}  # TODO: Implement Phase 2 specific plots
    
    report_data = {
        'title': 'Phase 2 Validation Report - RECS Building Characteristics Matching',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_results': validation_results,
        'metadata': metadata,
        'plots': plots
    }
    
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**report_data)
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path


def generate_phase3_report(buildings: pd.DataFrame, metadata: Dict[str, Any],
                         validation_results: Dict[str, Any]) -> Path:
    """Generate HTML validation report for Phase 3."""
    logger.info("Generating Phase 3 validation report")
    
    config = get_config()
    report_path = Path(config.get_data_path('phase3_validation'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Phase 3 specific plots would include activity patterns, etc.
    plots = {}  # TODO: Implement Phase 3 specific plots
    
    report_data = {
        'title': 'Phase 3 Validation Report - ATUS Activity Pattern Assignment',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_results': validation_results,
        'metadata': metadata,
        'plots': plots
    }
    
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**report_data)
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path


def generate_phase4_report(buildings: pd.DataFrame, metadata: Dict[str, Any],
                         validation_results: Dict[str, Any]) -> Path:
    """Generate HTML validation report for Phase 4 with weather integration metrics."""
    logger.info("Generating Phase 4 validation report")
    
    config = get_config()
    report_path = Path(config.get_data_path('phase4_validation'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate Phase 4 specific plots
    plots = {}
    
    try:
        # 1. Weather Coverage by State
        fig, ax = plt.subplots(figsize=(12, 6))
        state_coverage = {}
        for idx, building in buildings.iterrows():
            state = building.get('STATE', 'Unknown')
            has_weather = building.get('has_weather', False)
            if state not in state_coverage:
                state_coverage[state] = {'total': 0, 'with_weather': 0}
            state_coverage[state]['total'] += 1
            if has_weather:
                state_coverage[state]['with_weather'] += 1
        
        states = list(state_coverage.keys())
        coverage_rates = [state_coverage[s]['with_weather']/max(state_coverage[s]['total'], 1) * 100 
                         for s in states]
        
        ax.bar(states, coverage_rates)
        ax.set_xlabel('State')
        ax.set_ylabel('Weather Coverage (%)')
        ax.set_title('Weather Data Coverage by State')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plots['weather_coverage'] = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # 2. Temperature Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        temperatures = []
        hdds = []
        cdds = []
        
        for idx, building in buildings.head(min(100, len(buildings))).iterrows():
            if 'weather_summary' in building and building['weather_summary']:
                summary = building['weather_summary']
                if 'temp_mean' in summary:
                    temperatures.append(summary['temp_mean'])
                if 'HDD' in summary:
                    hdds.append(summary['HDD'])
                if 'CDD' in summary:
                    cdds.append(summary['CDD'])
        
        if temperatures:
            ax1.hist(temperatures, bins=20, edgecolor='black')
            ax1.set_xlabel('Mean Temperature (°C)')
            ax1.set_ylabel('Count')
            ax1.set_title('Temperature Distribution')
            ax1.axvline(np.mean(temperatures), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(temperatures):.1f}°C')
            ax1.legend()
        
        if hdds and cdds:
            ax2.scatter(hdds, cdds, alpha=0.6)
            ax2.set_xlabel('Heating Degree Days')
            ax2.set_ylabel('Cooling Degree Days')
            ax2.set_title('HDD vs CDD Distribution')
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plots['temperature_distribution'] = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # 3. Activity-Weather Alignment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        total_activities = 0
        aligned_activities = 0
        outdoor_minutes_dist = []
        
        for idx, building in buildings.head(min(100, len(buildings))).iterrows():
            if 'persons' in building and isinstance(building['persons'], list):
                for person in building['persons']:
                    if isinstance(person, dict):
                        if 'activity_sequence' in person:
                            for activity in person['activity_sequence']:
                                total_activities += 1
                                if 'weather' in activity:
                                    aligned_activities += 1
                        
                        if 'weather_exposure' in person:
                            outdoor_min = person['weather_exposure'].get('outdoor_minutes', 0)
                            outdoor_minutes_dist.append(outdoor_min)
        
        # Alignment pie chart
        if total_activities > 0:
            alignment_rate = aligned_activities / total_activities * 100
            ax1.pie([aligned_activities, total_activities - aligned_activities], 
                   labels=[f'Aligned ({alignment_rate:.1f}%)', 
                          f'Not Aligned ({100-alignment_rate:.1f}%)'],
                   autopct='%1.0f%%', startangle=90)
            ax1.set_title(f'Activity-Weather Alignment\n(Total: {total_activities} activities)')
        
        # Outdoor exposure distribution
        if outdoor_minutes_dist:
            ax2.hist(outdoor_minutes_dist, bins=20, edgecolor='black')
            ax2.set_xlabel('Outdoor Minutes per Person')
            ax2.set_ylabel('Count')
            ax2.set_title('Outdoor Exposure Distribution')
            ax2.axvline(np.mean(outdoor_minutes_dist), color='red', linestyle='--',
                       label=f'Mean: {np.mean(outdoor_minutes_dist):.0f} min')
            ax2.legend()
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plots['activity_alignment'] = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # 4. Solar Radiation and Energy Metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        solar_totals = []
        energy_metrics = {'has_heating': 0, 'has_cooling': 0, 'total': 0}
        
        for idx, building in buildings.head(min(100, len(buildings))).iterrows():
            if 'weather_summary' in building and building['weather_summary']:
                summary = building['weather_summary']
                if 'solar_ghi_total' in summary:
                    solar_totals.append(summary['solar_ghi_total'])
                
                energy_metrics['total'] += 1
                if 'HDD' in summary and summary['HDD'] > 0:
                    energy_metrics['has_heating'] += 1
                if 'CDD' in summary and summary['CDD'] > 0:
                    energy_metrics['has_cooling'] += 1
        
        if solar_totals:
            ax1.hist(solar_totals, bins=20, edgecolor='black')
            ax1.set_xlabel('Total Solar Radiation (Wh/m²)')
            ax1.set_ylabel('Count')
            ax1.set_title('Solar Radiation Distribution')
            ax1.axvline(np.mean(solar_totals), color='red', linestyle='--',
                       label=f'Mean: {np.mean(solar_totals):.0f} Wh/m²')
            ax1.legend()
        
        # Energy demand indicators
        if energy_metrics['total'] > 0:
            categories = ['Heating\nDemand', 'Cooling\nDemand', 'Neither']
            values = [
                energy_metrics['has_heating'],
                energy_metrics['has_cooling'],
                energy_metrics['total'] - max(energy_metrics['has_heating'], 
                                              energy_metrics['has_cooling'])
            ]
            ax2.bar(categories, values)
            ax2.set_ylabel('Number of Buildings')
            ax2.set_title('Energy Demand Indicators')
            
            for i, v in enumerate(values):
                ax2.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plots['energy_metrics'] = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
    except Exception as e:
        logger.warning(f"Error generating Phase 4 plots: {e}")
    
    # Calculate additional Phase 4 statistics
    phase4_stats = {
        'total_buildings': len(buildings),
        'buildings_with_weather': sum(1 for _, b in buildings.iterrows() if b.get('has_weather', False)),
        'weather_coverage_rate': 0,
        'avg_outdoor_exposure': 0,
        'activity_alignment_rate': 0,
        'states_processed': set(),
        'temperature_range': 'N/A',
        'solar_radiation_range': 'N/A'
    }
    
    if phase4_stats['total_buildings'] > 0:
        phase4_stats['weather_coverage_rate'] = (phase4_stats['buildings_with_weather'] / 
                                                 phase4_stats['total_buildings'] * 100)
    
    # Extract states processed
    for idx, building in buildings.iterrows():
        if 'STATE' in building:
            phase4_stats['states_processed'].add(building['STATE'])
    
    phase4_stats['states_processed'] = list(phase4_stats['states_processed'])
    
    # Add Phase 4 specific stats to validation results
    if 'metrics' not in validation_results:
        validation_results['metrics'] = {}
    validation_results['metrics'].update(phase4_stats)
    
    report_data = {
        'title': 'Phase 4 Validation Report - Weather Data Integration',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_results': validation_results,
        'metadata': metadata,
        'plots': plots,
        'phase4_stats': phase4_stats
    }
    
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**report_data)
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Phase 4 validation report saved to {report_path}")
    
    return report_path