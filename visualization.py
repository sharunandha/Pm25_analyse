"""
Visualization Module
Generates visual outputs for PM2.5 estimation results.
Creates heatmaps, before/after comparisons, and time-series graphs.

Author: PM2.5 Estimation System
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import csv
from typing import Dict, Tuple, List


class PM25Visualizer:
    """
    Creates visualizations for PM2.5 estimation results.
    """
    
    def __init__(self, results_dir: str = 'static/results'):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory to save visualization outputs
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set matplotlib style for better-looking plots
        plt.rcParams['figure.facecolor'] = 'white'
    
    def create_heatmap(self, image_path: str, pm25_value: float, 
                      output_name: str = 'heatmap.png') -> str:
        """
        Create a PM2.5 concentration heatmap overlay on the satellite image.
        
        Args:
            image_path: Path to original satellite image
            pm25_value: Estimated PM2.5 concentration
            output_name: Name for output file
            
        Returns:
            str: Path to saved heatmap image
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (800, 600))
        
        # Create heatmap based on image intensity and PM2.5 value
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert and normalize - darker areas = higher pollution
        heatmap_data = 255 - gray
        heatmap_data = cv2.GaussianBlur(heatmap_data, (21, 21), 0)
        
        # Normalize to PM2.5 scale
        heatmap_data = (heatmap_data / 255.0) * pm25_value
        
        # Create colormap
        # Blue (low) -> Green -> Yellow -> Red (high)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_data / max(pm25_value, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        # Add colorbar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                       gridspec_kw={'width_ratios': [20, 1]})
        
        # Display overlay
        ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'PM2.5 Spatial Distribution (Est. {pm25_value:.1f} µg/m³)', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add colorbar
        norm = plt.Normalize(vmin=0, vmax=pm25_value)
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax2)
        cbar.set_label('PM2.5 (µg/m³)', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_before_after(self, image_path: str, 
                           output_name: str = 'before_after.png') -> str:
        """
        Create before/after pollution visualization.
        Before = original, After = contrast-enhanced (simulating reduced pollution)
        
        Args:
            image_path: Path to original satellite image
            output_name: Name for output file
            
        Returns:
            str: Path to saved comparison image
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (800, 600))
        
        # Create "after" version with enhanced clarity
        # Simulates what the area would look like with less pollution
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Increase saturation slightly
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 30)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Current Conditions\n(With Pollution)', 
                     fontsize=13, fontweight='bold')
        ax1.axis('off')
        
        # After
        ax2.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        ax2.set_title('Simulated Clear Conditions\n(Reduced Pollution)', 
                     fontsize=13, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_timeseries_graph(self, current_pm25: float, 
                               history_file: str = 'data/pm25_history.csv',
                               output_name: str = 'timeseries.png') -> str:
        """
        Create date-wise PM2.5 time series graph.
        
        Args:
            current_pm25: Current PM2.5 estimate to add
            history_file: Path to CSV file with historical data
            output_name: Name for output file
            
        Returns:
            str: Path to saved graph
        """
        # Load or create history
        dates = []
        pm25_values = []
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dates.append(row['date'])
                        pm25_values.append(float(row['pm25']))
            except Exception as e:
                print(f"Error reading history file: {e}")
        
        # Add current measurement
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        dates.append(current_date)
        pm25_values.append(float(current_pm25))
        
        # Keep only last 30 measurements
        if len(dates) > 30:
            dates = dates[-30:]
            pm25_values = pm25_values[-30:]
        
        # Save updated history
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'pm25'])
            writer.writeheader()
            for date, pm25 in zip(dates, pm25_values):
                writer.writerow({'date': date, 'pm25': pm25})
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line
        ax.plot(range(len(pm25_values)), pm25_values, marker='o', linewidth=2, 
               markersize=6, color='#2E86AB', label='PM2.5')
        
        # Color zones based on AQI categories
        ax.axhspan(0, 12, alpha=0.1, color='green', label='Good')
        ax.axhspan(12, 35.4, alpha=0.1, color='yellow', label='Moderate')
        ax.axhspan(35.4, 55.4, alpha=0.1, color='orange', label='Unhealthy (Sensitive)')
        ax.axhspan(55.4, 150, alpha=0.1, color='red', label='Unhealthy')
        ax.axhspan(150, 300, alpha=0.1, color='purple', label='Very Unhealthy')
        
        # Styling
        ax.set_xlabel('Measurement Timeline', fontsize=12, fontweight='bold')
        ax.set_ylabel('PM2.5 Concentration (µg/m³)', fontsize=12, fontweight='bold')
        ax.set_title('PM2.5 Levels Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # X-axis labels - show every 5th date if too many
        if len(dates) > 10:
            step = max(1, len(dates) // 10)
            indices = range(0, len(dates), step)
            ax.set_xticks(indices)
            ax.set_xticklabels([dates[i].split()[0] for i in indices], 
                              rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.split()[0] for d in dates], 
                              rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_feature_chart(self, features: Dict[str, float],
                           output_name: str = 'features.png') -> str:
        """
        Create a chart showing all extracted atmospheric features.
        
        Args:
            features: Dictionary of atmospheric features
            output_name: Name for output file
            
        Returns:
            str: Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        feature_names = [
            'Haze\nScore',
            'Turbidity',
            'Visibility',
            'Contrast',
            'Brightness\n(norm)',
            'Saturation\n(norm)'
        ]
        
        values = [
            features.get('haze_score', 0),
            features.get('turbidity', 0),
            features.get('visibility', 0),
            features.get('contrast', 0),
            (features.get('brightness', 128) / 255) * 100,
            (features.get('saturation', 128) / 255) * 100
        ]
        
        colors = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Create bar chart
        bars = ax.bar(feature_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_title('Atmospheric Feature Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_all_visualizations(self, image_path: str, pm25_value: float, 
                                 features: Dict[str, float]) -> Dict[str, str]:
        """
        Create all visualizations at once.
        
        Args:
            image_path: Path to original satellite image
            pm25_value: Estimated PM2.5 concentration
            features: Atmospheric features
            
        Returns:
            dict: Paths to all generated visualizations
        """
        results = {
            'heatmap': self.create_heatmap(image_path, pm25_value),
            'before_after': self.create_before_after(image_path),
            'timeseries': self.create_timeseries_graph(pm25_value),
            'features': self.create_feature_chart(features)
        }
        
        return results
