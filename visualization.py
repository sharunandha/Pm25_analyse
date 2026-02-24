"""
Visualization Module
Generates visual outputs for PM2.5 estimation results.
Creates heatmaps, before/after comparisons, dehazed images, and time-series graphs.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
import os
import csv
from typing import Dict, Tuple, List


class PM25Visualizer:
    """Creates visualizations for PM2.5 estimation results."""

    def __init__(self, results_dir: str = 'static/results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        plt.rcParams.update({
            'figure.facecolor': '#ffffff',
            'axes.facecolor': '#fafbff',
            'axes.edgecolor': '#d1d5db',
            'axes.labelcolor': '#374151',
            'text.color': '#111827',
            'xtick.color': '#6b7280',
            'ytick.color': '#6b7280',
            'grid.color': '#e5e7eb',
            'grid.alpha': 0.6,
            'font.family': 'sans-serif',
            'font.size': 11,
        })

    def create_heatmap(self, image_path: str, pm25_value: float,
                       output_name: str = 'heatmap.png') -> str:
        """
        Create a smooth multi-band PM2.5 heatmap overlay with contour lines.

        Style: low→high bands with clear boundaries (similar to GIS heatmaps)
        instead of binary square blocks.
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LANCZOS4)

        # Convert to useful color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Per-pixel pollution likelihood field
        brightness = hsv[:, :, 2] / 255.0
        saturation = hsv[:, :, 1] / 255.0
        haze_score = brightness * (1.0 - saturation)

        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        veg_index = (green - red) / (green + red + 1e-6)
        veg_score = np.clip((veg_index + 1.0) / 2.0, 0, 1)

        local_edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        local_contrast = cv2.GaussianBlur(local_edges, (0, 0), 2.0)
        local_contrast = local_contrast / (np.max(local_contrast) + 1e-6)

        score = (0.50 * haze_score +
                 0.30 * (1.0 - veg_score) +
                 0.20 * (1.0 - local_contrast))

        # Scale by predicted PM2.5: higher PM2.5 amplifies hazy regions more
        # This keeps clear sky blue but turns hazy/polluted areas red
        pm25_factor = np.clip(pm25_value / 150.0, 0.8, 1.6)
        score = np.clip(score * pm25_factor, 0, 1)

        # Smooth field for organic heatmap boundaries
        score = cv2.GaussianBlur(score, (0, 0), 5.5)
        score = cv2.GaussianBlur(score, (0, 0), 2.0)

        # Plot filled bands + contour lines
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        levels = [0.00, 0.18, 0.32, 0.46, 0.60, 0.74, 1.01]
        fill_colors = ['#f8f9fa', '#0099ff', '#00ffff', '#00ff00', '#ffff00', '#ff6600', '#ff0000']

        ax.contourf(score, levels=levels, colors=fill_colors, alpha=0.65, origin='upper', antialiased=True)
        ax.contour(score,
                   levels=[0.32, 0.46, 0.60, 0.74],
                   colors=['#0066cc', '#00cc00', '#ff9900', '#ff0000'],
                   linewidths=2.0,
                   alpha=0.95,
                   origin='upper')

        ax.set_title(f'PM2.5 Heatmap Overlay (Low-Medium-High)  |  Est: {pm25_value:.1f} ug/m3',
                     fontsize=14, fontweight='bold', pad=14)
        ax.axis('off')

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#0099ff', edgecolor='#000', label='Low PM2.5'),
            Patch(facecolor='#ffff00', edgecolor='#000', label='Moderate PM2.5'),
            Patch(facecolor='#ff0000', edgecolor='#000', label='High PM2.5'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                  framealpha=0.9, facecolor='#ffffff', edgecolor='#d1d5db',
                  labelcolor='#111827')

        plt.tight_layout()
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        return output_path

    def _generate_dehazed_image(self, image: np.ndarray) -> np.ndarray:
        """Generate a smoother dehazed image with reduced artifacts."""
        # Resize with high-quality interpolation to reduce jagged edges
        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LANCZOS4)

        # Convert to LAB and gently enhance luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Denoise luminance first to avoid CLAHE tile artifacts
        l = cv2.bilateralFilter(l, d=7, sigmaColor=35, sigmaSpace=35)

        # Use softer CLAHE settings and larger tile size for smoother regions
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(16, 16))
        l_enhanced = clahe.apply(l)

        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Very mild saturation boost
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) * 1.06
        hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Light denoise + conservative unsharp mask
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        enhanced = cv2.addWeighted(enhanced, 1.08, blur, -0.08, 0)

        return enhanced

    def create_dehazed(self, image_path: str,
                       output_name: str = 'dehazed.png') -> str:
        """Create a standalone dehazed (PM2.5 removal) image."""
        image = cv2.imread(image_path)
        output_path = os.path.join(self.results_dir, output_name)
        enhanced = self._generate_dehazed_image(image)

        # Save directly to avoid matplotlib resampling artifacts
        cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return output_path

    def create_before_after(self, image_path: str,
                            output_name: str = 'before_after.png') -> str:
        """Create before/after pollution comparison."""
        image = cv2.imread(image_path)
        original = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LANCZOS4)
        enhanced = self._generate_dehazed_image(image)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original (With Pollution)', fontsize=13, fontweight='bold', color='#dc2626')
        ax1.axis('off')
        ax2.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        ax2.set_title('After PM2.5 Removal (Dehazed)', fontsize=13, fontweight='bold', color='#16a34a')
        ax2.axis('off')

        # Add border between images
        fig.subplots_adjust(wspace=0.04)
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        return output_path

    def create_timeseries_graph(self, current_pm25: float,
                                history_file: str = 'data/pm25_history.csv',
                                output_name: str = 'timeseries.png') -> str:
        """Create a styled PM2.5 time series line graph."""
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

        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        dates.append(current_date)
        pm25_values.append(float(current_pm25))

        if len(dates) > 30:
            dates = dates[-30:]
            pm25_values = pm25_values[-30:]

        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'pm25'])
            writer.writeheader()
            for date, pm25 in zip(dates, pm25_values):
                writer.writerow({'date': date, 'pm25': pm25})

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(pm25_values))
        # Line with gradient markers
        ax.plot(x, pm25_values, marker='o', linewidth=2.5, markersize=7,
                color='#818cf8', markerfacecolor='#c7d2fe', markeredgecolor='#6366f1',
                markeredgewidth=1.5, label='PM2.5', zorder=5)
        ax.fill_between(x, pm25_values, alpha=0.08, color='#818cf8')

        # AQI zone bands
        zones = [
            (0, 12, '#22c55e', 'Good', 0.08),
            (12, 35.4, '#eab308', 'Moderate', 0.07),
            (35.4, 55.4, '#f97316', 'USG', 0.06),
            (55.4, 150, '#ef4444', 'Unhealthy', 0.05),
            (150, 300, '#a855f7', 'Very Unhealthy', 0.04),
        ]
        for lo, hi, c, lbl, a in zones:
            ax.axhspan(lo, hi, alpha=a, color=c)

        ax.set_xlabel('Measurement Timeline', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('PM2.5 Concentration (ug/m3)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('PM2.5 Levels Over Time', fontsize=14, fontweight='bold', pad=16)
        ax.grid(True, alpha=0.15, linestyle='--')

        if len(dates) > 10:
            step = max(1, len(dates) // 10)
            indices = list(range(0, len(dates), step))
            ax.set_xticks(indices)
            ax.set_xticklabels([dates[i].split()[0] for i in indices], rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.split()[0] for d in dates], rotation=45, ha='right')

        ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        return output_path

    def create_feature_chart(self, features: Dict[str, float],
                             output_name: str = 'features.png') -> str:
        """Create a combined bar + line chart for atmospheric features."""
        fig, ax1 = plt.subplots(figsize=(11, 6))

        feature_names = ['Haze\nScore', 'Turbidity', 'Visibility', 'Contrast', 'Brightness\n(norm)', 'Saturation\n(norm)']
        values = [
            features.get('haze_score', 0),
            features.get('turbidity', 0),
            features.get('visibility', 0),
            features.get('contrast', 0),
            (features.get('brightness', 128) / 255) * 100,
            (features.get('saturation', 128) / 255) * 100
        ]

        bar_colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899']
        x = np.arange(len(feature_names))
        width = 0.55

        # Bar chart
        bars = ax1.bar(x, values, width, color=bar_colors, alpha=0.85, edgecolor='none', zorder=3)

        # Value labels on top
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=10, color='#e5e7eb')

        ax1.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_title('Atmospheric Feature Analysis', fontsize=14, fontweight='bold', pad=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, fontsize=9)
        ax1.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 110)
        ax1.grid(axis='y', alpha=0.12, linestyle='--')

        # Overlay line chart on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, values, marker='D', linewidth=2, markersize=7,
                 color='#facc15', markerfacecolor='#fef08a', markeredgecolor='#eab308',
                 markeredgewidth=1.5, label='Trend', zorder=5)
        ax2.set_ylabel('Trend Line', fontsize=11, color='#facc15', labelpad=10)
        ax2.set_ylim(ax1.get_ylim())
        ax2.tick_params(axis='y', colors='#facc15')

        # Legends
        ax1.legend(['Features (Bar)'], loc='upper left', fontsize=9, framealpha=0.3)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.results_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        return output_path

    def create_all_visualizations(self, image_path: str, pm25_value: float,
                                  features: Dict[str, float]) -> Dict[str, str]:
        """Create all visualizations at once."""
        return {
            'heatmap': self.create_heatmap(image_path, pm25_value),
            'dehazed': self.create_dehazed(image_path),
            'before_after': self.create_before_after(image_path),
            'timeseries': self.create_timeseries_graph(pm25_value),
            'features': self.create_feature_chart(features)
        }
