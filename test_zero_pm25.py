#!/usr/bin/env python3
"""Test heatmap generation with PM2.5=0.0 (good air quality)"""

import os
import glob
from visualization import PM25Visualizer

# Find first available image
images = glob.glob('datasets_images/real/delhi/test/clean/*.jpg') + \
         glob.glob('datasets_images/real/delhi/test/clean/*.jpeg')

if not images:
    print("No test images found. Checking alternate paths...")
    images = glob.glob('datasets_images/**/*.jpg', recursive=True) + \
             glob.glob('datasets_images/**/*.jpeg', recursive=True)

if images:
    test_image = images[0]
    print(f"Testing with image: {test_image}")
    
    visualizer = PM25Visualizer(results_dir='static/results')
    
    # Test with PM2.5 = 0.0 (GOOD air quality - should show NO red areas)
    output = visualizer.create_heatmap(test_image, pm25_value=0.0, 
                                      output_name='test_pm25_zero_good_air.png')
    
    print("\n✓ Heatmap generated for PM2.5=0.0 µg/m³")
    print("  Expected: Clean satellite image with NO red hotspots")
    print(f"  Output: {output}")
    
else:
    print("ERROR: No test images found!")
