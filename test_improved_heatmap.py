"""
Test Script for Improved PM2.5 Heatmap Detection
Regenerates heatmaps with enhanced pollution detection algorithm
"""

import os
import numpy as np
from visualization import PM25Visualizer
from datetime import datetime


def regenerate_all_heatmaps():
    """Regenerate all heatmaps with improved PM2.5 detection."""
    
    print("=" * 70)
    print("PM2.5 HEATMAP REGENERATION WITH IMPROVED DETECTION")
    print("=" * 70)
    
    visualizer = PM25Visualizer('static/results')
    
    # Check for sample images
    upload_dir = 'static/uploads'
    if not os.path.exists(upload_dir):
        print(f"\nNo uploaded images found in {upload_dir}")
        print("\nGenerating test heatmaps with sample PM2.5 values...")
        
        # If no images, create from datasets
        dataset_dir = 'datasets_images/real/delhi/train/clean'
        test_dir = 'datasets_images/real/delhi/test/clean'
        
        image_found = False
        
        # Try to find images in datasets
        for search_dir in [test_dir, dataset_dir]:
            if os.path.exists(search_dir):
                images = [f for f in os.listdir(search_dir) if f.endswith(('.png', '.jpg', '.tif'))]
                if images:
                    sample_image = os.path.join(search_dir, images[0])
                    image_found = True
                    print(f"\n✓ Found sample image: {sample_image}")
                    
                    # Generate heatmaps with different PM2.5 values
                    pm25_values = [20, 50, 85, 120, 180]
                    
                    for pm25 in pm25_values:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_name = f'heatmap_detection_pm25_{pm25}_{timestamp}.png'
                        
                        print(f"\n  Generating heatmap for PM2.5={pm25} µg/m³...")
                        path = visualizer.create_heatmap(sample_image, pm25, output_name)
                        print(f"  ✓ Saved: {output_name}")
                    
                    break
        
        if not image_found:
            print("⚠ No images found in datasets. Please upload an image first.")
            return
    
    else:
        # Regenerate from uploaded images
        images = [f for f in os.listdir(upload_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        if not images:
            print(f"⚠ No images found in {upload_dir}")
            return
        
        print(f"\n✓ Found {len(images)} uploaded image(s)")
        
        # Regenerate heatmaps for each image
        for img_file in images[:3]:  # Process first 3 images
            image_path = os.path.join(upload_dir, img_file)
            print(f"\n  Processing: {img_file}")
            
            # Create heatmaps with different PM2.5 levels
            pm25_values = [25, 60, 95, 150]
            
            for pm25 in pm25_values:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_name = f'heatmap_{img_file.split(".")[0]}_pm25_{pm25}_{timestamp}.png'
                
                try:
                    path = visualizer.create_heatmap(image_path, pm25, output_name)
                    print(f"    ✓ PM2.5={pm25} µg/m³")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("HEATMAP REGENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files are saved in: static/results/")
    print(f"\nIMPROVEMENTS:")
    print("  ✓ Enhanced PM2.5 detection using multiple air quality indicators")
    print("  ✓ Green-to-Red color gradient (Green=Clean, Red=Polluted)")
    print("  ✓ Brown color detection for typical pollution patterns")
    print("  ✓ Vegetation-based filtering")
    print("  ✓ Morphological noise reduction")
    print("  ✓ Better AQI category classification")


if __name__ == '__main__':
    regenerate_all_heatmaps()
