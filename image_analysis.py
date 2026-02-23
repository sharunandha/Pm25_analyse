"""
Image Analysis Module
Extracts atmospheric indicators from satellite images
for PM2.5 estimation without machine learning.

Author: PM2.5 Estimation System
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class ImageAnalyzer:
    """
    Analyzes satellite images to extract atmospheric features
    that correlate with PM2.5 pollution levels.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the analyzer with an image path.
        
        Args:
            image_path: Path to the satellite image
        """
        self.image_path = image_path
        self.image = None
        self.gray_image = None
        self.hsv_image = None
        
    def load_and_preprocess(self) -> bool:
        """
        Load image and prepare it for analysis.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image
            self.image = cv2.imread(self.image_path)
            
            if self.image is None:
                raise ValueError("Failed to load image")
            
            # Resize to standard size for consistent processing
            self.image = cv2.resize(self.image, (800, 600))
            
            # Convert to grayscale
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Convert to HSV for saturation analysis
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def calculate_haze_score(self) -> float:
        """
        Calculate haze score based on image clarity.
        Higher haze = lower edge sharpness = higher PM2.5
        
        Returns:
            float: Haze score (0-100)
        """
        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(self.gray_image, cv2.CV_64F)
        edge_variance = laplacian.var()
        
        # Lower variance = more haze
        # Normalize to 0-100 scale (inverted)
        max_variance = 1000  # Typical max for clear images
        haze_score = max(0, 100 - (edge_variance / max_variance * 100))
        
        return min(100, haze_score)
    
    def calculate_brightness(self) -> float:
        """
        Calculate average brightness.
        Very high or very low brightness can indicate atmospheric conditions.
        
        Returns:
            float: Brightness score (0-255)
        """
        # Calculate mean brightness
        brightness = np.mean(self.gray_image)
        return float(brightness)
    
    def calculate_contrast(self) -> float:
        """
        Calculate image contrast.
        Low contrast often correlates with high pollution.
        
        Returns:
            float: Contrast score (0-100)
        """
        # Standard deviation represents contrast
        contrast_std = np.std(self.gray_image)
        
        # Normalize to 0-100 scale
        # Typical std range: 0-80
        contrast_score = (contrast_std / 80) * 100
        
        return min(100, contrast_score)
    
    def calculate_saturation(self) -> float:
        """
        Calculate average saturation.
        Lower saturation often indicates hazy, polluted conditions.
        
        Returns:
            float: Saturation score (0-255)
        """
        # Extract saturation channel from HSV
        saturation_channel = self.hsv_image[:, :, 1]
        avg_saturation = np.mean(saturation_channel)
        
        return float(avg_saturation)
    
    def calculate_atmospheric_turbidity(self) -> float:
        """
        Calculate atmospheric turbidity using dark channel prior concept.
        
        Returns:
            float: Turbidity score (0-100)
        """
        # Split into BGR channels
        b, g, r = cv2.split(self.image)
        
        # Dark channel: minimum of RGB channels
        dark_channel = cv2.min(cv2.min(r, g), b)
        
        # Higher dark channel values = more atmospheric scattering
        turbidity = np.mean(dark_channel)
        
        # Normalize to 0-100
        turbidity_score = (turbidity / 255) * 100
        
        return float(turbidity_score)
    
    def calculate_visibility_index(self) -> float:
        """
        Calculate visibility index based on histogram distribution.
        
        Returns:
            float: Visibility score (0-100, lower = worse visibility)
        """
        # Calculate histogram
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        
        # Normalize histogram
        hist_norm = hist.ravel() / hist.sum()
        
        # Calculate entropy (higher entropy = better visibility)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Normalize to 0-100 (max entropy ≈ 8 for 8-bit image)
        visibility_score = (entropy / 8) * 100
        
        return min(100, visibility_score)
    
    def analyze(self) -> Dict[str, float]:
        """
        Perform complete image analysis and extract all features.
        
        Returns:
            dict: Dictionary containing all atmospheric indicators
        """
        if not self.load_and_preprocess():
            raise ValueError("Failed to load and preprocess image")
        
        features = {
            'haze_score': self.calculate_haze_score(),
            'brightness': self.calculate_brightness(),
            'contrast': self.calculate_contrast(),
            'saturation': self.calculate_saturation(),
            'turbidity': self.calculate_atmospheric_turbidity(),
            'visibility': self.calculate_visibility_index()
        }
        
        return features
    
    def get_processed_image(self) -> np.ndarray:
        """
        Get the preprocessed image for visualization.
        
        Returns:
            np.ndarray: Processed image
        """
        return self.image


def analyze_image(image_path: str) -> Dict[str, float]:
    """
    Convenience function to analyze an image and return features.
    
    Args:
        image_path: Path to the satellite image
        
    Returns:
        dict: Atmospheric features extracted from the image
    """
    analyzer = ImageAnalyzer(image_path)
    return analyzer.analyze()
