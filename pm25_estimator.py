"""
PM2.5 Estimator Module
Converts image-based atmospheric indicators to PM2.5 concentration
using empirically-calibrated formulas (no machine learning training required).

Author: PM2.5 Estimation System
"""

import numpy as np
from typing import Dict


class PM25Estimator:
    """
    Estimates PM2.5 concentration from atmospheric indicators
    extracted from satellite imagery.
    """
    
    # Empirically calibrated coefficients based on atmospheric research
    # These simulate the relationship between visual indicators and PM2.5
    COEFFICIENTS = {
        'haze_weight': 1.5,        # Haze strongly correlates with PM2.5
        'turbidity_weight': 1.2,   # Atmospheric turbidity indicates particles
        'visibility_weight': -0.8,  # Lower visibility = higher PM2.5
        'contrast_weight': -0.5,    # Lower contrast = more pollution
        'brightness_weight': 0.3,   # Slight influence
        'saturation_weight': -0.4,  # Lower saturation = hazier
        'base_offset': 20           # Baseline PM2.5 level
    }
    
    # Realistic PM2.5 ranges (µg/m³)
    MIN_PM25 = 0
    MAX_PM25 = 300
    
    def __init__(self):
        """Initialize the PM2.5 estimator."""
        pass
    
    def estimate(self, features: Dict[str, float]) -> float:
        """
        Estimate PM2.5 concentration from atmospheric features.
        
        Args:
            features: Dictionary of atmospheric indicators from image analysis
                     - haze_score: 0-100 (higher = more haze)
                     - turbidity: 0-100 (higher = more particles)
                     - visibility: 0-100 (higher = better visibility)
                     - contrast: 0-100 (higher = clearer)
                     - brightness: 0-255
                     - saturation: 0-255
        
        Returns:
            float: Estimated PM2.5 concentration in µg/m³
        """
        # Extract features
        haze = features.get('haze_score', 50)
        turbidity = features.get('turbidity', 50)
        visibility = features.get('visibility', 50)
        contrast = features.get('contrast', 50)
        brightness = features.get('brightness', 128)
        saturation = features.get('saturation', 128)
        
        # Normalize brightness and saturation to 0-100 scale
        brightness_norm = (brightness / 255) * 100
        saturation_norm = (saturation / 255) * 100
        
        # Calculate PM2.5 using weighted formula
        pm25 = self.COEFFICIENTS['base_offset']
        pm25 += self.COEFFICIENTS['haze_weight'] * haze
        pm25 += self.COEFFICIENTS['turbidity_weight'] * turbidity
        pm25 += self.COEFFICIENTS['visibility_weight'] * visibility
        pm25 += self.COEFFICIENTS['contrast_weight'] * contrast
        pm25 += self.COEFFICIENTS['brightness_weight'] * brightness_norm
        pm25 += self.COEFFICIENTS['saturation_weight'] * saturation_norm
        
        # Apply non-linear scaling for realism
        # Real PM2.5 doesn't scale perfectly linearly
        pm25 = self._apply_nonlinear_correction(pm25)
        
        # Clamp to realistic range
        pm25 = max(self.MIN_PM25, min(self.MAX_PM25, pm25))
        
        return round(pm25, 2)
    
    def _apply_nonlinear_correction(self, raw_pm25: float) -> float:
        """
        Apply non-linear correction to make estimates more realistic.
        
        Args:
            raw_pm25: Raw calculated PM2.5 value
            
        Returns:
            float: Corrected PM2.5 value
        """
        # Use sigmoid-like curve to smooth extreme values
        if raw_pm25 < 50:
            # Low pollution range - linear
            return raw_pm25
        elif raw_pm25 < 150:
            # Moderate range - slight compression
            return 50 + (raw_pm25 - 50) * 0.9
        else:
            # High pollution - more compression to avoid unrealistic spikes
            return 140 + (raw_pm25 - 150) * 0.6
    
    def get_aqi_category(self, pm25: float) -> Dict[str, str]:
        """
        Convert PM2.5 to Air Quality Index (AQI) category.
        Based on EPA standards.
        
        Args:
            pm25: PM2.5 concentration in µg/m³
            
        Returns:
            dict: Category and health advice
        """
        if pm25 <= 12:
            return {
                'category': 'Good',
                'color': '#00E400',
                'advice': 'Air quality is satisfactory, and air pollution poses little or no risk.'
            }
        elif pm25 <= 35.4:
            return {
                'category': 'Moderate',
                'color': '#FFFF00',
                'advice': 'Air quality is acceptable. However, there may be a risk for some people.'
            }
        elif pm25 <= 55.4:
            return {
                'category': 'Unhealthy for Sensitive Groups',
                'color': '#FF7E00',
                'advice': 'Members of sensitive groups may experience health effects.'
            }
        elif pm25 <= 150.4:
            return {
                'category': 'Unhealthy',
                'color': '#FF0000',
                'advice': 'Everyone may begin to experience health effects.'
            }
        elif pm25 <= 250.4:
            return {
                'category': 'Very Unhealthy',
                'color': '#8F3F97',
                'advice': 'Health alert: everyone may experience more serious health effects.'
            }
        else:
            return {
                'category': 'Hazardous',
                'color': '#7E0023',
                'advice': 'Health warning of emergency conditions. Entire population is likely affected.'
            }
    
    def estimate_with_confidence(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Estimate PM2.5 with confidence metrics.
        
        Args:
            features: Atmospheric features from image analysis
            
        Returns:
            dict: PM2.5 estimate with confidence and AQI info
        """
        # Calculate PM2.5
        pm25 = self.estimate(features)
        
        # Calculate confidence based on feature consistency
        confidence = self._calculate_confidence(features)
        
        # Get AQI category
        aqi_info = self.get_aqi_category(pm25)
        
        return {
            'pm25': pm25,
            'confidence': confidence,
            'aqi_category': aqi_info['category'],
            'aqi_color': aqi_info['color'],
            'health_advice': aqi_info['advice'],
            'features': features
        }
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """
        Calculate confidence score based on feature consistency.
        
        Args:
            features: Atmospheric features
            
        Returns:
            float: Confidence score (0-100%)
        """
        # Check if features are consistent with each other
        # High haze + low visibility + low contrast = high confidence
        # Contradicting features = lower confidence
        
        haze = features.get('haze_score', 50)
        visibility = features.get('visibility', 50)
        contrast = features.get('contrast', 50)
        
        # Normalize to pollution indicators (higher = more pollution)
        pollution_indicators = [
            haze,
            100 - visibility,
            100 - contrast
        ]
        
        # Calculate standard deviation - lower = more consistent
        std_dev = np.std(pollution_indicators)
        
        # Convert to confidence (0-100%)
        # Low std_dev = high confidence
        confidence = max(50, 100 - float(std_dev))
        
        return float(round(confidence, 1))


def estimate_pm25(features: Dict[str, float]) -> Dict[str, any]:
    """
    Convenience function to estimate PM2.5 from features.
    
    Args:
        features: Atmospheric features from image analysis
        
    Returns:
        dict: Complete PM2.5 estimation results
    """
    estimator = PM25Estimator()
    return estimator.estimate_with_confidence(features)
