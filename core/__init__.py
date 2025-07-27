"""
Core package for Unified Defect Detection System
"""

from .detection import DetectionCore
from .enhanced_detection import (
    analyze_defect_predictions_enhanced,
    extract_enhanced_bounding_boxes
)

__all__ = [
    'DetectionCore', 
    'analyze_defect_predictions_enhanced',
    'extract_enhanced_bounding_boxes'
]