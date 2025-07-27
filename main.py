# main.py - SIMPLE FIXED VERSION - API Server Compatible
"""
Main Unified Defect Detector class - SIMPLE & CLEAN
Uses direct import approach (same as working test.py)
Compatible with your custom config.py AND API server
"""

import os
import torch

# Import your custom config
from config import *

# Import model loader
try:
    from models.model_loader import ModelLoader
except ImportError:
    from model_loader import ModelLoader

# Try to import other components (optional - graceful fallback for API server)
try:
    from core.detection import DetectionCore
    from processors.image_processor import ImageProcessor
    from processors.video_processor import VideoProcessor
    PROCESSORS_AVAILABLE = True
    print(" Advanced processors available")
except ImportError:
    print(" Advanced processors not available - using simple detection mode")
    PROCESSORS_AVAILABLE = False


class UnifiedDefectDetector:
    """
    Unified Defect Detection System - SIMPLE FIXED VERSION
    Compatible with API server imports
    """
    
    def __init__(self, anomalib_model_path=None, hrnet_model_path=None, 
                 anomalib_path=None, hrnet_path=None, device=None, auto_load=True):
        """
        Initialize the detector - API server compatible
        
        Args:
            anomalib_model_path: Path to Anomalib model (standard parameter)
            hrnet_model_path: Path to HRNet model (standard parameter)
            anomalib_path: Path to Anomalib model (API server compatibility)
            hrnet_path: Path to HRNet model (API server compatibility)
            device: Device to use
            auto_load: Auto load models
        """
        # Handle parameter compatibility - API server uses different names
        final_anomalib_path = anomalib_path or anomalib_model_path
        final_hrnet_path = hrnet_path or hrnet_model_path
        
        # Device setup
        self.device = device if device else DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Initialize model loader - FIXED VERSION
        self.model_loader = ModelLoader(device=self.device)
        self.detection_core = None
        self.image_processor = None
        self.video_processor = None
        
        print(f"Initializing SIMPLE Unified Defect Detection System on {self.device}")
        
        # Load models
        if auto_load:
            success = self.load_models(final_anomalib_path, final_hrnet_path)
            if success:
                self._init_processors()
                print(" System ready!")
            else:
                print(" Model loading failed")
    
    def load_models(self, anomalib_path=None, hrnet_path=None):
        """Load models"""
        try:
            return self.model_loader.load_models(anomalib_path, hrnet_path)
        except Exception as e:
            print(f" Error loading models: {e}")
            return False
    
    def _init_processors(self):
        """Initialize processors if available"""
        if PROCESSORS_AVAILABLE:
            try:
                anomalib_model, hrnet_model = self.model_loader.get_models()
                self.detection_core = DetectionCore(anomalib_model, hrnet_model, self.device)
                self.image_processor = ImageProcessor(self.detection_core)
                self.video_processor = VideoProcessor(self.detection_core)
                print(" Advanced processors initialized")
            except Exception as e:
                print(f" Processor initialization failed: {e}")
    
    def is_ready(self):
        """Check if system is ready"""
        return self.model_loader.is_ready()
    
    def detect_anomaly(self, image_path):
        """
        Simple anomaly detection - always available
        API server compatible method
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded")
        
        print(f"Detecting anomaly in: {image_path}")
        
        # Get the anomalib model directly
        anomalib_model, _ = self.model_loader.get_models()
        
        # Run prediction
        result = anomalib_model.predict(image=image_path)
        
        # Process result
        if hasattr(result, 'pred_score'):
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
        else:
            anomaly_score = 0.0
        
        if hasattr(result, 'pred_label'):
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
        else:
            is_anomalous = anomaly_score > ANOMALY_THRESHOLD
        
        # Return simple result compatible with API server
        return {
            'image_path': image_path,
            'anomaly_score': anomaly_score,
            'is_anomalous': is_anomalous,
            'decision': 'ANOMALY' if is_anomalous else 'NORMAL',
            'threshold_used': ANOMALY_THRESHOLD,
            'final_decision': 'DEFECT' if is_anomalous else 'GOOD',  # API server compatibility
            'processing_time': 0.1,  # Simple fallback
            'timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else "2024-01-01T00:00:00",
            'anomaly_detection': {
                'anomaly_score': anomaly_score,
                'decision': 'DEFECT' if is_anomalous else 'GOOD',
                'threshold_used': ANOMALY_THRESHOLD
            },
            'detected_defect_types': ['anomaly'] if is_anomalous else []
        }
    
    def process_image(self, image_path, output_dir=None):
        """
        Process image - API server compatible
        Uses advanced processing if available, otherwise simple detection
        """
        import time
        start_time = time.time()
        
        if self.image_processor:
            # Use advanced processing
            return self.image_processor.process_single_image(image_path, output_dir)
        else:
            # Use simple detection with enhanced format for API compatibility
            result = self.detect_anomaly(image_path)
            
            # Enhance result for API server compatibility
            processing_time = time.time() - start_time
            result.update({
                'processing_time': processing_time,
                'timestamp': self._get_timestamp(),
                'status': 'completed',
                'visualization_path': None,
                'report_path': None
            })
            
            return result
    
    def _get_timestamp(self):
        """Get current timestamp - fallback for different environments"""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except:
            return "2024-01-01T00:00:00"
    
    def process_batch(self, input_folder, output_folder=None):
        """Process batch of images - API server compatible"""
        if self.image_processor:
            return self.image_processor.process_batch_images(input_folder, output_folder)
        else:
            raise RuntimeError("Batch processing requires advanced processors")
    
    def process_video(self, video_path, output_dir=None, save_video=True, frame_skip=None):
        """Process video - API server compatible"""
        if self.video_processor:
            return self.video_processor.process_video(video_path, output_dir, save_video, frame_skip)
        else:
            raise RuntimeError("Video processing requires advanced processors")
    
    def get_system_info(self):
        """Get system information - API server compatible"""
        return {
            'device': self.device,
            'models_loaded': self.is_ready(),
            'system_ready': self.is_ready(),
            'advanced_processors': PROCESSORS_AVAILABLE and self.detection_core is not None,
            'anomaly_threshold': ANOMALY_THRESHOLD,
            'defect_threshold': DEFECT_CONFIDENCE_THRESHOLD,
            'supported_classes': SPECIFIC_DEFECT_CLASSES,
            'defect_classes': len(SPECIFIC_DEFECT_CLASSES),
            'version': 'SIMPLE_FIXED_API_COMPATIBLE',
            'detector_ready': self.is_ready(),
            'supported_formats': ['JPG', 'PNG', 'BMP'],
            'max_file_size': '5MB'
        }


# Convenience functions - API server compatible
def create_detector(anomalib_path=None, hrnet_path=None, device=None):
    """
    Quick detector creation - API server compatible
    This is what the API server imports as create_detector
    """
    print("Creating SIMPLE detector (API server compatible)...")
    return UnifiedDefectDetector(
        anomalib_path=anomalib_path,
        hrnet_path=hrnet_path, 
        device=device,
        auto_load=True
    )

def quick_detect(image_path, anomalib_path=None, hrnet_path=None):
    """Quick single image detection - API server compatible"""
    detector = create_detector(anomalib_path, hrnet_path)
    return detector.detect_anomaly(image_path)


# Simple test function
def test_system():
    """Simple system test - API server compatible"""
    print(" Testing SIMPLE System (API Server Compatible)...")
    print("=" * 50)
    
    try:
        # Test dengan parameter yang benar untuk API server compatibility
        detector = UnifiedDefectDetector(auto_load=True)
        
        if detector.is_ready():
            print(" System ready!")
            
            system_info = detector.get_system_info()
            print(f"   Device: {system_info['device']}")
            print(f"   Models: {'' if system_info['models_loaded'] else ''}")
            print(f"   Advanced: {'' if system_info['advanced_processors'] else ''}")
            print(f"   API Compatible: ")
            
            # Test the method that API server uses
            print("\n Testing API server compatibility...")
            if hasattr(detector, 'process_image') and callable(detector.process_image):
                print("    process_image method available")
            if hasattr(detector, 'get_system_info') and callable(detector.get_system_info):
                print("    get_system_info method available")
            if hasattr(detector, 'is_ready') and callable(detector.is_ready):
                print("    is_ready method available")
            
            # Test API server create_detector function
            print("\n Testing create_detector function (API server uses this)...")
            api_detector = create_detector()
            if api_detector and api_detector.is_ready():
                print("    create_detector() works correctly")
            else:
                print("    create_detector() created detector but not ready")
            
            return True
        else:
            print(" System not ready")
            return False
            
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# This is important for API server imports
def auto_load_models(device='cuda'):
    """Auto load models - API server compatibility"""
    detector = create_detector(device=device)
    if detector.is_ready():
        return detector.model_loader.get_models()
    else:
        raise RuntimeError("Failed to load models")


if __name__ == "__main__":
    print("Unified Defect Detection System - SIMPLE FIXED VERSION (API Compatible)")
    print("=" * 70)
    
    if test_system():
        print("\n System working!")
        print("\nAPI Server Compatible Methods:")
        print("   detector = create_detector()")
        print("   result = detector.process_image(image_path)")
        print("   info = detector.get_system_info()")
        print("   ready = detector.is_ready()")
        print("\nConvenience Functions:")
        print("   result = quick_detect(image_path)")
        print("   models = auto_load_models()")
        print("\n Ready for API server integration!")
    else:
        print("\n System test failed. Check model paths in config.py")
        print(" Make sure your model files exist and are accessible")
    
    print("\n This main.py is compatible with your API server!")
    print("   API server can now import: from main import UnifiedDefectDetector, create_detector")


# Convenience functions
def create_detector(anomalib_path=None, hrnet_path=None, device=None):
    """Quick detector creation"""
    return UnifiedDefectDetector(
        anomalib_path=anomalib_path,
        hrnet_path=hrnet_path, 
        device=device,
        auto_load=True
    )

def quick_detect(image_path, anomalib_path=None, hrnet_path=None):
    """Quick single image detection"""
    detector = create_detector(anomalib_path, hrnet_path)
    return detector.detect_anomaly(image_path)


# Simple test function
def test_system():
    """Simple system test"""
    print(" Testing SIMPLE System...")
    print("=" * 40)
    
    try:
        detector = create_detector()
        
        if detector.is_ready():
            print(" System ready!")
            
            system_info = detector.get_system_info()
            print(f"   Device: {system_info['device']}")
            print(f"   Models: {'' if system_info['models_loaded'] else ''}")
            print(f"   Advanced: {'' if system_info['advanced_processors'] else ''}")
            
            return True
        else:
            print(" System not ready")
            return False
            
    except Exception as e:
        print(f" Test failed: {e}")
        return False


if __name__ == "__main__":
    print("Unified Defect Detection System - SIMPLE FIXED VERSION")
    print("=" * 60)
    
    if test_system():
        print("\n System working!")
        print("\nAvailable methods:")
        print("   detector = create_detector()")
        print("   result = detector.detect_anomaly(image_path)")
        print("   result = quick_detect(image_path)")
        print("\n Ready for use!")
    else:
        print("\n System test failed. Check model paths in config.py")