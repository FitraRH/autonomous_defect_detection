# main.py
"""
Main Unified Defect Detector class - Entry point for the system
"""

import os
import torch
from models.model_loader import ModelLoader, auto_load_models, load_custom_models
from core.detection import DetectionCore
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from processors.realtime_processor import RealTimeProcessor
from config import *


class UnifiedDefectDetector:
    """
    Unified Defect Detection System
    
    Combines Anomalib (anomaly detection) and HRNet (defect classification)
    for comprehensive product quality inspection.
    """
    
    def __init__(self, anomalib_model_path=None, hrnet_model_path=None, 
                 device=None, auto_load=True):
        """
        Initialize the unified defect detector
        
        Args:
            anomalib_model_path: Custom path to Anomalib model (optional)
            hrnet_model_path: Custom path to HRNet model (optional)
            device: Device to run inference on ('cuda' or 'cpu')
            auto_load: Automatically load models from config paths
        """
        # Device setup
        self.device = device if device else DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Initialize model loader
        self.model_loader = ModelLoader(device=self.device)
        self.detection_core = None
        self.image_processor = None
        self.video_processor = None
        self.realtime_processor = None
        
        print(f"Initializing Unified Defect Detection System on {self.device}")
        
        # Load models
        if auto_load:
            if anomalib_model_path or hrnet_model_path:
                # Load custom models
                success = self.load_custom_models(anomalib_model_path, hrnet_model_path)
            else:
                # Auto-load models from config
                success = self.load_models()
            
            if success:
                self._initialize_processors()
                print("Unified Defect Detection System ready!")
            else:
                print("System initialized but models not loaded. Call load_models() manually.")
        else:
            print("System initialized without models. Call load_models() to proceed.")
    
    def load_models(self):
        """Load models using config paths"""
        success = self.model_loader.load_models()
        if success:
            self._initialize_processors()
        return success
    
    def load_custom_models(self, anomalib_path=None, hrnet_path=None):
        """Load models from custom paths"""
        success = self.model_loader.load_models(anomalib_path, hrnet_path)
        if success:
            self._initialize_processors()
        return success
    
    def _initialize_processors(self):
        """Initialize processing components including real-time processor"""
        anomalib_model, hrnet_model = self.model_loader.get_models()
        
        if anomalib_model and hrnet_model:
            self.detection_core = DetectionCore(anomalib_model, hrnet_model, self.device)
            self.image_processor = ImageProcessor(self.detection_core)
            self.video_processor = VideoProcessor(self.detection_core)
            
            # Initialize real-time processor
            try:
                performance_tracker = getattr(self, 'performance_tracker', None)
                self.realtime_processor = RealTimeProcessor(
                    self.detection_core, 
                    performance_tracker
                )
                print("✅ Real-time processor initialized")
            except Exception as e:
                print(f"⚠️ Real-time processor initialization failed: {e}")
                self.realtime_processor = None
            
            return True
        return False
    
    def is_ready(self):
        """Check if system is ready for processing"""
        return (self.model_loader.is_ready() and 
                self.detection_core is not None and 
                self.image_processor is not None and 
                self.video_processor is not None)
    
    # Image Processing Methods
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image for defect detection
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
            
        Returns:
            dict: Complete analysis results
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.image_processor.process_single_image(image_path, output_dir)
    
    def process_batch(self, input_folder, output_folder=None):
        """
        Process all images in a folder
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Directory to save results (optional)
            
        Returns:
            dict: Batch processing results and summary
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.image_processor.process_batch_images(input_folder, output_folder)
    
    # Video Processing Methods
    def process_video(self, video_path, output_dir=None, save_video=True, frame_skip=None):
        """
        Process video file for defect detection
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save results (optional)
            save_video: Whether to save processed video
            frame_skip: Number of frames to skip (optional)
            
        Returns:
            dict: Video processing statistics
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.video_processor.process_video(video_path, output_dir, save_video, frame_skip)
    
    def start_camera(self, camera_id=0, output_dir=None):
        """
        Start real-time camera processing
        
        Args:
            camera_id: Camera device ID (default: 0)
            output_dir: Directory to save captured frames (optional)
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.video_processor.process_camera_realtime(camera_id, output_dir)
    
    # Real-Time Processing Methods
    def start_realtime_session(self):
        """
        Start a new real-time detection session
        
        Returns:
            bool: Success status
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        if not self.realtime_processor:
            raise RuntimeError("Real-time processor not available.")
        
        return self.realtime_processor.start_session()

    def stop_realtime_session(self):
        """
        Stop current real-time detection session
        
        Returns:
            dict: Session report
        """
        if not self.realtime_processor:
            raise RuntimeError("Real-time processor not available.")
        
        return self.realtime_processor.stop_session()

    def process_realtime_frame(self, frame_data, auto_capture_defects=True):
        """
        Process single frame from real-time camera feed
        
        Args:
            frame_data: Base64 encoded image data or numpy array
            auto_capture_defects: Automatically capture screenshots of defects
            
        Returns:
            dict: Frame processing result
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        if not self.realtime_processor:
            raise RuntimeError("Real-time processor not available.")
        
        return self.realtime_processor.process_frame(frame_data, auto_capture_defects)

    def capture_realtime_screenshot(self, frame_data, detection_result=None):
        """
        Manually capture screenshot from real-time feed
        
        Args:
            frame_data: Base64 encoded image data
            detection_result: Optional detection result to associate
            
        Returns:
            dict: Screenshot information
        """
        if not self.realtime_processor:
            raise RuntimeError("Real-time processor not available.")
        
        return self.realtime_processor.capture_manual_screenshot(frame_data, detection_result)

    def get_realtime_session_stats(self):
        """
        Get current real-time session statistics
        
        Returns:
            dict: Session statistics
        """
        if not self.realtime_processor:
            return {'error': 'Real-time processor not available'}
        
        return self.realtime_processor.get_session_statistics()

    def is_realtime_session_active(self):
        """
        Check if real-time session is currently active
        
        Returns:
            bool: Session active status
        """
        return (self.realtime_processor and 
                self.realtime_processor.session_active)

    def get_realtime_session_history(self, limit=20):
        """
        Get real-time session history
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            list: Session history records
        """
        if not self.realtime_processor:
            return []
        
        return self.realtime_processor.get_session_history(limit)

    def cleanup_realtime_data(self, days_to_keep=30):
        """
        Clean up old real-time data and files
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        if self.realtime_processor:
            self.realtime_processor.cleanup_old_sessions(days_to_keep)
    
    # Direct Detection Methods (for advanced users)
    def detect_anomaly(self, image_path):
        """
        Run only anomaly detection step
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: Anomaly detection results
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.detection_core.detect_anomaly(image_path)
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Run only defect classification step
        
        Args:
            image_path: Path to image
            region_mask: Optional region mask
            
        Returns:
            dict: Defect classification results
        """
        if not self.is_ready():
            raise RuntimeError("System not ready. Load models first.")
        
        return self.detection_core.classify_defects(image_path, region_mask)
    
    # Utility Methods
    def get_system_info(self):
        """Get system information and status including real-time capabilities"""
        base_info = {
            'device': self.device,
            'models_loaded': self.model_loader.is_ready(),
            'system_ready': self.is_ready(),
            'anomaly_threshold': ANOMALY_THRESHOLD,
            'defect_threshold': DEFECT_CONFIDENCE_THRESHOLD,
            'supported_classes': SPECIFIC_DEFECT_CLASSES
        }
        
        # Add real-time capabilities
        base_info.update({
            'realtime_available': self.realtime_processor is not None,
            'realtime_session_active': self.is_realtime_session_active(),
            'realtime_features': {
                'live_detection': True,
                'auto_screenshot': True,
                'session_reports': True,
                'performance_tracking': hasattr(self, 'performance_tracker') and self.performance_tracker is not None
            }
        })
        
        return base_info
    
    def update_thresholds(self, anomaly_threshold=None, defect_threshold=None):
        """
        Update detection thresholds
        
        Args:
            anomaly_threshold: New anomaly detection threshold (0.0-1.0)
            defect_threshold: New defect classification threshold (0.0-1.0)
        """
        global ANOMALY_THRESHOLD, DEFECT_CONFIDENCE_THRESHOLD
        
        if anomaly_threshold is not None:
            ANOMALY_THRESHOLD = anomaly_threshold
            print(f"Anomaly threshold updated to: {anomaly_threshold}")
        
        if defect_threshold is not None:
            DEFECT_CONFIDENCE_THRESHOLD = defect_threshold
            print(f"Defect confidence threshold updated to: {defect_threshold}")


# Convenience functions for quick usage
def create_detector(anomalib_path=None, hrnet_path=None, device=None):
    """
    Quick detector creation with automatic model loading
    
    Args:
        anomalib_path: Path to Anomalib model (optional, uses config if None)
        hrnet_path: Path to HRNet model (optional, uses config if None)
        device: Device to use (optional, uses config if None)
        
    Returns:
        UnifiedDefectDetector: Ready-to-use detector instance
    """
    return UnifiedDefectDetector(
        anomalib_model_path=anomalib_path,
        hrnet_model_path=hrnet_path,
        device=device,
        auto_load=True
    )

def create_realtime_detector(anomalib_path=None, hrnet_path=None, device=None):
    """
    Quick real-time detector creation with automatic model loading
    
    Args:
        anomalib_path: Path to Anomalib model (optional, uses config if None)
        hrnet_path: Path to HRNet model (optional, uses config if None)
        device: Device to use (optional, uses config if None)
        
    Returns:
        UnifiedDefectDetector: Ready-to-use detector instance with real-time capabilities
    """
    detector = UnifiedDefectDetector(
        anomalib_model_path=anomalib_path,
        hrnet_model_path=hrnet_path,
        device=device,
        auto_load=True
    )
    
    if detector.realtime_processor:
        print("✅ Real-time detector ready with live processing capabilities")
    else:
        print("⚠️ Real-time detector created but real-time processor unavailable")
    
    return detector


def quick_detect(image_path, anomalib_path=None, hrnet_path=None):
    """
    Quick single image detection
    
    Args:
        image_path: Path to image to analyze
        anomalib_path: Path to Anomalib model (optional)
        hrnet_path: Path to HRNet model (optional)
        
    Returns:
        dict: Detection results
    """
    detector = create_detector(anomalib_path, hrnet_path)
    return detector.process_image(image_path)


# Demo and testing functions
def demo_single_image(image_path):
    """Demo function for single image processing"""
    print("DEMO: Single Image Processing")
    print("=" * 50)
    
    detector = create_detector()
    result = detector.process_image(image_path)
    
    if result:
        print(f"Results for {os.path.basename(image_path)}:")
        print(f"   Final Decision: {result['final_decision']}")
        print(f"   Anomaly Score: {result['anomaly_detection']['anomaly_score']:.4f}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        
        if result['final_decision'] == 'DEFECT' and 'detected_defect_types' in result:
            print(f"   Defect Types: {result['detected_defect_types']}")
        
        print(f"   Visualization: {result.get('visualization_path', 'Not saved')}")
        print(f"   Report: {result.get('report_path', 'Not saved')}")
    else:
        print("Processing failed")


def demo_batch_processing(input_folder):
    """Demo function for batch processing"""
    print("DEMO: Batch Processing")
    print("=" * 50)
    
    detector = create_detector()
    batch_result = detector.process_batch(input_folder)
    
    if batch_result:
        summary = batch_result['summary']
        print(f"Batch Processing Summary:")
        print(f"   Total Images: {summary['total_images']}")
        print(f"   Good Products: {summary['good_products']}")
        print(f"   Defective Products: {summary['defective_products']}")
        print(f"   Defect Types Found: {summary['defect_types_found']}")
        print(f"   Average Processing Time: {summary['avg_processing_time']:.2f}s")
    else:
        print("Batch processing failed")

def demo_realtime_processing():
    """Demo function for real-time processing capabilities"""
    print("DEMO: Real-Time Processing Capabilities")
    print("=" * 50)
    
    detector = create_realtime_detector()
    
    if detector.realtime_processor:
        print("Real-time processor available!")
        print("\nAvailable real-time methods:")
        print("   - detector.start_realtime_session()")
        print("   - detector.process_realtime_frame(frame_data)")
        print("   - detector.capture_realtime_screenshot(frame_data)")
        print("   - detector.get_realtime_session_stats()")
        print("   - detector.stop_realtime_session()")
        print("\nReal-time features:")
        print("   ✅ Live camera feed processing")
        print("   ✅ Real-time bounding box detection")
        print("   ✅ Automatic defect screenshot capture")
        print("   ✅ Session statistics and reporting")
        print("   ✅ Performance tracking integration")
        print("   ✅ Database storage for session data")
    else:
        print("Real-time processor not available. Check model loading.")


if __name__ == "__main__":
    # Example usage
    print("Unified Defect Detection System - Enhanced with Real-Time Processing")
    print("=" * 80)
    
    # Initialize detector
    detector = create_realtime_detector()
    
    if detector.is_ready():
        print("System ready for processing!")
        print("\nAvailable methods:")
        print("   - detector.process_image(image_path)")
        print("   - detector.process_batch(folder_path)")
        print("   - detector.process_video(video_path)")
        print("   - detector.start_camera()")
        
        if detector.realtime_processor:
            print("\n🚀 Real-Time Processing Available:")
            print("   - detector.start_realtime_session()")
            print("   - detector.process_realtime_frame(frame_data)")
            print("   - detector.capture_realtime_screenshot(frame_data)")
            print("   - detector.get_realtime_session_stats()")
            print("   - detector.stop_realtime_session()")
            print("\n   📱 Web Interface: http://localhost:5000/realtime")
        
        print("\n   - quick_detect(image_path)  # Convenience function")
    else:
        print("System not ready. Check model paths in config.py")
    
    print("\nSystem is now enhanced with real-time capabilities and ready for integration!")