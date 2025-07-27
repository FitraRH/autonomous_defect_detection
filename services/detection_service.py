# services/detection_service.py
"""
Detection Service - Business Logic Layer
Handles all detection-related business logic
Integrates with existing core, models, and processors
"""

import os
import cv2
import numpy as np
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

# Import existing modules from your project structure
from main import create_detector
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from processors.realtime_processor import RealTimeProcessor
from utils.performance_tracker import EnhancedPerformanceTracker


class DetectionService:
    """
    Detection Service - Core business logic for defect detection
    Utilizes existing project components without modification
    """
    
    def __init__(self):
        # Initialize existing components
        self.detector = None
        self.image_processor = None
        self.video_processor = None
        self.realtime_processor = None
        self.performance_tracker = None
        
        # Service state
        self.is_initialized = False
        self.initialization_error = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all detection components using existing code"""
        try:
            print("Initializing detection service components...")
            
            # Initialize main detector (from your existing main.py)
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Main detector initialization failed")
            
            # Initialize performance tracker (from your existing utils)
            try:
                self.performance_tracker = EnhancedPerformanceTracker()
                print("Performance tracker initialized")
            except Exception as e:
                print(f"Performance tracker initialization failed: {e}")
                self.performance_tracker = None
            
            # Initialize processors (from your existing processors)
            if hasattr(self.detector, 'detection_core') and self.detector.detection_core:
                self.image_processor = ImageProcessor(self.detector.detection_core)
                self.video_processor = VideoProcessor(self.detector.detection_core)
                
                # Initialize real-time processor if available
                try:
                    self.realtime_processor = RealTimeProcessor(
                        self.detector.detection_core, 
                        self.performance_tracker
                    )
                    print("Real-time processor initialized")
                except Exception as e:
                    print(f"Real-time processor initialization failed: {e}")
                    self.realtime_processor = None
                
                print("All processors initialized successfully")
            else:
                print("Warning: Detection core not available, using basic detection only")
            
            self.is_initialized = True
            print("Detection service initialization completed")
            
        except Exception as e:
            self.initialization_error = str(e)
            print(f"Detection service initialization failed: {e}")
            self.is_initialized = False
    
    def get_health_status(self):
        """Get health status of all components"""
        return {
            'detector': {
                'available': self.detector is not None,
                'ready': self.detector.is_ready() if self.detector else False,
                'status': 'operational' if self.detector and self.detector.is_ready() else 'not_ready'
            },
            'image_processor': {
                'available': self.image_processor is not None,
                'status': 'operational' if self.image_processor else 'not_available'
            },
            'video_processor': {
                'available': self.video_processor is not None,
                'status': 'operational' if self.video_processor else 'not_available'
            },
            'realtime_processor': {
                'available': self.realtime_processor is not None,
                'status': 'operational' if self.realtime_processor else 'not_available'
            },
            'performance_tracker': {
                'available': self.performance_tracker is not None,
                'status': 'operational' if self.performance_tracker else 'not_available'
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error
        }
    
    def get_system_information(self):
        """Get detailed system information using existing detector methods"""
        if not self.detector:
            return {
                'status': 'not_initialized',
                'error': 'Detector not available'
            }
        
        try:
            # Use existing detector's get_system_info method
            system_info = self.detector.get_system_info()
            
            # Add service-specific information
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'components': {
                    'main_detector': True,
                    'image_processor': self.image_processor is not None,
                    'video_processor': self.video_processor is not None,
                    'realtime_processor': self.realtime_processor is not None,
                    'performance_tracker': self.performance_tracker is not None
                },
                'api_version': '1.0.0',
                'service_uptime': self._get_service_uptime()
            })
            
            return system_info
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'fallback_info': {
                    'device': 'unknown',
                    'models_loaded': False,
                    'system_ready': False
                }
            }
    
    def get_current_status(self):
        """Get current system status"""
        return {
            'system_ready': self.is_initialized,
            'detector_ready': self.detector.is_ready() if self.detector else False,
            'processing_capabilities': {
                'single_image': self.detector is not None,
                'batch_processing': self.image_processor is not None,
                'video_processing': self.video_processor is not None,
                'realtime_processing': self.realtime_processor is not None
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'last_check': datetime.now().isoformat()
        }
    
    def process_single_image(self, image_data, filename):
        """Process single image using existing image processor or detector"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            # Save image data to temporary file
            temp_path = self._save_temp_image(image_data, filename)
            
            # Start performance tracking if available
            if self.performance_tracker:
                perf_start = self.performance_tracker.start_measurement()
            
            # Process using existing components
            if self.image_processor:
                # Use advanced image processor
                result = self.image_processor.process_single_image(temp_path)
            else:
                # Fallback to basic detector
                result = self.detector.process_image(temp_path)
            
            # End performance tracking
            if self.performance_tracker and perf_start:
                self.performance_tracker.end_measurement(perf_start, result)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Cleanup on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def process_image_batch(self, batch_data):
        """Process batch of images using existing batch processor"""
        if not self.image_processor:
            raise RuntimeError("Batch processing not available - image processor not initialized")
        
        try:
            # Create temporary directory for batch
            temp_dir = tempfile.mkdtemp(prefix="batch_")
            
            # Save all images to temp directory
            temp_paths = []
            for i, image_item in enumerate(batch_data):
                temp_path = os.path.join(temp_dir, f"batch_image_{i+1}_{image_item['filename']}")
                with open(temp_path, 'wb') as f:
                    f.write(image_item['data'])
                temp_paths.append(temp_path)
            
            # Process batch using existing batch processor
            batch_result = self.image_processor.process_batch_images(temp_dir)
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return batch_result.get('results', []) if batch_result else []
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Cleanup on error
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    def process_video(self, video_data):
        """Process video using existing video processor"""
        if not self.video_processor:
            raise RuntimeError("Video processing not available - video processor not initialized")
        
        try:
            # Save video data to temporary file
            temp_path = self._save_temp_video(video_data['data'], video_data['filename'])
            
            # Process using existing video processor
            result = self.video_processor.process_video(temp_path)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Error processing video: {e}")
            # Cleanup on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def start_realtime_session(self):
        """Start real-time detection session"""
        if not self.realtime_processor:
            raise RuntimeError("Real-time processing not available")
        
        try:
            success = self.realtime_processor.start_session()
            if success:
                return self.realtime_processor.current_session_id
            else:
                raise RuntimeError("Failed to start real-time session")
                
        except Exception as e:
            print(f"Error starting real-time session: {e}")
            raise
    
    def stop_realtime_session(self, session_id):
        """Stop real-time detection session"""
        if not self.realtime_processor:
            raise RuntimeError("Real-time processing not available")
        
        try:
            session_report = self.realtime_processor.stop_session()
            return session_report
            
        except Exception as e:
            print(f"Error stopping real-time session: {e}")
            raise
    
    def process_realtime_frame(self, session_id, frame_data):
        """Process single frame in real-time session"""
        if not self.realtime_processor:
            raise RuntimeError("Real-time processing not available")
        
        try:
            # Convert frame data to base64 string (expected by realtime processor)
            import base64
            frame_base64 = base64.b64encode(frame_data['data']).decode('utf-8')
            
            # Process frame using existing realtime processor
            result = self.realtime_processor.process_frame(frame_base64)
            
            # Add frame metadata
            result['frame_id'] = f"{session_id}_{int(time.time())}"
            result['timestamp'] = frame_data.get('timestamp', time.time())
            
            return result
            
        except Exception as e:
            print(f"Error processing real-time frame: {e}")
            raise
    
    def get_realtime_session_stats(self, session_id):
        """Get real-time session statistics"""
        if not self.realtime_processor:
            raise RuntimeError("Real-time processing not available")
        
        try:
            return self.realtime_processor.get_session_statistics()
        except Exception as e:
            print(f"Error getting session stats: {e}")
            raise
    
    def get_thresholds(self):
        """Get current detection thresholds from config"""
        from config import ANOMALY_THRESHOLD, DEFECT_CONFIDENCE_THRESHOLD
        
        return {
            'anomaly_threshold': ANOMALY_THRESHOLD,
            'defect_confidence_threshold': DEFECT_CONFIDENCE_THRESHOLD,
            'configurable': True,
            'last_updated': datetime.now().isoformat()
        }
    
    def update_thresholds(self, new_thresholds):
        """Update detection thresholds"""
        try:
            # In a real implementation, you would update the config file
            # For now, we'll just validate the input
            required_fields = ['anomaly_threshold', 'defect_confidence_threshold']
            
            for field in required_fields:
                if field not in new_thresholds:
                    raise ValueError(f"Missing required field: {field}")
                
                value = new_thresholds[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    raise ValueError(f"Invalid value for {field}: must be between 0 and 1")
            
            # Update would happen here in real implementation
            print(f"Thresholds update requested: {new_thresholds}")
            return True
            
        except Exception as e:
            print(f"Error updating thresholds: {e}")
            return False
    
    def _save_temp_image(self, image_data, filename):
        """Save image data to temporary file"""
        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(filename)[1] or '.jpg'
        temp_filename = f"temp_image_{uuid.uuid4().hex}{file_ext}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        return temp_path
    
    def _save_temp_video(self, video_data, filename):
        """Save video data to temporary file"""
        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(filename)[1] or '.mp4'
        temp_filename = f"temp_video_{uuid.uuid4().hex}{file_ext}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(video_data)
        
        return temp_path
    
    def _get_service_uptime(self):
        """Get service uptime (mock implementation)"""
        # In real implementation, track actual start time
        return "Service uptime tracking not implemented"
    
    def _get_current_load(self):
        """Get current system load"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'status': 'normal'
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'status': 'monitoring_unavailable'
            }
    
    def _get_memory_usage(self):
        """Get memory usage information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent_used': memory.percent
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0
            }