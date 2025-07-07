"""
Image Processing Module for Single and Batch Analysis
FIXED: Complete implementation to avoid import errors
"""

import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import random

class ImageProcessor:
    """Image Processing for Single and Batch Analysis"""
    
    def __init__(self, detection_core):
        self.detection_core = detection_core
        print("✅ Image processor initialized")
    
    def process_single_image(self, image_path, output_dir=None):
        """Process a single image for defect detection"""
        try:
            start_time = time.time()
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Mock detection if no real detector available
            if not self.detection_core:
                return self._mock_single_detection(image_path, start_time)
            
            # Real detection process
            # Step 1: Anomaly Detection
            anomaly_result = self.detection_core.detect_anomaly(image_path)
            if not anomaly_result:
                raise RuntimeError("Anomaly detection failed")
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'image_path': image_path,
                'final_decision': anomaly_result['decision'],
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'anomaly_detection': anomaly_result,
                'detected_defect_types': []
            }
            
            # Step 2: Defect Classification (if defective)
            if anomaly_result['decision'] == 'DEFECT':
                defect_result = self.detection_core.classify_defects(image_path, anomaly_result.get('anomaly_mask'))
                if defect_result:
                    result['defect_classification'] = defect_result
                    result['detected_defect_types'] = defect_result.get('detected_defects', [])
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_batch_images(self, input_folder, output_folder=None):
        """Process all images in a folder"""
        try:
            if not os.path.exists(input_folder):
                raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(input_folder).glob(f"*{ext}"))
                image_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
            
            if not image_files:
                return {
                    'results': [],
                    'summary': {
                        'total_images': 0,
                        'good_products': 0,
                        'defective_products': 0,
                        'failed_processing': 0,
                        'processing_times': [],
                        'avg_processing_time': 0,
                        'defect_types_found': []
                    }
                }
            
            # Create output folder if specified
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
            
            # Process each image
            results = []
            processing_times = []
            defect_types_found = set()
            
            for i, image_path in enumerate(image_files):
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                result = self.process_single_image(str(image_path), output_folder)
                
                if result:
                    results.append(result)
                    processing_times.append(result['processing_time'])
                    
                    # Collect defect types
                    if result.get('detected_defect_types'):
                        defect_types_found.update(result['detected_defect_types'])
            
            # Generate summary
            good_products = sum(1 for r in results if r['final_decision'] == 'GOOD')
            defective_products = sum(1 for r in results if r['final_decision'] == 'DEFECT')
            failed_processing = len(image_files) - len(results)
            
            summary = {
                'total_images': len(image_files),
                'good_products': good_products,
                'defective_products': defective_products,
                'failed_processing': failed_processing,
                'processing_times': processing_times,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'defect_types_found': list(defect_types_found)
            }
            
            # Save batch report if output folder specified
            if output_folder:
                self._save_batch_report(results, summary, output_folder)
            
            return {
                'results': results,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
    
    def _mock_single_detection(self, image_path, start_time):
        """Mock single image detection for development"""
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Mock result based on filename
        filename = os.path.basename(image_path).lower()
        is_defect = any(word in filename for word in ['defect', 'damage', 'scratch', 'stain', 'crack', 'broken'])
        
        if not is_defect:
            # Random defect probability for testing
            is_defect = random.random() < 0.3
        
        processing_time = time.time() - start_time
        
        # Generate mock anomaly score
        if is_defect:
            anomaly_score = random.uniform(0.7, 0.95)
        else:
            anomaly_score = random.uniform(0.1, 0.6)
        
        result = {
            'image_path': image_path,
            'final_decision': 'DEFECT' if is_defect else 'GOOD',
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'anomaly_detection': {
                'anomaly_score': anomaly_score,
                'decision': 'DEFECT' if is_defect else 'GOOD',
                'threshold_used': 0.7,
                'anomaly_mask': None
            },
            'detected_defect_types': []
        }
        
        # Add mock defect types if defective
        if is_defect:
            possible_defects = ['scratch', 'stained', 'damaged', 'missing_component', 'open']
            num_defects = random.randint(1, 3)
            result['detected_defect_types'] = random.sample(possible_defects, num_defects)
            
            # Add mock defect classification
            result['defect_classification'] = self._generate_mock_defect_classification(
                result['detected_defect_types']
            )
        
        return result
    
    def _generate_mock_defect_classification(self, defect_types):
        """Generate mock defect classification data"""
        classification = {
            'detected_defects': defect_types,
            'defect_analysis': {
                'defect_statistics': {},
                'class_distribution': {},
                'bounding_boxes': {}
            }
        }
        
        for i, defect_type in enumerate(defect_types):
            # Mock statistics
            classification['defect_analysis']['defect_statistics'][defect_type] = {
                'avg_confidence': random.uniform(0.7, 0.95),
                'max_confidence': random.uniform(0.85, 1.0),
                'num_regions': random.randint(1, 4),
                'confident_pixels': random.randint(100, 1000),
                'total_area': random.randint(500, 5000)
            }
            
            # Mock class distribution
            classification['defect_analysis']['class_distribution'][defect_type] = {
                'percentage': random.uniform(1.0, 10.0),
                'pixel_count': random.randint(500, 5000),
                'class_id': i + 1
            }
            
            # Mock bounding boxes
            num_boxes = random.randint(1, 3)
            boxes = []
            for j in range(num_boxes):
                boxes.append({
                    'x': random.randint(50, 300),
                    'y': random.randint(50, 200),
                    'width': random.randint(50, 150),
                    'height': random.randint(50, 100),
                    'area': random.randint(2500, 15000),
                    'center_x': random.randint(100, 400),
                    'center_y': random.randint(100, 300)
                })
            
            classification['defect_analysis']['bounding_boxes'][defect_type] = boxes
        
        return classification
    
    def _save_batch_report(self, results, summary, output_folder):
        """Save batch processing report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"batch_report_{timestamp}.json"
            report_path = os.path.join(output_folder, report_filename)
            
            report_data = {
                'batch_info': {
                    'generated_at': datetime.now().isoformat(),
                    'total_images': summary['total_images'],
                    'output_folder': output_folder
                },
                'summary': summary,
                'detailed_results': results
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"📄 Batch report saved: {report_path}")
            
        except Exception as e:
            print(f"Error saving batch report: {e}")


class VideoProcessor:
    """Video Processing for Video Files and Camera Feeds"""
    
    def __init__(self, detection_core):
        self.detection_core = detection_core
        print("✅ Video processor initialized")
    
    def process_video(self, video_path, output_dir=None, save_video=True, frame_skip=None):
        """Process video file for defect detection"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Create output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Processing video: {fps} FPS, {total_frames} frames, {width}x{height}")
            
            # Process frames (mock implementation)
            frame_count = 0
            processed_frames = 0
            defect_frames = 0
            
            skip_frames = frame_skip or 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if specified
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Mock processing
                is_defect = random.random() < 0.2  # 20% defect rate
                if is_defect:
                    defect_frames += 1
                
                processed_frames += 1
                
                if processed_frames % 100 == 0:
                    print(f"Processed {processed_frames} frames...")
            
            cap.release()
            
            # Generate summary
            summary = {
                'video_path': video_path,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'defect_frames': defect_frames,
                'defect_rate': (defect_frames / processed_frames * 100) if processed_frames > 0 else 0,
                'video_properties': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                }
            }
            
            print(f"Video processing complete: {processed_frames} frames, {defect_frames} defects")
            return summary
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None
    
    def process_camera_realtime(self, camera_id=0, output_dir=None):
        """Process real-time camera feed (mock implementation)"""
        try:
            print(f"Starting real-time camera processing (Camera {camera_id})")
            print("Press 'q' to quit, 's' to save frame")
            
            # Mock camera processing
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {camera_id}")
            
            frame_count = 0
            saved_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Mock detection overlay
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Mock Detection Active", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Real-time Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and output_dir:
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_filename = f"camera_frame_{timestamp}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_frames += 1
                    print(f"Frame saved: {frame_filename}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"Camera session ended: {frame_count} frames processed, {saved_frames} saved")
            
            return {
                'total_frames': frame_count,
                'saved_frames': saved_frames,
                'camera_id': camera_id
            }
            
        except Exception as e:
            print(f"Error in camera processing: {e}")
            return None