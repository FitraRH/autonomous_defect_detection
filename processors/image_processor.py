"""
Image Processing Module for Single and Batch Analysis
Real implementation without mock data
"""

import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

class ImageProcessor:
    """Image Processing for Single and Batch Analysis"""
    
    def __init__(self, detection_core):
        if not detection_core:
            raise ValueError("Detection core is required for image processing")
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
            
            # Verify detection core is available
            if not self.detection_core:
                raise RuntimeError("Detection core not available")
            
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
            
            # Verify detection core is available
            if not self.detection_core:
                raise RuntimeError("Detection core not available for batch processing")
            
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
            failed_count = 0
            
            print(f"Processing {len(image_files)} images...")
            
            for i, image_path in enumerate(image_files):
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                try:
                    result = self.process_single_image(str(image_path), output_folder)
                    
                    if result:
                        results.append(result)
                        processing_times.append(result['processing_time'])
                        
                        # Collect defect types
                        if result.get('detected_defect_types'):
                            defect_types_found.update(result['detected_defect_types'])
                    else:
                        failed_count += 1
                        print(f"Failed to process: {image_path.name}")
                        
                except Exception as e:
                    failed_count += 1
                    print(f"Error processing {image_path.name}: {e}")
            
            # Generate summary
            good_products = sum(1 for r in results if r['final_decision'] == 'GOOD')
            defective_products = sum(1 for r in results if r['final_decision'] == 'DEFECT')
            
            summary = {
                'total_images': len(image_files),
                'good_products': good_products,
                'defective_products': defective_products,
                'failed_processing': failed_count,
                'processing_times': processing_times,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'min_processing_time': np.min(processing_times) if processing_times else 0,
                'max_processing_time': np.max(processing_times) if processing_times else 0,
                'total_processing_time': np.sum(processing_times) if processing_times else 0,
                'defect_types_found': list(defect_types_found),
                'success_rate': ((len(results) / len(image_files)) * 100) if image_files else 0
            }
            
            # Save batch report if output folder specified
            if output_folder:
                self._save_batch_report(results, summary, output_folder)
            
            print(f"Batch processing complete:")
            print(f"  - Total images: {summary['total_images']}")
            print(f"  - Successfully processed: {len(results)}")
            print(f"  - Good products: {good_products}")
            print(f"  - Defective products: {defective_products}")
            print(f"  - Failed processing: {failed_count}")
            print(f"  - Average processing time: {summary['avg_processing_time']:.3f}s")
            
            return {
                'results': results,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
    
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
                    'output_folder': output_folder,
                    'processing_mode': 'real_detection'
                },
                'summary': summary,
                'detailed_results': results
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"📄 Batch report saved: {report_path}")
            
            # Also save a summary text file
            self._save_batch_summary_text(summary, output_folder, timestamp)
            
        except Exception as e:
            print(f"Error saving batch report: {e}")
    
    def _save_batch_summary_text(self, summary, output_folder, timestamp):
        """Save human-readable batch summary"""
        try:
            summary_filename = f"batch_summary_{timestamp}.txt"
            summary_path = os.path.join(output_folder, summary_filename)
            
            with open(summary_path, 'w') as f:
                f.write("BATCH PROCESSING SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processing Mode: Real Detection\n\n")
                
                f.write("RESULTS:\n")
                f.write(f"  Total Images: {summary['total_images']}\n")
                f.write(f"  Good Products: {summary['good_products']} ({(summary['good_products']/summary['total_images']*100):.1f}%)\n")
                f.write(f"  Defective Products: {summary['defective_products']} ({(summary['defective_products']/summary['total_images']*100):.1f}%)\n")
                f.write(f"  Failed Processing: {summary['failed_processing']}\n")
                f.write(f"  Success Rate: {summary['success_rate']:.1f}%\n\n")
                
                f.write("PERFORMANCE:\n")
                f.write(f"  Average Processing Time: {summary['avg_processing_time']:.3f}s\n")
                f.write(f"  Minimum Processing Time: {summary['min_processing_time']:.3f}s\n")
                f.write(f"  Maximum Processing Time: {summary['max_processing_time']:.3f}s\n")
                f.write(f"  Total Processing Time: {summary['total_processing_time']:.1f}s\n")
                f.write(f"  Throughput: {summary['total_images']/summary['total_processing_time']:.2f} images/second\n\n")
                
                if summary['defect_types_found']:
                    f.write("DEFECT TYPES FOUND:\n")
                    for defect_type in sorted(summary['defect_types_found']):
                        f.write(f"  - {defect_type.replace('_', ' ').title()}\n")
                else:
                    f.write("DEFECT TYPES FOUND: None\n")
            
            print(f"📄 Batch summary saved: {summary_path}")
            
        except Exception as e:
            print(f"Error saving batch summary: {e}")
    
    def validate_detection_core(self):
        """Validate that detection core is properly configured"""
        if not self.detection_core:
            return False, "Detection core not initialized"
        
        try:
            # Check if detection core has required methods
            required_methods = ['detect_anomaly', 'classify_defects']
            for method in required_methods:
                if not hasattr(self.detection_core, method):
                    return False, f"Detection core missing {method} method"
            
            return True, "Detection core validated"
            
        except Exception as e:
            return False, f"Detection core validation failed: {e}"
    
    def get_processor_info(self):
        """Get information about the image processor"""
        valid, message = self.validate_detection_core()
        
        return {
            'processor_type': 'ImageProcessor',
            'detection_core_available': self.detection_core is not None,
            'detection_core_valid': valid,
            'validation_message': message,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            'capabilities': {
                'single_image_processing': True,
                'batch_processing': True,
                'anomaly_detection': valid,
                'defect_classification': valid,
                'report_generation': True
            }
        }


class VideoProcessor:
    """Video Processing for Video Files and Camera Feeds"""
    
    def __init__(self, detection_core):
        if not detection_core:
            raise ValueError("Detection core is required for video processing")
        self.detection_core = detection_core
        self.image_processor = ImageProcessor(detection_core)
        self.bg_subtractor = None
        self.reset_background_model()
        print("✅ Video processor initialized")
    
    def reset_background_model(self):
        """Reset background subtractor for video processing"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)
    
    def process_video(self, video_path, output_dir=None, save_video=True, frame_skip=None):
        """Process video file for defect detection using real detection"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Verify detection core is available
            if not self.detection_core:
                raise RuntimeError("Detection core not available for video processing")
            
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
            
            # Initialize video writer if saving
            video_writer = None
            if save_video and output_dir:
                output_video_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Process frames using real detection
            frame_count = 0
            processed_frames = 0
            defect_frames = 0
            good_frames = 0
            failed_frames = 0
            processing_times = []
            
            skip_frames = frame_skip or 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames if specified
                    if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                        if video_writer:
                            video_writer.write(frame)
                        continue
                    
                    # Process frame with real detection
                    processed_frame, frame_result = self._process_video_frame(
                        frame, output_dir, frame_count, total_frames
                    )
                    
                    if frame_result:
                        processed_frames += 1
                        processing_times.append(frame_result.get('processing_time', 0))
                        
                        if frame_result.get('final_decision') == 'DEFECT':
                            defect_frames += 1
                        elif frame_result.get('final_decision') == 'GOOD':
                            good_frames += 1
                    else:
                        failed_frames += 1
                    
                    # Save processed frame
                    if video_writer:
                        video_writer.write(processed_frame)
                    
                    # Progress update
                    if processed_frames % 50 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
            finally:
                cap.release()
                if video_writer:
                    video_writer.release()
            
            # Generate summary
            summary = {
                'video_path': video_path,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'good_frames': good_frames,
                'defect_frames': defect_frames,
                'failed_frames': failed_frames,
                'defect_rate': (defect_frames / processed_frames * 100) if processed_frames > 0 else 0,
                'success_rate': (processed_frames / total_frames * 100) if total_frames > 0 else 0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'total_processing_time': np.sum(processing_times) if processing_times else 0,
                'video_properties': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                }
            }
            
            print(f"Video processing complete:")
            print(f"  - Total frames: {total_frames}")
            print(f"  - Processed frames: {processed_frames}")
            print(f"  - Good frames: {good_frames}")
            print(f"  - Defect frames: {defect_frames}")
            print(f"  - Failed frames: {failed_frames}")
            print(f"  - Defect rate: {summary['defect_rate']:.1f}%")
            
            return summary
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None
    
    def _process_video_frame(self, frame, output_dir, frame_idx, total_frames):
        """Process a single video frame using real detection"""
        try:
            # Save frame temporarily for processing
            temp_frame_path = os.path.join(output_dir, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Process frame with real detection
            result = self.image_processor.process_single_image(temp_frame_path)
            
            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            # Annotate frame with real results
            annotated_frame = self._annotate_frame(frame, result, frame_idx, total_frames)
            
            return annotated_frame, result
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return frame, None
    
    def _annotate_frame(self, frame, result, frame_idx, total_frames):
        """Add annotations to video frame based on real detection results"""
        height = frame.shape[0]
        annotated_frame = frame.copy()
        
        if result:
            if result['final_decision'] == 'GOOD':
                # Add green border for good products
                annotated_frame = cv2.copyMakeBorder(annotated_frame, 5, 5, 5, 5, 
                                                   cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated_frame, "GOOD", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif result['final_decision'] == 'DEFECT':
                # Add red border for defective products
                annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, 
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated_frame, "DEFECT", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw real defect bounding boxes if available
                if result.get('defect_classification'):
                    defect_analysis = result['defect_classification'].get('defect_analysis', {})
                    bounding_boxes = defect_analysis.get('bounding_boxes', {})
                    
                    for defect_type, boxes in bounding_boxes.items():
                        for bbox in boxes:
                            x, y = bbox['x'], bbox['y']
                            w, h = bbox['width'], bbox['height']
                            
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, defect_type.upper(),
                                      (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Add processing info
            cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", (20, height - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Score: {result['anomaly_detection']['anomaly_score']:.3f}", 
                      (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Frame processing failed
            cv2.putText(annotated_frame, "PROCESSING FAILED", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", (20, height - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_camera_realtime(self, camera_id=0, output_dir=None):
        """Process real-time camera feed using real detection"""
        try:
            print(f"Starting real-time camera processing (Camera {camera_id})")
            print("Controls:")
            print("  - Press 'q' to quit")
            print("  - Press 's' to save frame")
            print("  - Press 'c' to capture and analyze current frame")
            
            # Verify detection core is available
            if not self.detection_core:
                raise RuntimeError("Detection core not available for camera processing")
            
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {camera_id}")
            
            # Create output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            frame_count = 0
            saved_frames = 0
            last_analysis_time = 0
            current_status = "READY"
            last_result = None
            analysis_interval = 2.0  # Analyze every 2 seconds
            
            print("Real-time processing started!")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Periodic analysis with real detection
                if current_time - last_analysis_time > analysis_interval:
                    display_frame, current_status, last_result = self._process_camera_frame(
                        frame, output_dir, current_time
                    )
                    last_analysis_time = current_time
                else:
                    # Just add overlay without processing
                    display_frame = self._add_camera_overlay(frame, current_status, last_result, frame_count)
                
                # Show frame
                cv2.imshow('Real-time Defect Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping real-time processing...")
                    break
                elif key == ord('s') and output_dir:
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_filename = f"camera_frame_{timestamp}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_frames += 1
                    print(f"Frame saved: {frame_filename}")
                elif key == ord('c'):
                    # Force analysis of current frame
                    print("Analyzing current frame...")
                    display_frame, current_status, last_result = self._process_camera_frame(
                        frame, output_dir, current_time
                    )
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"Camera session ended:")
            print(f"  - Frames processed: {frame_count}")
            print(f"  - Frames saved: {saved_frames}")
            
            return {
                'total_frames': frame_count,
                'saved_frames': saved_frames,
                'camera_id': camera_id,
                'last_status': current_status
            }
            
        except Exception as e:
            print(f"Error in camera processing: {e}")
            return None
    
    def _process_camera_frame(self, frame, output_dir, current_time):
        """Process camera frame with real detection"""
        try:
            # Save frame temporarily
            temp_frame_path = os.path.join(output_dir, "temp_camera_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Process with real detection
            result = self.image_processor.process_single_image(temp_frame_path)
            
            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            if result:
                status = result['final_decision']
                print(f"Detection result: {status} (score: {result['anomaly_detection']['anomaly_score']:.3f})")
                if result.get('detected_defect_types'):
                    print(f"  Defects: {', '.join(result['detected_defect_types'])}")
            else:
                status = "ANALYSIS_FAILED"
                result = None
            
            # Create display frame
            display_frame = self._add_camera_overlay(frame, status, result, 0)
            
            return display_frame, status, result
            
        except Exception as e:
            print(f"Error processing camera frame: {e}")
            return self._add_camera_overlay(frame, "ERROR", None, 0), "ERROR", None
    
    def _add_camera_overlay(self, frame, status, result, frame_count):
        """Add overlay information to camera frame"""
        display_frame = frame.copy()
        height = display_frame.shape[0]
        
        # Status-based border and text
        if status == "GOOD":
            color = (0, 255, 0)
            display_frame = cv2.copyMakeBorder(display_frame, 5, 5, 5, 5, 
                                             cv2.BORDER_CONSTANT, value=color)
        elif status == "DEFECT":
            color = (0, 0, 255)
            display_frame = cv2.copyMakeBorder(display_frame, 10, 10, 10, 10, 
                                             cv2.BORDER_CONSTANT, value=color)
        elif status == "ERROR":
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)
        
        # Add status text
        cv2.putText(display_frame, f"Status: {status}", (20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Add detailed info if available
        if result:
            score = result['anomaly_detection']['anomaly_score']
            cv2.putText(display_frame, f"Anomaly Score: {score:.3f}", (20, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if result.get('detected_defect_types'):
                defects_text = ", ".join(result['detected_defect_types'][:3])
                cv2.putText(display_frame, f"Defects: {defects_text}", (20, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame counter and controls
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, height - 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, "Q=Quit, S=Save, C=Analyze", 
                  (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def get_processor_info(self):
        """Get information about the video processor"""
        valid, message = self.image_processor.validate_detection_core()
        
        return {
            'processor_type': 'VideoProcessor',
            'detection_core_available': self.detection_core is not None,
            'detection_core_valid': valid,
            'validation_message': message,
            'supported_video_formats': ['.mp4', '.avi', '.mov', '.mkv'],
            'capabilities': {
                'video_processing': True,
                'camera_processing': True,
                'real_time_detection': valid,
                'frame_annotation': True,
                'background_subtraction': True
            }
        }