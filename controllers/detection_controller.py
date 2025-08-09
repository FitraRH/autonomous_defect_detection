# controllers/detection_controller.py - COMPLETELY FIXED VERSION
"""
Detection Controller for JSON API - COMPLETELY FIXED
Handles detection-related requests and responses
FIXED: All form data and JSON handling issues
"""

from flask import jsonify
import json
import base64
import os
import time
from datetime import datetime


class DetectionController:
    """
    Controller for detection-related API endpoints - COMPLETELY FIXED
    Handles request processing and response formatting
    FIXED: Proper form data and JSON handling with correct Content-Type detection
    """
    
    def __init__(self, detection_service, database_service):
        self.detection_service = detection_service
        self.database_service = database_service
        self.realtime_active = False
        self.realtime_session_id = None
    
    def health_check(self):
        """Health check endpoint"""
        try:
            status = self.detection_service.get_health_status()
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'services': status,
                'api_version': '1.0.0'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_info(self):
        """Get detailed system information"""
        try:
            info = self.detection_service.get_system_information()
            return jsonify({
                'status': 'success',
                'data': info,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_status(self):
        """Get current system status"""
        try:
            status = self.detection_service.get_current_status()
            return jsonify({
                'status': 'success',
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_image(self, request):
        """Process single image for defect detection - COMPLETELY FIXED VERSION"""
        try:
            print(f"DEBUG: Request method: {request.method}")
            print(f"DEBUG: Content type: {request.content_type}")
            print(f"DEBUG: Has files: {'image' in request.files if request.files else False}")
            print(f"DEBUG: Has form: {bool(request.form)}")
            print(f"DEBUG: Has JSON: {bool(request.get_json(silent=True))}")
            print(f"DEBUG: Request data length: {len(request.data) if request.data else 0}")
            
            # COMPLETELY FIXED: Handle both form data and JSON properly
            image_data = None
            filename = None
            
            # Method 1: Try multipart/form-data first (file upload)
            if request.files and 'image' in request.files:
                print("DEBUG: Processing as multipart form data")
                file = request.files['image']
                if file.filename == '':
                    return jsonify({
                        'status': 'error',
                        'error': 'No file selected',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                image_data = file.read()
                filename = file.filename or f"upload_{int(time.time())}.jpg"
                print(f"DEBUG: Got file data - {len(image_data)} bytes, filename: {filename}")
                
            # Method 2: Try JSON with base64 (application/json)
            elif request.content_type and 'application/json' in request.content_type:
                print("DEBUG: Processing as JSON request")
                json_data = request.get_json(silent=True)
                if json_data and 'image_base64' in json_data:
                    base64_data = json_data['image_base64']
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',')[1]
                    
                    try:
                        image_data = base64.b64decode(base64_data)
                        filename = json_data.get('filename', f"upload_{int(time.time())}.jpg")
                        print(f"DEBUG: Decoded base64 data - {len(image_data)} bytes")
                    except Exception as decode_error:
                        return jsonify({
                            'status': 'error',
                            'error': f'Invalid base64 image data: {str(decode_error)}',
                            'timestamp': datetime.now().isoformat()
                        }), 400
                else:
                    return jsonify({
                        'status': 'error',
                        'error': 'JSON request must contain "image_base64" field',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            # Method 3: Try to parse JSON from any other content type 
            else:
                print("DEBUG: Trying to parse as JSON fallback")
                try:
                    json_data = request.get_json(force=True, silent=True)  # Force JSON parsing
                    if json_data and 'image_base64' in json_data:
                        base64_data = json_data['image_base64']
                        if base64_data.startswith('data:image'):
                            base64_data = base64_data.split(',')[1]
                        
                        image_data = base64.b64decode(base64_data)
                        filename = json_data.get('filename', f"upload_{int(time.time())}.jpg")
                        print(f"DEBUG: Fallback JSON parsing successful - {len(image_data)} bytes")
                except Exception as json_error:
                    print(f"DEBUG: JSON fallback failed: {json_error}")
            
            # If still no image data found
            if not image_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No valid image data found. Send either: 1) multipart/form-data with "image" field, or 2) JSON with "image_base64" field',
                    'debug_info': {
                        'content_type': request.content_type,
                        'method': request.method,
                        'has_files': bool(request.files),
                        'files_keys': list(request.files.keys()) if request.files else [],
                        'has_form': bool(request.form),
                        'form_keys': list(request.form.keys()) if request.form else [],
                        'has_json': bool(request.get_json(silent=True)),
                        'data_length': len(request.data) if request.data else 0
                    },
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate file size (10MB limit)
            if len(image_data) > 10 * 1024 * 1024:
                return jsonify({
                    'status': 'error',
                    'error': 'File too large. Maximum size is 10MB',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate image data (basic check)
            if len(image_data) < 100:  # Too small to be a real image
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid image data - file too small',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            print(f"DEBUG: Successfully got image data - filename: {filename}, size: {len(image_data)} bytes")
            
            # Process image using detection service
            try:
                result = self.detection_service.process_single_image(image_data, filename)
            except Exception as processing_error:
                print(f"ERROR in detection service: {str(processing_error)}")
                return jsonify({
                    'status': 'error',
                    'error': f'Image processing failed: {str(processing_error)}',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Image processing returned no result',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            # Save to database
            try:
                analysis_id = self.database_service.save_analysis(result)
            except Exception as db_error:
                print(f"WARNING: Database save failed: {str(db_error)}")
                analysis_id = None  # Continue without database save
            
            # Format response
            response = self._format_detection_response(result, analysis_id)
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"CRITICAL ERROR in process_image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'status': 'error',
                'error': f'Critical processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_batch(self, request):
        """Process batch of images - FIXED VERSION"""
        try:
            # Handle JSON batch request
            json_data = None
            
            # Try to get JSON data with multiple methods
            if request.content_type and 'application/json' in request.content_type:
                json_data = request.get_json(silent=True)
            else:
                # Force JSON parsing for other content types
                json_data = request.get_json(force=True, silent=True)
            
            if not json_data or 'images' not in json_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No images array provided in request body. Expected JSON with "images" array.',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            images_data = json_data['images']
            if not isinstance(images_data, list) or len(images_data) == 0:
                return jsonify({
                    'status': 'error',
                    'error': 'Images must be a non-empty array',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Extract batch data
            batch_images = []
            for i, image_item in enumerate(images_data):
                if not isinstance(image_item, dict) or 'image_base64' not in image_item:
                    continue
                
                base64_data = image_item['image_base64']
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                
                try:
                    image_data = base64.b64decode(base64_data)
                    batch_images.append({
                        'data': image_data,
                        'filename': image_item.get('filename', f"batch_image_{i+1}.jpg")
                    })
                except Exception as decode_error:
                    print(f"Error decoding image {i+1}: {decode_error}")
                    continue
            
            if not batch_images:
                return jsonify({
                    'status': 'error',
                    'error': 'No valid images found in batch',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process batch
            results = self.detection_service.process_image_batch(batch_images)
            
            # Save batch results
            batch_id = self.database_service.save_batch_results(results)
            
            # Format response
            response = self._format_batch_response(results, batch_id)
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"ERROR in process_batch: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': f'Batch processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_video(self, request):
        """Process video for defect detection - FIXED VERSION"""
        try:
            # Handle video file upload or base64
            video_data = None
            filename = None
            
            if request.files and 'video' in request.files:
                file = request.files['video']
                if file.filename == '':
                    return jsonify({
                        'status': 'error',
                        'error': 'No video file selected',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                video_data = file.read()
                filename = file.filename or f"video_{int(time.time())}.mp4"
                
            elif request.get_json(silent=True) or request.get_json(force=True, silent=True):
                json_data = request.get_json(silent=True) or request.get_json(force=True, silent=True)
                if json_data and 'video_base64' in json_data:
                    base64_data = json_data['video_base64']
                    try:
                        video_data = base64.b64decode(base64_data)
                        filename = json_data.get('filename', f"video_{int(time.time())}.mp4")
                    except Exception as decode_error:
                        return jsonify({
                            'status': 'error',
                            'error': f'Invalid base64 video data: {str(decode_error)}',
                            'timestamp': datetime.now().isoformat()
                        }), 400
            
            if not video_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No video provided. Send either file upload or JSON with "video_base64"',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process video
            result = self.detection_service.process_video({'data': video_data, 'filename': filename})
            
            # Save video analysis
            video_id = self.database_service.save_video_analysis(result)
            
            # Format response
            response = self._format_video_response(result, video_id)
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': f'Video processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def start_realtime_session(self):
        """Start real-time detection session"""
        try:
            if self.realtime_active:
                return jsonify({
                    'status': 'error',
                    'error': 'Real-time session already active',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            session_id = self.detection_service.start_realtime_session()
            
            if session_id:
                self.realtime_active = True
                self.realtime_session_id = session_id
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'session_id': session_id,
                        'started_at': datetime.now().isoformat(),
                        'message': 'Real-time session started successfully'
                    },
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to start real-time session',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def stop_realtime_session(self):
        """Stop real-time detection session"""
        try:
            if not self.realtime_active:
                return jsonify({
                    'status': 'error',
                    'error': 'No active real-time session',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            session_report = self.detection_service.stop_realtime_session(self.realtime_session_id)
            
            self.realtime_active = False
            self.realtime_session_id = None
            
            return jsonify({
                'status': 'success',
                'data': {
                    'session_report': session_report,
                    'stopped_at': datetime.now().isoformat(),
                    'message': 'Real-time session stopped successfully'
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_realtime_frame(self, request):
        """Process single frame in real-time session - FIXED VERSION"""
        try:
            if not self.realtime_active:
                return jsonify({
                    'status': 'error',
                    'error': 'No active real-time session',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Extract frame data with flexible JSON parsing
            json_data = request.get_json(silent=True) or request.get_json(force=True, silent=True)
            
            if not json_data or 'frame_base64' not in json_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No frame_base64 data provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            base64_data = json_data['frame_base64']
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            try:
                frame_data = base64.b64decode(base64_data)
            except Exception as decode_error:
                return jsonify({
                    'status': 'error',
                    'error': f'Invalid base64 frame data: {str(decode_error)}',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            frame_info = {
                'data': frame_data,
                'timestamp': json_data.get('timestamp', time.time())
            }
            
            # Process frame
            result = self.detection_service.process_realtime_frame(
                self.realtime_session_id,
                frame_info
            )
            
            # Format real-time response
            response = self._format_realtime_response(result)
            
            return jsonify({
                'status': 'success',
                'data': response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_realtime_session_status(self):
        """Get current real-time session status"""
        try:
            if not self.realtime_active:
                return jsonify({
                    'status': 'success',
                    'data': {
                        'session_active': False,
                        'message': 'No active real-time session'
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            session_stats = self.detection_service.get_realtime_session_stats(self.realtime_session_id)
            
            return jsonify({
                'status': 'success',
                'data': {
                    'session_active': True,
                    'session_id': self.realtime_session_id,
                    'session_stats': session_stats
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_detection_thresholds(self):
        """Get current detection thresholds"""
        try:
            thresholds = self.detection_service.get_thresholds()
            return jsonify({
                'status': 'success',
                'data': thresholds,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def update_detection_thresholds(self, request):
        """Update detection thresholds - FIXED VERSION"""
        try:
            # Get JSON data with flexible parsing
            new_thresholds = request.get_json(silent=True) or request.get_json(force=True, silent=True)
            
            if not new_thresholds:
                return jsonify({
                    'status': 'error',
                    'error': 'No threshold data provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            success = self.detection_service.update_thresholds(new_thresholds)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'data': {'message': 'Thresholds updated successfully'},
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to update thresholds',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def _format_detection_response(self, result, analysis_id):
        """Format single image detection response"""
        return {
            'analysis_id': analysis_id,
            'final_decision': result.get('final_decision'),
            'processing_time': result.get('processing_time'),
            'anomaly_detection': {
                'anomaly_score': result.get('anomaly_detection', {}).get('anomaly_score'),
                'decision': result.get('anomaly_detection', {}).get('decision'),
                'threshold_used': result.get('anomaly_detection', {}).get('threshold_used')
            },
            'detected_defects': result.get('detected_defect_types', []),
            'defect_count': len(result.get('detected_defect_types', [])),
            'confidence_level': self._calculate_confidence_level(result),
            'result_summary': {
                'is_defective': result.get('final_decision') == 'DEFECT',
                'defect_types_found': result.get('detected_defect_types', []),
                'processing_status': 'completed'
            }
        }
    
    def _format_batch_response(self, results, batch_id):
        """Format batch processing response"""
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)
        
        defective_count = sum(1 for r in successful_results if r.get('final_decision') == 'DEFECT')
        good_count = sum(1 for r in successful_results if r.get('final_decision') == 'GOOD')
        
        return {
            'batch_id': batch_id,
            'summary': {
                'total_images': len(results),
                'successful_processing': len(successful_results),
                'failed_processing': failed_count,
                'defective_products': defective_count,
                'good_products': good_count,
                'defect_rate': (defective_count / len(successful_results) * 100) if successful_results else 0
            },
            'results': [self._format_detection_response(r, None) for r in successful_results],
            'processing_stats': {
                'avg_processing_time': sum(r.get('processing_time', 0) for r in successful_results) / len(successful_results) if successful_results else 0,
                'total_processing_time': sum(r.get('processing_time', 0) for r in successful_results)
            }
        }
    
    def _format_video_response(self, result, video_id):
        """Format video processing response"""
        return {
            'video_id': video_id,
            'processing_summary': result.get('summary', {}),
            'frame_analysis': result.get('frame_results', []),
            'defect_timeline': result.get('defect_timeline', []),
            'video_stats': result.get('video_stats', {})
        }
    
    def _format_realtime_response(self, result):
        """Format real-time frame processing response"""
        return {
            'frame_id': result.get('frame_id'),
            'detection_result': {
                'final_decision': result.get('final_decision'),
                'anomaly_score': result.get('anomaly_detection', {}).get('anomaly_score'),
                'detected_defects': result.get('detected_defect_types', []),
                'processing_time': result.get('processing_time')
            },
            'session_stats': result.get('session_stats', {}),
            'frame_timestamp': result.get('timestamp')
        }
    
    def _calculate_confidence_level(self, result):
        """Calculate confidence level from detection result"""
        score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
        decision = result.get('final_decision', 'UNKNOWN')
        
        if decision == 'GOOD':
            if score < 0.2:
                return "very_high"
            elif score < 0.4:
                return "high"
            elif score < 0.6:
                return "medium"
            else:
                return "low"
        else:  # DEFECT
            if score > 0.9:
                return "very_high"
            elif score > 0.8:
                return "high"
            elif score > 0.7:
                return "medium"
            else:
                return "low"