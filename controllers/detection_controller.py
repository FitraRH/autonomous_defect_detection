"""
Detection Controller for JSON API
Handles detection-related requests and responses
Business logic delegated to services
"""

from flask import jsonify
import json
import base64
import os
import time
from datetime import datetime


class DetectionController:
    """
    Controller for detection-related API endpoints
    Handles request processing and response formatting
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
        """Process single image for defect detection"""
        try:
            # Validate request
            validation_result = self._validate_image_request(request)
            if validation_result['error']:
                return jsonify(validation_result), 400
            
            # Extract image data
            image_data = self._extract_image_data(request)
            if not image_data:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to extract image data',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process image
            result = self.detection_service.process_single_image(
                image_data['data'],
                image_data['filename']
            )
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Image processing failed',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            # Save to database
            analysis_id = self.database_service.save_analysis(result)
            
            # Format response
            response = self._format_detection_response(result, analysis_id)
            
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
    
    def process_batch(self, request):
        """Process batch of images"""
        try:
            # Validate batch request
            validation_result = self._validate_batch_request(request)
            if validation_result['error']:
                return jsonify(validation_result), 400
            
            # Extract batch data
            batch_data = self._extract_batch_data(request)
            if not batch_data:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to extract batch data',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process batch
            results = self.detection_service.process_image_batch(batch_data)
            
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
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_video(self, request):
        """Process video for defect detection"""
        try:
            # Validate video request
            validation_result = self._validate_video_request(request)
            if validation_result['error']:
                return jsonify(validation_result), 400
            
            # Extract video data
            video_data = self._extract_video_data(request)
            if not video_data:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to extract video data',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process video
            result = self.detection_service.process_video(video_data)
            
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
                'error': str(e),
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
        """Process single frame in real-time session"""
        try:
            if not self.realtime_active:
                return jsonify({
                    'status': 'error',
                    'error': 'No active real-time session',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Extract frame data
            frame_data = self._extract_frame_data(request)
            if not frame_data:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to extract frame data',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Process frame
            result = self.detection_service.process_realtime_frame(
                self.realtime_session_id,
                frame_data
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
        """Update detection thresholds"""
        try:
            new_thresholds = request.json
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
    
    def _validate_image_request(self, request):
        """Validate image detection request"""
        if not request:
            return {'error': 'No request data'}
        
        # Check for image data in files or JSON
        has_file = 'image' in request.files and request.files['image']
        has_json = request.json and 'image_base64' in request.json
        
        if not has_file and not has_json:
            return {'error': 'No image data provided'}
        
        return {'error': None}
    
    def _validate_batch_request(self, request):
        """Validate batch processing request"""
        if not request or not request.json:
            return {'error': 'No request data'}
        
        if 'images' not in request.json:
            return {'error': 'No images array provided'}
        
        if not isinstance(request.json['images'], list):
            return {'error': 'Images must be an array'}
        
        if len(request.json['images']) == 0:
            return {'error': 'Images array is empty'}
        
        return {'error': None}
    
    def _validate_video_request(self, request):
        """Validate video processing request"""
        if not request:
            return {'error': 'No request data'}
        
        has_file = 'video' in request.files and request.files['video']
        has_json = request.json and 'video_base64' in request.json
        
        if not has_file and not has_json:
            return {'error': 'No video data provided'}
        
        return {'error': None}
    
    def _extract_image_data(self, request):
        """Extract image data from request"""
        try:
            if 'image' in request.files:
                file = request.files['image']
                return {
                    'data': file.read(),
                    'filename': file.filename or f"upload_{int(time.time())}.jpg"
                }
            elif request.json and 'image_base64' in request.json:
                base64_data = request.json['image_base64']
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                
                return {
                    'data': base64.b64decode(base64_data),
                    'filename': request.json.get('filename', f"upload_{int(time.time())}.jpg")
                }
        except Exception as e:
            print(f"Error extracting image data: {e}")
            return None
    
    def _extract_batch_data(self, request):
        """Extract batch data from request"""
        try:
            batch_images = []
            images_data = request.json.get('images', [])
            
            for i, image_item in enumerate(images_data):
                if 'image_base64' in image_item:
                    base64_data = image_item['image_base64']
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',')[1]
                    
                    batch_images.append({
                        'data': base64.b64decode(base64_data),
                        'filename': image_item.get('filename', f"batch_image_{i+1}.jpg")
                    })
            
            return batch_images
            
        except Exception as e:
            print(f"Error extracting batch data: {e}")
            return None
    
    def _extract_video_data(self, request):
        """Extract video data from request"""
        try:
            if 'video' in request.files:
                file = request.files['video']
                return {
                    'data': file.read(),
                    'filename': file.filename or f"video_{int(time.time())}.mp4"
                }
            elif request.json and 'video_base64' in request.json:
                base64_data = request.json['video_base64']
                return {
                    'data': base64.b64decode(base64_data),
                    'filename': request.json.get('filename', f"video_{int(time.time())}.mp4")
                }
        except Exception as e:
            print(f"Error extracting video data: {e}")
            return None
    
    def _extract_frame_data(self, request):
        """Extract frame data from real-time request"""
        try:
            if request.json and 'frame_base64' in request.json:
                base64_data = request.json['frame_base64']
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]
                
                return {
                    'data': base64.b64decode(base64_data),
                    'timestamp': request.json.get('timestamp', time.time())
                }
        except Exception as e:
            print(f"Error extracting frame data: {e}")
            return None
    
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