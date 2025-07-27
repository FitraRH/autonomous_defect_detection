# controllers/analysis_controller.py
"""
Analysis Controller for JSON API
Handles analysis history, statistics, and configuration endpoints
"""

from flask import jsonify
import json
from datetime import datetime


class AnalysisController:
    """
    Controller for analysis-related API endpoints
    Manages analysis history, statistics, and system configuration
    """
    
    def __init__(self, database_service):
        self.database_service = database_service
    
    def get_analysis_history(self, request):
        """Get paginated analysis history"""
        try:
            # Extract pagination parameters
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            
            # Extract filter parameters
            filters = {
                'decision': request.args.get('decision'),
                'date_from': request.args.get('date_from'),
                'date_to': request.args.get('date_to'),
                'defect_type': request.args.get('defect_type')
            }
            
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # Get history from database service
            history_data = self.database_service.get_analysis_history(
                page=page,
                per_page=per_page,
                filters=filters
            )
            
            return jsonify({
                'status': 'success',
                'data': {
                    'analyses': history_data['analyses'],
                    'pagination': history_data['pagination'],
                    'filters_applied': filters
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_analysis_details(self, analysis_id):
        """Get detailed information for specific analysis"""
        try:
            analysis = self.database_service.get_analysis_by_id(analysis_id)
            
            if not analysis:
                return jsonify({
                    'status': 'error',
                    'error': 'Analysis not found',
                    'timestamp': datetime.now().isoformat()
                }), 404
            
            # Get defect statistics if available
            defect_stats = self.database_service.get_defect_statistics(analysis_id)
            
            # Format detailed response
            detailed_analysis = self._format_detailed_analysis(analysis, defect_stats)
            
            return jsonify({
                'status': 'success',
                'data': detailed_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def delete_analysis(self, analysis_id):
        """Delete specific analysis record"""
        try:
            success = self.database_service.delete_analysis(analysis_id)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'data': {'message': f'Analysis {analysis_id} deleted successfully'},
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Analysis not found or could not be deleted',
                    'timestamp': datetime.now().isoformat()
                }), 404
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_analysis_statistics(self):
        """Get comprehensive analysis statistics"""
        try:
            # Get statistics from database service
            stats = self.database_service.get_comprehensive_statistics()
            
            return jsonify({
                'status': 'success',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def export_analysis_data(self, request):
        """Export analysis data in specified format"""
        try:
            export_params = request.json
            
            if not export_params:
                return jsonify({
                    'status': 'error',
                    'error': 'Export parameters required',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate export parameters
            export_format = export_params.get('format', 'json')
            date_range = export_params.get('date_range', {})
            include_details = export_params.get('include_details', False)
            
            if export_format not in ['json', 'csv', 'xlsx']:
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid export format. Supported: json, csv, xlsx',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Generate export
            export_result = self.database_service.export_analysis_data(
                format=export_format,
                date_range=date_range,
                include_details=include_details
            )
            
            return jsonify({
                'status': 'success',
                'data': {
                    'export_file': export_result['file_path'],
                    'export_format': export_format,
                    'records_exported': export_result['record_count'],
                    'file_size': export_result['file_size'],
                    'generated_at': export_result['generated_at']
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_settings(self):
        """Get current system settings"""
        try:
            settings = self.database_service.get_system_settings()
            
            return jsonify({
                'status': 'success',
                'data': settings,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def update_settings(self, request):
        """Update system settings"""
        try:
            new_settings = request.json
            
            if not new_settings:
                return jsonify({
                    'status': 'error',
                    'error': 'Settings data required',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate settings
            validation_result = self._validate_settings(new_settings)
            if validation_result['error']:
                return jsonify({
                    'status': 'error',
                    'error': validation_result['error'],
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Update settings
            success = self.database_service.update_system_settings(new_settings)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'data': {'message': 'Settings updated successfully'},
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to update settings',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def _format_detailed_analysis(self, analysis, defect_stats):
        """Format detailed analysis response"""
        try:
            # Parse detected defects if stored as JSON string
            detected_defects = analysis.get('detected_defects', '[]')
            if isinstance(detected_defects, str):
                detected_defects = json.loads(detected_defects)
            
            detailed = {
                'analysis_info': {
                    'id': analysis['id'],
                    'image_name': analysis['image_name'],
                    'image_path': analysis.get('image_path'),
                    'original_size': analysis.get('original_size'),
                    'analysis_date': analysis['analysis_date'],
                    'status': analysis.get('status', 'completed')
                },
                'detection_results': {
                    'final_decision': analysis['final_decision'],
                    'anomaly_score': analysis.get('anomaly_score', 0),
                    'confidence_level': analysis.get('confidence_level'),
                    'processing_time': analysis.get('processing_time', 0),
                    'detected_defects': detected_defects,
                    'defect_count': len(detected_defects) if detected_defects else 0
                },
                'defect_statistics': [],
                'quality_metrics': self._calculate_quality_metrics(analysis),
                'visualization_info': {
                    'visualization_path': analysis.get('visualization_path'),
                    'has_visualization': bool(analysis.get('visualization_path'))
                }
            }
            
            # Add defect statistics if available
            if defect_stats:
                for stat in defect_stats:
                    detailed['defect_statistics'].append({
                        'defect_type': stat['defect_type'],
                        'confidence': stat['confidence'],
                        'area_percentage': stat.get('area_percentage', 0),
                        'bbox_count': stat.get('bbox_count', 0),
                        'severity_level': stat.get('severity_level'),
                        'location_data': json.loads(stat.get('location_data', '[]')) if stat.get('location_data') else []
                    })
            
            return detailed
            
        except Exception as e:
            print(f"Error formatting detailed analysis: {e}")
            # Return basic format on error
            return {
                'analysis_info': {'id': analysis.get('id'), 'error': 'Formatting error'},
                'detection_results': analysis,
                'defect_statistics': [],
                'quality_metrics': {},
                'visualization_info': {}
            }
    
    def _calculate_quality_metrics(self, analysis):
        """Calculate quality metrics for analysis"""
        try:
            anomaly_score = analysis.get('anomaly_score', 0)
            final_decision = analysis.get('final_decision', 'UNKNOWN')
            processing_time = analysis.get('processing_time', 0)
            
            # Calculate accuracy indicator
            if final_decision == 'GOOD':
                accuracy_indicator = (1 - anomaly_score) * 100
            elif final_decision == 'DEFECT':
                accuracy_indicator = anomaly_score * 100
            else:
                accuracy_indicator = 50
            
            # Calculate processing efficiency
            target_time = 1.0  # 1 second target
            efficiency = min(100, (target_time / processing_time) * 100) if processing_time > 0 else 100
            
            # Calculate confidence score
            confidence_mapping = {
                'Very High': 95,
                'High': 85,
                'Medium': 70,
                'Low': 50
            }
            confidence_level = analysis.get('confidence_level', 'Medium')
            confidence_score = confidence_mapping.get(confidence_level, 70)
            
            return {
                'accuracy_indicator': round(accuracy_indicator, 2),
                'processing_efficiency': round(efficiency, 2),
                'confidence_score': confidence_score,
                'overall_quality': round((accuracy_indicator + efficiency + confidence_score) / 3, 2),
                'performance_rating': self._get_performance_rating(accuracy_indicator, efficiency, confidence_score)
            }
            
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return {
                'accuracy_indicator': 0,
                'processing_efficiency': 0,
                'confidence_score': 0,
                'overall_quality': 0,
                'performance_rating': 'unknown'
            }
    
    def _get_performance_rating(self, accuracy, efficiency, confidence):
        """Get performance rating based on metrics"""
        average_score = (accuracy + efficiency + confidence) / 3
        
        if average_score >= 90:
            return 'excellent'
        elif average_score >= 80:
            return 'good'
        elif average_score >= 70:
            return 'satisfactory'
        elif average_score >= 60:
            return 'needs_improvement'
        else:
            return 'poor'
    
    def _validate_settings(self, settings):
        """Validate settings data"""
        valid_settings = [
            'anomaly_threshold',
            'defect_threshold', 
            'notification_enabled',
            'auto_save_results',
            'preferred_format',
            'max_batch_size',
            'processing_timeout',
            'quality_threshold'
        ]
        
        for key, value in settings.items():
            if key not in valid_settings:
                return {'error': f'Invalid setting: {key}'}
            
            # Validate specific setting types
            if key in ['anomaly_threshold', 'defect_threshold', 'quality_threshold']:
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    return {'error': f'{key} must be a number between 0 and 1'}
            
            elif key in ['notification_enabled', 'auto_save_results']:
                if not isinstance(value, bool):
                    return {'error': f'{key} must be a boolean value'}
            
            elif key == 'preferred_format':
                if value not in ['JSON', 'XML', 'CSV']:
                    return {'error': f'preferred_format must be JSON, XML, or CSV'}
            
            elif key in ['max_batch_size', 'processing_timeout']:
                if not isinstance(value, int) or value <= 0:
                    return {'error': f'{key} must be a positive integer'}
        
        return {'error': None}