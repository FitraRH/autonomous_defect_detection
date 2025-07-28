# api_server.py - Clean API endpoints only
"""
Clean JSON API Server for Defect Detection System
Contains only endpoint definitions and routing
All business logic moved to services layer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime

# Import controllers (business logic handlers)
from controllers.detection_controller import DetectionController
from controllers.analysis_controller import AnalysisController
from controllers.performance_controller import PerformanceController

# Import services for initialization
from services.detection_service import DetectionService
from services.database_service import DatabaseService

class JSONAPIServer:
    """
    JSON API Server
    Contains only endpoint definitions and request routing
    All logic delegated to controllers and services
    """
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        
        # Initialize services
        self.database_service = DatabaseService()
        self.detection_service = DetectionService()
        
        # Initialize controllers
        self.detection_controller = DetectionController(self.detection_service, self.database_service)
        self.analysis_controller = AnalysisController(self.database_service)
        self.performance_controller = PerformanceController(self.database_service)
        
        # Setup routes
        self._setup_routes()
        
        print("JSON API Server initialized")
    
    def _setup_routes(self):
        """Setup all API endpoints"""
        
        # Health and system endpoints
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.detection_controller.health_check()
        
        @self.app.route('/api/system/info', methods=['GET'])
        def system_info():
            return self.detection_controller.get_system_info()
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            return self.detection_controller.get_system_status()
        
        # Detection endpoints
        @self.app.route('/api/detection/image', methods=['POST'])
        def detect_image():
            return self.detection_controller.process_image(request)
        
        @self.app.route('/api/detection/batch', methods=['POST'])
        def detect_batch():
            return self.detection_controller.process_batch(request)
        
        @self.app.route('/api/detection/video', methods=['POST'])
        def detect_video():
            return self.detection_controller.process_video(request)
        
        # Analysis endpoints
        @self.app.route('/api/analysis/history', methods=['GET'])
        def analysis_history():
            return self.analysis_controller.get_analysis_history(request)
        
        @self.app.route('/api/analysis/<int:analysis_id>', methods=['GET'])
        def get_analysis(analysis_id):
            return self.analysis_controller.get_analysis_details(analysis_id)
        
        @self.app.route('/api/analysis/<int:analysis_id>', methods=['DELETE'])
        def delete_analysis(analysis_id):
            return self.analysis_controller.delete_analysis(analysis_id)
        
        @self.app.route('/api/analysis/stats', methods=['GET'])
        def analysis_stats():
            return self.analysis_controller.get_analysis_statistics()
        
        @self.app.route('/api/analysis/export', methods=['POST'])
        def export_analysis():
            return self.analysis_controller.export_analysis_data(request)
        
        # Performance endpoints
        @self.app.route('/api/performance/metrics', methods=['GET'])
        def performance_metrics():
            return self.performance_controller.get_performance_metrics()
        
        @self.app.route('/api/performance/trends', methods=['GET'])
        def performance_trends():
            return self.performance_controller.get_performance_trends(request)
        
        @self.app.route('/api/performance/charts', methods=['GET'])
        def performance_charts():
            return self.performance_controller.get_performance_charts()
        
        # Real-time processing endpoints
        @self.app.route('/api/realtime/session/start', methods=['POST'])
        def start_realtime_session():
            return self.detection_controller.start_realtime_session()
        
        @self.app.route('/api/realtime/session/stop', methods=['POST'])
        def stop_realtime_session():
            return self.detection_controller.stop_realtime_session()
        
        @self.app.route('/api/realtime/frame/process', methods=['POST'])
        def process_realtime_frame():
            return self.detection_controller.process_realtime_frame(request)
        
        @self.app.route('/api/realtime/session/status', methods=['GET'])
        def realtime_session_status():
            return self.detection_controller.get_realtime_session_status()
        
        # Configuration endpoints
        @self.app.route('/api/config/settings', methods=['GET'])
        def get_settings():
            return self.analysis_controller.get_settings()
        
        @self.app.route('/api/config/settings', methods=['PUT'])
        def update_settings():
            return self.analysis_controller.update_settings(request)
        
        @self.app.route('/api/config/thresholds', methods=['GET'])
        def get_thresholds():
            return self.detection_controller.get_detection_thresholds()
        
        @self.app.route('/api/config/thresholds', methods=['PUT'])
        def update_thresholds():
            return self.detection_controller.update_detection_thresholds(request)
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An internal error occurred while processing the request',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad request',
                'message': 'Invalid request data or parameters',
                'timestamp': datetime.now().isoformat()
            }), 400
    
    def run(self, debug=False):
        """Start the JSON API server"""
        print("Starting JSON API Server")
        print("=" * 50)
        print(f"Server URL: http://{self.host}:{self.port}")
        print("API Documentation:")
        print(f"  Health Check: GET /api/health")
        print(f"  System Info:  GET /api/system/info")
        print(f"  Detect Image: POST /api/detection/image")
        print(f"  Batch Detect: POST /api/detection/batch")
        print(f"  Analysis History: GET /api/analysis/history")
        print(f"  Performance Metrics: GET /api/performance/metrics")
        print("=" * 50)
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)


def create_api_server(host='0.0.0.0', port=5000):
    """Factory function to create API server instance"""
    return JSONAPIServer(host=host, port=port)


if __name__ == "__main__":
    # Create and start the API server
    api_server = create_api_server()
    api_server.run(debug=True)