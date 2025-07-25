"""
Enhanced Flask API Server for Unified Defect Detection System
Features:
- RESTful API endpoints for Flutter and web integration
- Database integration for storing analysis history
- Dashboard statistics and analytics
- File upload and image processing
- Real-time data updates
- User management and authentication
- Performance tracking integration
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
import tempfile
import uuid
import shutil
import statistics
import sqlite3
from pathlib import Path
import threading
import time
from collections import defaultdict
import random

# Import performance tracker with proper error handling
try:
    from utils.performance_tracker import EnhancedPerformanceTracker
    performance_tracker = EnhancedPerformanceTracker()  # Create instance
    PERFORMANCE_TRACKER_AVAILABLE = True
    print(" Performance tracker module imported successfully")
except ImportError as e:
    print(f" Performance tracker not available: {e}")
    print("Using basic tracking")
    PERFORMANCE_TRACKER_AVAILABLE = False
    performance_tracker = None

# Import core detection system
try:
    from main import UnifiedDefectDetector, create_detector
    DETECTOR_AVAILABLE = True
    print(" Main detection module imported successfully")
except ImportError as e:
    print(f"Warning: Main detection module not available: {e}")
    print("Using mock responses for development")
    DETECTOR_AVAILABLE = False


class EnhancedDefectDetectionAPI:
    """
    Enhanced API Server with Database Integration and Performance Tracking
    
    This class provides:
    - RESTful API endpoints for defect detection
    - Database management for analysis history
    - Statistics and analytics calculation
    - File upload and management
    - Mock data generation for testing
    - Performance tracking integration
    """
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        self.host = host
        self.port = port
        self.detector = None
        
        # Initialize performance tracker
        if PERFORMANCE_TRACKER_AVAILABLE and performance_tracker:
            self.performance_tracker = performance_tracker
            print(" Performance tracker initialized")
        else:
            self.performance_tracker = None
            print(" Using basic performance tracking")
        
        # Initialize database
        self.init_database()
        
        # Initialize detector
        self._initialize_detector()
        
        # Initialize real-time processor
        self._initialize_realtime_processor()
        
        # Setup routes
        self._setup_api_routes()
        self._setup_web_routes()
        
        # Setup static file serving
        self._setup_static_files()
        
        # Start background tasks
        self._start_background_tasks()
    
    def init_database(self):
        """
        Initialize SQLite database for storing analysis data
        
        Creates tables for:
        - analyses: Main analysis records
        - defect_statistics: Detailed defect information
        - system_stats: Dashboard statistics
        - user_settings: User preferences and configuration
        """
        self.db_path = "defect_detection.db"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                image_path TEXT,
                original_size TEXT,
                final_decision TEXT NOT NULL,
                anomaly_score REAL,
                confidence_level TEXT,
                detected_defects TEXT,
                processing_time REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Completed',
                user_id TEXT DEFAULT 'default',
                notes TEXT,
                visualization_path TEXT
            )
        ''')
        
        # Detailed defect statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defect_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                defect_type TEXT,
                confidence REAL,
                area_percentage REAL,
                bbox_count INTEGER,
                severity_level TEXT,
                location_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # System statistics for dashboard
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_date DATE,
                total_analyses INTEGER DEFAULT 0,
                defects_detected INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                avg_processing_time REAL DEFAULT 0.0,
                peak_usage_hour INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User settings and preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                anomaly_threshold REAL DEFAULT 0.7,
                defect_threshold REAL DEFAULT 0.85,
                notification_enabled BOOLEAN DEFAULT 1,
                auto_save_results BOOLEAN DEFAULT 1,
                preferred_format TEXT DEFAULT 'JSON',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                processing_time REAL,
                decision TEXT,
                anomaly_score REAL,
                memory_usage REAL,
                cpu_usage REAL,
                image_size TEXT,
                confidence_score REAL,
                defect_types TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # System performance history for enhanced tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                memory_total REAL,
                memory_used REAL,
                memory_percent REAL,
                cpu_percent REAL,
                disk_usage REAL,
                active_processes INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(" Database initialized successfully")
    
    def _initialize_detector(self):
        """
        Initialize the defect detection system
        
        Attempts to load the main detector, falls back to mock mode if unavailable
        """
        if DETECTOR_AVAILABLE:
            try:
                self.detector = create_detector()
                if self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready():
                    print(" Detection system ready for production")
                    self.detector_status = "ready"
                else:
                    print("  Detection system initialized but models not loaded")
                    self.detector_status = "models_not_loaded"
            except Exception as e:
                print(f" Failed to initialize detector: {e}")
                self.detector = None
                self.detector_status = "failed"
        else:
            print("  Using mock detector for development/testing")
            self.detector = None
            self.detector_status = "mock"
            
    def _initialize_realtime_processor(self):
        """
        Initialize real-time processor for live camera detection
        """
        try:
            # Try to import real-time processor if available
            from processors.realtime_processor import RealTimeProcessor
            
            if self.detector:
                self.realtime_processor = RealTimeProcessor(self.detector, self.performance_tracker)
                print(" Real-time processor initialized")
            else:
                # Mock real-time processor for development
                self.realtime_processor = MockRealtimeProcessor()
                print(" Using mock real-time processor (no detector)")
                
        except ImportError:
            print(" Real-time processor module not available, using mock processor")
            self.realtime_processor = MockRealtimeProcessor()
        except Exception as e:
            print(f" Failed to initialize real-time processor: {e}")
            self.realtime_processor = MockRealtimeProcessor()
    
    def _setup_static_files(self):
        """
        Setup static file serving for uploads and results
        """
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            return send_from_directory('static', filename)
        
        # Create necessary directories
        directories = [
            'static',
            'static/uploads',
            'static/results', 
            'static/visualizations',
            'templates'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _start_background_tasks(self):
        """
        Start background threads for data processing and statistics
        """
        def update_stats_worker():
            """Background worker to update statistics periodically"""
            while True:
                try:
                    self.update_daily_stats()
                    self.cleanup_old_files()
                    time.sleep(3600)  # Update every hour
                except Exception as e:
                    print(f"Error in background worker: {e}")
                    time.sleep(300)  # Retry in 5 minutes
        
        # Start background thread
        stats_thread = threading.Thread(target=update_stats_worker, daemon=True)
        stats_thread.start()
        print(" Background tasks started")
    
    def update_daily_stats(self):
        """
        Update daily statistics in database for dashboard
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Calculate today's statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects,
                AVG(a.processing_time) as avg_time,
                AVG(CASE WHEN final_decision = 'GOOD' THEN 1.0 ELSE 0.0 END) as accuracy,
                strftime('%H', analysis_date) as hour,
                COUNT(strftime('%H', analysis_date)) as hour_count
            FROM analyses 
            WHERE DATE(analysis_date) = ?
            GROUP BY strftime('%H', analysis_date)
            ORDER BY hour_count DESC
            LIMIT 1
        ''', (today,))
        
        peak_hour_data = cursor.fetchone()
        peak_hour = peak_hour_data[4] if peak_hour_data else 0
        
        # Get overall today stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects,
                AVG(a.processing_time) as avg_time,
                AVG(CASE WHEN final_decision = 'GOOD' THEN 1.0 ELSE 0.0 END) as accuracy
            FROM analyses 
            WHERE DATE(analysis_date) = ?
        ''', (today,))
        
        stats = cursor.fetchone()
        
        # Update or insert today's stats
        cursor.execute('''
            INSERT OR REPLACE INTO system_stats 
            (stat_date, total_analyses, defects_detected, accuracy_rate, 
             avg_processing_time, peak_usage_hour, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (today, stats[0], stats[1] or 0, stats[3] or 0.0, stats[2] or 0.0, peak_hour))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_files(self):
        """
        Clean up old uploaded files and visualizations
        """
        try:
            # Remove files older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for directory in ['static/uploads', 'static/results', 'static/visualizations']:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                            if file_time < cutoff_date:
                                os.remove(filepath)
                                
        except Exception as e:
            print(f"Error cleaning up files: {e}")
    
    def save_analysis_to_db(self, result, image_name, image_size=None):
        """
        Save analysis result to database with comprehensive information - RETURNS analysis_id
        
        Args:
            result: Detection result dictionary
            image_name: Original image filename
            image_size: Tuple of (width, height)
            
        Returns:
            analysis_id: Database ID of saved analysis
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            size_str = f"{image_size[0]}x{image_size[1]}" if image_size else "unknown"
            confidence_level = self._calculate_confidence_level(result)
            
            # Insert main analysis record
            cursor.execute('''
                INSERT INTO analyses 
                (image_name, image_path, original_size, final_decision, anomaly_score, 
                 confidence_level, detected_defects, processing_time, status, visualization_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_name,
                result.get('image_path', ''),
                size_str,
                result.get('final_decision', 'Unknown'),
                result.get('anomaly_detection', {}).get('anomaly_score', 0.0),
                confidence_level,
                json.dumps(result.get('detected_defect_types', [])),
                result.get('processing_time', 0.0),
                'Completed',
                result.get('visualization_path', '')
            ))
            
            analysis_id = cursor.lastrowid  # GET THE INSERTED ID
            
            # Insert defect statistics if available
            if result.get('defect_classification') and result['final_decision'] == 'DEFECT':
                self._save_defect_statistics(cursor, analysis_id, result['defect_classification'])
            
            conn.commit()
            conn.close()
            
            return analysis_id  # RETURN THE ID
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            return None
    
    def has_performance_tracker(self):
        """Check if performance tracker is available"""
        return self.performance_tracker is not None
    
    def _calculate_confidence_level(self, result):
        """Calculate human-readable confidence level"""
        score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
        
        if score >= 0.9:
            return "Very High"
        elif score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _save_defect_statistics(self, cursor, analysis_id, defect_classification):
        """Save detailed defect statistics"""
        defect_stats = defect_classification.get('defect_analysis', {})
        
        for defect_type, stats in defect_stats.get('defect_statistics', {}).items():
            # Calculate severity based on area and confidence
            area_pct = defect_stats.get('class_distribution', {}).get(defect_type, {}).get('percentage', 0.0)
            confidence = stats.get('avg_confidence', 0.0)
            severity = self._calculate_severity(area_pct, confidence)
            
            # Get location data
            bboxes = defect_stats.get('bounding_boxes', {}).get(defect_type, [])
            location_data = json.dumps([{
                'x': bbox['x'], 'y': bbox['y'], 
                'width': bbox['width'], 'height': bbox['height']
            } for bbox in bboxes])
            
            cursor.execute('''
                INSERT INTO defect_statistics
                (analysis_id, defect_type, confidence, area_percentage, bbox_count, 
                 severity_level, location_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                defect_type,
                stats.get('avg_confidence', 0.0),
                area_pct,
                stats.get('num_regions', 0),
                severity,
                location_data
            ))
    
    def _calculate_severity(self, area_percentage, confidence):
        """Calculate defect severity level"""
        if area_percentage > 10 and confidence > 0.8:
            return "Critical"
        elif area_percentage > 5 and confidence > 0.7:
            return "High"
        elif area_percentage > 2 and confidence > 0.6:
            return "Medium"
        else:
            return "Low"
    
    def get_dashboard_stats(self):
        """
        Get enhanced dashboard statistics with performance metrics integration
        
        Returns:
            dict: Enhanced dashboard statistics including trends and analytics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent stats (last 7 days)
        cursor.execute('''
            SELECT 
                COUNT(*) as total_analyses,
                SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects_detected,
                AVG(CASE WHEN final_decision = 'GOOD' THEN 1.0 ELSE 0.0 END) * 100 as accuracy,
                COUNT(CASE WHEN DATE(analysis_date) = DATE('now') THEN 1 END) as today_count,
                COUNT(CASE WHEN DATE(analysis_date) = DATE('now', '-1 day') THEN 1 END) as yesterday_count,
                AVG(processing_time) as avg_processing_time,
                MIN(processing_time) as min_processing_time,
                MAX(processing_time) as max_processing_time
            FROM analyses
            WHERE analysis_date >= DATE('now', '-7 days')
        ''')
        
        stats = cursor.fetchone()
        
        # Calculate week-over-week changes
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN DATE(analysis_date) >= DATE('now', '-7 days') THEN 1 END) as this_week,
                COUNT(CASE WHEN DATE(analysis_date) >= DATE('now', '-14 days') 
                            AND DATE(analysis_date) < DATE('now', '-7 days') THEN 1 END) as last_week,
                SUM(CASE WHEN DATE(analysis_date) >= DATE('now', '-7 days') 
                            AND final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects_this_week,
                SUM(CASE WHEN DATE(analysis_date) >= DATE('now', '-14 days') 
                            AND DATE(analysis_date) < DATE('now', '-7 days')
                            AND final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects_last_week
            FROM analyses
        ''')
        
        week_stats = cursor.fetchone()
        
        # Get defect type distribution with enhanced stats
        cursor.execute('''
            SELECT defect_type, COUNT(*) as count, AVG(confidence) as avg_confidence,
                AVG(area_percentage) as avg_area,
                AVG(bbox_count) as avg_regions,
                COUNT(CASE WHEN severity_level = 'Critical' THEN 1 END) as critical_count
            FROM defect_statistics ds
            JOIN analyses a ON ds.analysis_id = a.id
            WHERE a.analysis_date >= DATE('now', '-30 days')
            GROUP BY defect_type
            ORDER BY count DESC
        ''')
        
        defect_types = cursor.fetchall()
        
        # Get performance metrics from performance_tracker if available
        try:
            if self.performance_tracker:
                performance_metrics = self.performance_tracker.get_enhanced_metrics()
                realtime_metrics = self.performance_tracker.get_realtime_metrics()
                
                # Enhanced performance data
                perf_data = {
                    'current_throughput': realtime_metrics.get('current_throughput', 0),
                    'avg_processing_time_5min': realtime_metrics.get('avg_processing_time_5min', stats[5] or 0),
                    'system_memory_usage': realtime_metrics.get('current_memory_usage', 0),
                    'system_cpu_usage': realtime_metrics.get('current_cpu_usage', 0),
                    'processing_trend': realtime_metrics.get('trend_processing_time', 'stable'),
                    'quality_trend': performance_metrics.get('trends', {}).get('quality_trend', 'stable'),
                    'total_samples': performance_metrics.get('basic_stats', {}).get('total_tests', 0),
                    'avg_confidence': performance_metrics.get('quality_metrics', {}).get('avg_confidence', 0.85)
                }
            else:
                # Fallback if performance tracker not available
                perf_data = {
                    'current_throughput': 1 / (stats[5] or 1) if stats[5] else 0,
                    'avg_processing_time_5min': stats[5] or 0,
                    'system_memory_usage': 0,
                    'system_cpu_usage': 0,
                    'processing_trend': 'stable',
                    'quality_trend': 'stable',
                    'total_samples': stats[0] or 0,
                    'avg_confidence': 0.85
                }
        except:
            # Fallback if performance tracker not available
            perf_data = {
                'current_throughput': 1 / (stats[5] or 1) if stats[5] else 0,
                'avg_processing_time_5min': stats[5] or 0,
                'system_memory_usage': 0,
                'system_cpu_usage': 0,
                'processing_trend': 'stable',
                'quality_trend': 'stable',
                'total_samples': stats[0] or 0,
                'avg_confidence': 0.85
            }
        
        # Get hourly analysis pattern for today
        cursor.execute('''
            SELECT strftime('%H', analysis_date) as hour, COUNT(*) as count
            FROM analyses 
            WHERE DATE(analysis_date) = DATE('now')
            GROUP BY strftime('%H', analysis_date)
            ORDER BY hour
        ''')
        hourly_pattern = cursor.fetchall()
        
        # Get system health indicators
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN status = 'Completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'Failed' THEN 1 END) as failed,
                AVG(CASE WHEN final_decision = 'GOOD' THEN anomaly_score ELSE 1-anomaly_score END) as system_accuracy
            FROM analyses
            WHERE analysis_date >= DATE('now', '-24 hours')
        ''')
        health_stats = cursor.fetchone()
        
        conn.close()
        
        # Calculate percentage changes with enhanced logic
        def calc_change(current, previous):
            if previous == 0:
                return 100 if current > 0 else 0
            return ((current - previous) / previous) * 100
        
        # Enhanced change calculations
        today_change = calc_change(stats[3] or 0, stats[4] or 0)
        week_change = calc_change(week_stats[0] or 0, week_stats[1] or 0)
        defect_change = calc_change(week_stats[2] or 0, week_stats[3] or 0)
        
        # Calculate processing time trend
        processing_trend_indicator = "stable"
        if perf_data['processing_trend'] == 'increasing':
            processing_trend_indicator = "↗️ increasing"
        elif perf_data['processing_trend'] == 'decreasing':
            processing_trend_indicator = "↘️ improving"
        
        # Format defect types with enhanced information
        defect_colors = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899']
        
        enhanced_defect_types = []
        for i, row in enumerate(defect_types):
            enhanced_defect_types.append({
                'name': row[0].replace('_', ' ').title(),
                'count': row[1],
                'percentage': (row[1] / max(stats[1], 1)) * 100,
                'avg_confidence': row[2] or 0,
                'avg_area': row[3] or 0,
                'avg_regions': row[4] or 0,
                'critical_count': row[5] or 0,
                'color': defect_colors[i % len(defect_colors)],
                'severity_distribution': {
                    'critical': row[5] or 0,
                    'high': max(0, row[1] - (row[5] or 0)) // 2,
                    'medium': max(0, row[1] - (row[5] or 0)) // 3,
                    'low': max(0, row[1] - (row[5] or 0)) // 4
                }
            })
        
        # Enhanced return data
        return {
            # Basic metrics (enhanced)
            'total_defects': stats[1] or 0,
            'defect_increase': f"{abs(defect_change):.0f}% {'increase' if defect_change >= 0 else 'decrease'} this week",
            'images_processed': stats[0] or 0,
            'processing_increase': f"{abs(today_change):.0f}% {'increase' if today_change >= 0 else 'decrease'} today",
            'detection_accuracy': f"{stats[2]:.1f}%" if stats[2] else "0.0%",
            'accuracy_change': f"{abs(week_change):.1f}% {'increase' if week_change >= 0 else 'decrease'} this week",
            
            # Enhanced performance metrics
            'performance': {
                'avg_processing_time': f"{stats[5]:.3f}s" if stats[5] else "0.000s",
                'min_processing_time': f"{stats[6]:.3f}s" if stats[6] else "0.000s", 
                'max_processing_time': f"{stats[7]:.3f}s" if stats[7] else "0.000s",
                'current_throughput': f"{perf_data['current_throughput']:.1f} FPS",
                'processing_trend': processing_trend_indicator,
                'avg_processing_time_5min': f"{perf_data['avg_processing_time_5min']:.3f}s"
            },
            
            # System health metrics  
            'system_health': {
                'memory_usage': f"{perf_data['system_memory_usage']:.1f}%",
                'cpu_usage': f"{perf_data['system_cpu_usage']:.1f}%", 
                'completed_analyses_24h': health_stats[0] or 0,
                'failed_analyses_24h': health_stats[1] or 0,
                'system_accuracy': f"{(health_stats[2] or 0.85) * 100:.1f}%",
                'health_score': self._calculate_system_health_score(perf_data, health_stats)
            },
            
            # Enhanced defect analysis
            'defect_types': enhanced_defect_types,
            'defect_summary': {
                'total_types': len(defect_types),
                'most_common': defect_types[0][0].replace('_', ' ').title() if defect_types else 'None',
                'critical_defects': sum(row[5] or 0 for row in defect_types),
                'avg_confidence_overall': np.mean([row[2] for row in defect_types if row[2]]) if defect_types else 0
            },
            
            # Activity patterns
            'activity_patterns': {
                'hourly_distribution': [{'hour': f"{row[0]}:00", 'count': row[1]} for row in hourly_pattern],
                'peak_hour': max(hourly_pattern, key=lambda x: x[1])[0] + ":00" if hourly_pattern else "N/A",
                'busiest_day': 'Today' if stats[3] > stats[4] else 'Yesterday'
            },
            
            # Trends and insights
            'trends': {
                'processing_performance': perf_data['processing_trend'],
                'quality_trend': perf_data['quality_trend'],
                'volume_trend': 'increasing' if today_change > 0 else 'decreasing',
                'defect_rate_trend': 'increasing' if defect_change > 0 else 'decreasing'
            },
            
            # Quality insights
            'quality_insights': {
                'avg_confidence': f"{perf_data['avg_confidence'] * 100:.1f}%",
                'defect_rate': f"{(stats[1] / max(stats[0], 1)) * 100:.1f}%",
                'success_rate': f"{((health_stats[0] or 0) / max((health_stats[0] or 0) + (health_stats[1] or 0), 1)) * 100:.1f}%",
                'quality_score': self._calculate_quality_score(stats, perf_data)
            },
            
            # Real-time indicators
            'realtime': {
                'last_updated': datetime.now().isoformat(),
                'samples_analyzed': perf_data['total_samples'],
                'system_load': 'Normal' if perf_data['system_cpu_usage'] < 70 else 'High',
                'processing_status': 'Optimal' if perf_data['current_throughput'] > 0.5 else 'Slow'
            }
        }

    def _calculate_system_health_score(self, perf_data, health_stats):
        """Calculate overall system health score (0-100)"""
        try:
            # Performance health (30%)
            throughput_score = min(100, perf_data['current_throughput'] * 50)
            
            # System resource health (25%)
            memory_score = max(0, 100 - perf_data['system_memory_usage'])
            cpu_score = max(0, 100 - perf_data['system_cpu_usage'])
            resource_score = (memory_score + cpu_score) / 2
            
            # Analysis success rate (25%)
            total_analyses = (health_stats[0] or 0) + (health_stats[1] or 0)
            success_rate = ((health_stats[0] or 0) / max(total_analyses, 1)) * 100
            
            # Quality score (20%)
            quality_score = (health_stats[2] or 0.85) * 100
            
            # Weighted average
            overall_score = (
                throughput_score * 0.30 +
                resource_score * 0.25 + 
                success_rate * 0.25 +
                quality_score * 0.20
            )
            
            return round(overall_score, 1)
            
        except Exception:
            return 75.0  # Default good health score

    def _calculate_quality_score(self, stats, perf_data):
        """Calculate overall quality score based on accuracy and confidence"""
        try:
            # Accuracy score
            accuracy = stats[2] or 85.0
            
            # Confidence score  
            confidence = perf_data['avg_confidence'] * 100
            
            # Defect detection reliability
            defect_rate = (stats[1] / max(stats[0], 1)) * 100
            reliability_score = 100 - min(50, defect_rate * 2)  # Cap penalty at 50 points
            
            # Combined quality score
            quality_score = (accuracy * 0.4 + confidence * 0.4 + reliability_score * 0.2)
            
            return round(quality_score, 1)
            
        except Exception:
            return 85.0  # Default quality score
    
    def get_recent_analyses(self, limit=10):
        """
        Get recent analyses for dashboard display
        
        Args:
            limit: Number of recent analyses to return
            
        Returns:
            list: Recent analysis records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_name, analysis_date, detected_defects, final_decision, 
                   status, confidence_level, processing_time
            FROM analyses
            ORDER BY analysis_date DESC
            LIMIT ?
        ''', (limit,))
        
        analyses = []
        for row in cursor.fetchall():
            defects = json.loads(row[3]) if row[3] else []
            analyses.append({
                'id': row[0],
                'image_name': row[1],
                'date': datetime.fromisoformat(row[2]).strftime("%b %d, %Y"),
                'time': datetime.fromisoformat(row[2]).strftime("%I:%M %p"),
                'time_ago': self._time_ago(datetime.fromisoformat(row[2])),
                'defects': f"{len(defects)} defects" if defects else "No defects",
                'defect_list': defects,
                'status': row[5],
                'decision': row[4],
                'confidence': row[6],
                'processing_time': f"{row[7]:.2f}s" if row[7] else "0.00s"
            })
        
        conn.close()
        return analyses
    
    def _time_ago(self, dt):
        """Calculate human-readable time difference"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"about {diff.days} {'day' if diff.days == 1 else 'days'} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"in about {hours} {'hour' if hours == 1 else 'hours'}"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"about {minutes} {'minute' if minutes == 1 else 'minutes'} ago"
        else:
            return "just now"
    
    def _setup_api_routes(self):
        """Setup all API endpoints"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """System health check endpoint"""
            return jsonify({
                'status': 'ok',
                'detector_ready': self.detector.is_ready() if self.detector else False,
                'detector_status': getattr(self, 'detector_status', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'database': 'connected',
                'performance_tracker': 'available' if self.has_performance_tracker() else 'unavailable',
                'version': '2.0.0'
            })
        
        @self.app.route('/api/system-info', methods=['GET'])
        def get_system_info():
            """Get detailed system information"""
            if self.detector:
                try:
                    info = self.detector.get_system_info()
                    info['detector_status'] = self.detector_status
                    info['performance_tracker'] = self.has_performance_tracker()
                    return jsonify({'status': 'success', 'data': info})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            else:
                return jsonify({
                    'status': 'success',
                    'data': {
                        'device': 'cpu',
                        'models_loaded': False,
                        'system_ready': False,
                        'anomaly_threshold': 0.7,
                        'defect_threshold': 0.85,
                        'detector_status': self.detector_status,
                        'performance_tracker': self.has_performance_tracker(),
                        'supported_formats': ['JPG', 'PNG', 'BMP'],
                        'max_file_size': '5MB'
                    }
                })
        
        @self.app.route('/api/detect-image', methods=['POST'])
        def detect_image():
            """Main image detection endpoint with performance tracking"""
            analysis_id = None
            perf_start = None
            
            try:
                # Start performance measurement
                if self.performance_tracker:
                    perf_start = self.performance_tracker.start_measurement()
                
                # Handle different input formats (keep existing code)
                if 'image' in request.files:
                    image_file = request.files['image']
                    if image_file.filename == '':
                        return jsonify({'error': 'No file selected'}), 400
                    
                    image_data = image_file.read()
                    filename = image_file.filename
                    
                elif request.json and 'image_base64' in request.json:
                    base64_data = request.json['image_base64']
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',')[1]
                    
                    try:
                        image_data = base64.b64decode(base64_data)
                    except Exception:
                        return jsonify({'error': 'Invalid base64 image data'}), 400
                    
                    filename = request.json.get('filename', f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    
                else:
                    return jsonify({'error': 'No image provided. Use form-data with "image" field or JSON with "image_base64"'}), 400
                
                # Validate file size (5MB limit)
                if len(image_data) > 5 * 1024 * 1024:
                    return jsonify({'error': 'File too large. Maximum size is 5MB'}), 400
                
                # Save uploaded image
                upload_dir = 'static/uploads'
                os.makedirs(upload_dir, exist_ok=True)
                
                # Generate unique filename
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    file_ext = '.jpg'
                
                unique_filename = f"{uuid.uuid4().hex}{file_ext}"
                image_path = os.path.join(upload_dir, unique_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                # Get image dimensions
                try:
                    img = cv2.imread(image_path)
                    image_size = (img.shape[1], img.shape[0]) if img is not None else None
                except:
                    image_size = None
                
                # Process image with detection system
                start_time = time.time()
                
                if self.detector and hasattr(self.detector, 'process_image'):
                    # Use real detector
                    result = self.detector.process_image(image_path)
                else:
                    # Use enhanced mock detector
                    processing_time = time.time() - start_time
                    result = self._create_enhanced_mock_result(image_path, processing_time, filename)
                
                if result:
                    # Save to database FIRST to get analysis_id
                    analysis_id = self.save_analysis_to_db(result, filename, image_size)
                    
                    # End performance measurement with analysis_id
                    if self.performance_tracker and perf_start:
                        self.performance_tracker.end_measurement(perf_start, result, analysis_id)
                    
                    # Format response for API
                    response = self._format_api_response(result, analysis_id, unique_filename)
                    return jsonify(response)
                else:
                    return jsonify({'error': 'Image processing failed'}), 500
                    
            except Exception as e:
                print(f"Detection error: {e}")
                
                # End performance measurement even on error
                if self.performance_tracker and perf_start:
                    error_result = {
                        'final_decision': 'ERROR',
                        'anomaly_detection': {'anomaly_score': 0},
                        'processing_time': time.time() - (perf_start if perf_start else time.time()),
                        'detected_defect_types': []
                    }
                    self.performance_tracker.end_measurement(perf_start, error_result, analysis_id)
                
                return jsonify({'error': f'Processing error: {str(e)}'}), 500
                
        @self.app.route('/api/dashboard-stats', methods=['GET'])
        def dashboard_stats():
            """Get enhanced dashboard statistics with performance integration"""
            try:
                stats = self.get_dashboard_stats()  # Now uses enhanced version
                return jsonify({'status': 'success', 'data': stats})
            except Exception as e:
                print(f"Error getting dashboard stats: {e}")
                return jsonify({'error': str(e)}), 500        
        
        @self.app.route('/api/recent-analyses', methods=['GET'])
        def recent_analyses():
            """Get recent analyses for dashboard"""
            try:
                limit = request.args.get('limit', 10, type=int)
                analyses = self.get_recent_analyses(limit)
                return jsonify({'status': 'success', 'data': analyses})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analysis-history', methods=['GET'])
        def analysis_history():
            """Get paginated analysis history"""
            try:
                page = request.args.get('page', 1, type=int)
                per_page = request.args.get('per_page', 20, type=int)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute('SELECT COUNT(*) FROM analyses')
                total = cursor.fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * per_page
                cursor.execute('''
                    SELECT id, image_name, analysis_date, detected_defects, final_decision, 
                           status, anomaly_score, processing_time, confidence_level
                    FROM analyses
                    ORDER BY analysis_date DESC
                    LIMIT ? OFFSET ?
                ''', (per_page, offset))
                
                analyses = []
                for row in cursor.fetchall():
                    defects = json.loads(row[3]) if row[3] else []
                    analyses.append({
                        'id': row[0],
                        'image': row[1],
                        'analysis_date': datetime.fromisoformat(row[2]).strftime("%b %d, %Y"),
                        'analysis_time': datetime.fromisoformat(row[2]).strftime("%I:%M %p"),
                        'defects': ', '.join([d.replace('_', ' ').title() for d in defects]) if defects else 'No defects',
                        'defect_count': len(defects),
                        'status': row[5],
                        'decision': row[4],
                        'score': f"{row[6]:.3f}" if row[6] else "0.000",
                        'time': f"{row[7]:.2f}s" if row[7] else "0.00s",
                        'confidence': row[8] or "Unknown"
                    })
                
                conn.close()
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'analyses': analyses,
                        'pagination': {
                            'page': page,
                            'per_page': per_page,
                            'total': total,
                            'pages': (total + per_page - 1) // per_page
                        }
                    }
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def user_settings():
            """Get or update user settings"""
            if request.method == 'GET':
                # Get current settings
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT anomaly_threshold, defect_threshold, notification_enabled, 
                           auto_save_results, preferred_format
                    FROM user_settings WHERE user_id = ?
                ''', ('default',))
                
                settings = cursor.fetchone()
                conn.close()
                
                if settings:
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'anomaly_threshold': settings[0],
                            'defect_threshold': settings[1],
                            'notifications_enabled': bool(settings[2]),
                            'auto_save_results': bool(settings[3]),
                            'preferred_format': settings[4]
                        }
                    })
                else:
                    # Return default settings
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'anomaly_threshold': 0.7,
                            'defect_threshold': 0.85,
                            'notifications_enabled': True,
                            'auto_save_results': True,
                            'preferred_format': 'JSON'
                        }
                    })
            
            else:  # POST - Update settings
                try:
                    data = request.json
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO user_settings 
                        (user_id, anomaly_threshold, defect_threshold, notification_enabled, 
                         auto_save_results, preferred_format, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        'default',
                        data.get('anomaly_threshold', 0.7),
                        data.get('defect_threshold', 0.85),
                        data.get('notifications_enabled', True),
                        data.get('auto_save_results', True),
                        data.get('preferred_format', 'JSON')
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    return jsonify({'status': 'success', 'message': 'Settings updated successfully'})
                    
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analysis/<int:analysis_id>', methods=['GET'])
        def get_analysis_details(analysis_id):
            """Get detailed analysis information"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get main analysis data
                cursor.execute('''
                    SELECT * FROM analyses WHERE id = ?
                ''', (analysis_id,))
                
                analysis = cursor.fetchone()
                if not analysis:
                    return jsonify({'error': 'Analysis not found'}), 404
                
                # Get defect statistics
                cursor.execute('''
                    SELECT * FROM defect_statistics WHERE analysis_id = ?
                ''', (analysis_id,))
                
                defect_stats = cursor.fetchall()
                conn.close()
                
                # Format response
                detected_defects = json.loads(analysis[7]) if analysis[7] else []
                
                result = {
                    'id': analysis[0],
                    'image_name': analysis[1],
                    'image_path': analysis[2],
                    'original_size': analysis[3],
                    'final_decision': analysis[4],
                    'anomaly_score': analysis[5],
                    'confidence_level': analysis[6],
                    'detected_defects': detected_defects,
                    'processing_time': analysis[8],
                    'analysis_date': analysis[9],
                    'status': analysis[10],
                    'notes': analysis[12],
                    'visualization_path': analysis[13],
                    'defect_statistics': []
                }
                
                # Add defect statistics
                for stat in defect_stats:
                    result['defect_statistics'].append({
                        'defect_type': stat[2],
                        'confidence': stat[3],
                        'area_percentage': stat[4],
                        'bbox_count': stat[5],
                        'severity_level': stat[6],
                        'location_data': json.loads(stat[7]) if stat[7] else []
                    })
                
                return jsonify({'status': 'success', 'data': result})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/delete-analysis/<int:analysis_id>', methods=['DELETE'])
        def delete_analysis(analysis_id):
            """Delete an analysis record"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete defect statistics first (foreign key constraint)
                cursor.execute('DELETE FROM defect_statistics WHERE analysis_id = ?', (analysis_id,))
                
                # Delete performance metrics 
                cursor.execute('DELETE FROM performance_metrics WHERE analysis_id = ?', (analysis_id,))
                
                # Delete main analysis
                cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
                
                if cursor.rowcount == 0:
                    return jsonify({'error': 'Analysis not found'}), 404
                
                conn.commit()
                conn.close()
                
                return jsonify({'status': 'success', 'message': 'Analysis deleted successfully'})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/analysis-charts/<int:analysis_id>', methods=['GET'])
        def get_analysis_charts(analysis_id):
            """Get chart data for specific analysis"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get analysis data
                cursor.execute('''
                    SELECT final_decision, anomaly_score, processing_time, detected_defects
                    FROM analyses WHERE id = ?
                ''', (analysis_id,))
                
                analysis = cursor.fetchone()
                if not analysis:
                    return jsonify({'error': 'Analysis not found'}), 404
                
                # Get defect statistics
                cursor.execute('''
                    SELECT defect_type, confidence, area_percentage, severity_level
                    FROM defect_statistics WHERE analysis_id = ?
                ''', (analysis_id,))
                
                defect_stats = cursor.fetchall()
                conn.close()
                
                # Prepare chart data
                chart_data = {
                    'confidence_chart': {
                        'labels': [stat[0].replace('_', ' ').title() for stat in defect_stats],
                        'data': [stat[1] * 100 for stat in defect_stats],
                        'backgroundColor': ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6']
                    },
                    'area_chart': {
                        'labels': [stat[0].replace('_', ' ').title() for stat in defect_stats],
                        'data': [stat[2] for stat in defect_stats],
                        'backgroundColor': ['#fee2e2', '#fef3c7', '#dbeafe', '#d1fae5', '#ede9fe']
                    },
                    'severity_chart': {
                        'labels': ['Critical', 'High', 'Medium', 'Low'],
                        'data': [
                            sum(1 for stat in defect_stats if stat[3] == 'Critical'),
                            sum(1 for stat in defect_stats if stat[3] == 'High'),
                            sum(1 for stat in defect_stats if stat[3] == 'Medium'),
                            sum(1 for stat in defect_stats if stat[3] == 'Low')
                        ],
                        'backgroundColor': ['#dc2626', '#ea580c', '#d97706', '#65a30d']
                    },
                    'performance_metrics': {
                        'processing_time': analysis[2],
                        'anomaly_score': analysis[1],
                        'decision': analysis[0]
                    }
                }
                
                return jsonify({'status': 'success', 'data': chart_data})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard-charts', methods=['GET'])
        def get_dashboard_charts():
            """Enhanced dashboard charts with performance integration"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 1. Performance trend (last 30 days) - ENHANCED
                cursor.execute('''
                    SELECT DATE(a.analysis_date) as date, 
                           AVG(processing_time) as avg_time,
                           COUNT(*) as count,
                           AVG(0 as memory_usage) as avg_memory,
                           AVG(0 as cpu_usage) as avg_cpu,
                           MIN(a.processing_time) as min_time,
                           MAX(a.processing_time) as max_time
                    FROM performance_metrics pm
                    JOIN analyses a ON pm.analysis_id = a.id
                    WHERE a.analysis_date >= DATE('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''')
                
                perf_data = cursor.fetchall()
                
                # 2. Real-time performance from performance_tracker
                try:
                    if self.performance_tracker:
                        realtime_perf = self.performance_tracker.get_realtime_metrics()
                        perf_charts = self.performance_tracker.get_chart_data_for_dashboard()
                    else:
                        realtime_perf = {}
                        perf_charts = {}
                except:
                    # Fallback if performance_tracker not available
                    realtime_perf = {}
                    perf_charts = {}
                
                # 3. Defect distribution with confidence scores
                cursor.execute('''
                    SELECT defect_type, COUNT(*) as count, 
                           AVG(confidence) as avg_conf,
                           AVG(area_percentage) as avg_area,
                           MAX(confidence) as max_conf
                    FROM defect_statistics ds
                    JOIN analyses a ON ds.analysis_id = a.id
                    WHERE a.analysis_date >= DATE('now', '-30 days')
                    GROUP BY defect_type
                    ORDER BY count DESC
                    LIMIT 10
                ''')
                
                defect_dist = cursor.fetchall()
                
                # 4. Quality metrics over time (last 7 days)
                cursor.execute('''
                    SELECT DATE(analysis_date) as date,
                           COUNT(*) as total,
                           SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects,
                           AVG(processing_time) as avg_time,
                           AVG(anomaly_score) as avg_score,
                           COUNT(CASE WHEN final_decision = 'GOOD' THEN 1 END) as good_products
                    FROM analyses
                    WHERE analysis_date >= DATE('now', '-7 days')
                    GROUP BY DATE(analysis_date)
                    ORDER BY date
                ''')
                
                daily_stats = cursor.fetchall()
                
                # 5. Hourly analysis pattern (last 24 hours)
                cursor.execute('''
                    SELECT strftime('%H', analysis_date) as hour,
                           COUNT(*) as count,
                           AVG(processing_time) as avg_time,
                           SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects
                    FROM analyses
                    WHERE analysis_date >= DATETIME('now', '-24 hours')
                    GROUP BY strftime('%H', analysis_date)
                    ORDER BY hour
                ''')
                
                hourly_data = cursor.fetchall()
                
                # 6. System resource usage (if available)
                cursor.execute('''
                    SELECT datetime(timestamp) as time,
                           memory_percent,
                           cpu_percent,
                           disk_usage
                    FROM system_performance 
                    WHERE timestamp >= DATETIME('now', '-2 hours')
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')
                
                resource_data = cursor.fetchall()
                
                conn.close()
                
                # Format enhanced chart data
                charts = {
                    # Enhanced Performance Trend Chart
                    'performance_trend': {
                        'labels': [row[0] for row in perf_data],
                        'datasets': [
                            {
                                'label': 'Avg Processing Time (s)',
                                'data': [round(row[1], 3) if row[1] else 0 for row in perf_data],
                                'borderColor': '#3b82f6',
                                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                                'tension': 0.4,
                                'yAxisID': 'y',
                                'fill': True
                            },
                            {
                                'label': 'Analyses Count',
                                'data': [row[2] for row in perf_data],
                                'borderColor': '#10b981',
                                'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                                'tension': 0.4,
                                'yAxisID': 'y1',
                                'type': 'bar'
                            }
                        ],
                        'options': {
                            'scales': {
                                'y': {
                                    'type': 'linear',
                                    'display': True,
                                    'position': 'left',
                                    'title': {'display': True, 'text': 'Processing Time (s)'}
                                },
                                'y1': {
                                    'type': 'linear',
                                    'display': True,
                                    'position': 'right',
                                    'title': {'display': True, 'text': 'Count'},
                                    'grid': {'drawOnChartArea': False}
                                }
                            }
                        }
                    },
                    
                    # Enhanced Defect Distribution
                    'defect_distribution': {
                        'labels': [row[0].replace('_', ' ').title() for row in defect_dist],
                        'datasets': [{
                            'label': 'Defect Count',
                            'data': [row[1] for row in defect_dist],
                            'backgroundColor': [
                                '#ef4444', '#f59e0b', '#3b82f6', '#10b981', 
                                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
                            ],
                            'borderWidth': 2,
                            'borderColor': '#ffffff'
                        }],
                        'confidence_data': [row[2] for row in defect_dist],  # Additional data
                        'area_data': [row[3] for row in defect_dist]
                    },
                    
                    # Daily Analysis Trend
                    'daily_analysis': {
                        'labels': [row[0] for row in daily_stats],
                        'datasets': [
                            {
                                'label': 'Total Analyses',
                                'data': [row[1] for row in daily_stats],
                                'borderColor': '#10b981',
                                'backgroundColor': 'rgba(16, 185, 129, 0.2)',
                                'type': 'line',
                                'tension': 0.4
                            },
                            {
                                'label': 'Defects Found',
                                'data': [row[2] for row in daily_stats],
                                'borderColor': '#ef4444',
                                'backgroundColor': 'rgba(239, 68, 68, 0.2)',
                                'type': 'bar'
                            },
                            {
                                'label': 'Good Products',
                                'data': [row[5] for row in daily_stats],
                                'borderColor': '#22c55e',
                                'backgroundColor': 'rgba(34, 197, 94, 0.2)',
                                'type': 'bar'
                            }
                        ]
                    },
                    
                    # Hourly Pattern Analysis
                    'hourly_pattern': {
                        'labels': [f"{row[0].zfill(2)}:00" for row in hourly_data],
                        'datasets': [
                            {
                                'label': 'Analyses per Hour',
                                'data': [row[1] for row in hourly_data],
                                'backgroundColor': 'rgba(59, 130, 246, 0.8)',
                                'borderColor': '#3b82f6',
                                'borderWidth': 2
                            },
                            {
                                'label': 'Defects per Hour',
                                'data': [row[3] for row in hourly_data],
                                'backgroundColor': 'rgba(239, 68, 68, 0.8)',
                                'borderColor': '#ef4444',
                                'borderWidth': 2
                            }
                        ]
                    },
                    
                    # System Resource Usage (Real-time)
                    'resource_usage': {
                        'labels': [row[0][-8:] for row in resource_data],  # HH:MM:SS
                        'datasets': [
                            {
                                'label': 'Memory Usage (%)',
                                'data': [row[1] for row in resource_data],
                                'borderColor': '#8b5cf6',
                                'backgroundColor': 'rgba(139, 92, 246, 0.1)',
                                'tension': 0.4,
                                'fill': True
                            },
                            {
                                'label': 'CPU Usage (%)',
                                'data': [row[2] for row in resource_data],
                                'borderColor': '#f59e0b',
                                'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                                'tension': 0.4,
                                'fill': True
                            }
                        ]
                    },
                    
                    # Quality Metrics Summary
                    'quality_metrics': {
                        'accuracy': self._calculate_accuracy_from_daily_stats(daily_stats),
                        'throughput': self._calculate_avg_throughput(perf_data),
                        'defect_rate': self._calculate_defect_rate(daily_stats),
                        'avg_confidence': self._calculate_avg_confidence(defect_dist),
                        'system_efficiency': {
                            'memory_efficiency': 100 - (sum(row[1] for row in resource_data[-10:]) / min(10, len(resource_data))) if resource_data else 85,
                            'cpu_efficiency': 100 - (sum(row[2] for row in resource_data[-10:]) / min(10, len(resource_data))) if resource_data else 88
                        }
                    },
                    
                    # Performance Summary Radar Data
                    'performance_radar': {
                        'labels': ['Speed', 'Accuracy', 'Efficiency', 'Reliability', 'Consistency'],
                        'datasets': [{
                            'label': 'Performance Score',
                            'data': [
                                self._calculate_speed_score(perf_data),
                                self._calculate_accuracy_from_daily_stats(daily_stats),
                                self._calculate_efficiency_score(resource_data),
                                self._calculate_reliability_score(daily_stats),
                                self._calculate_consistency_score(perf_data)
                            ],
                            'borderColor': '#3b82f6',
                            'backgroundColor': 'rgba(59, 130, 246, 0.2)',
                            'pointBackgroundColor': '#3b82f6',
                            'pointBorderColor': '#ffffff',
                            'pointHoverBackgroundColor': '#ffffff',
                            'pointHoverBorderColor': '#3b82f6'
                        }]
                    },
                    
                    # Real-time Metrics from Performance Tracker
                    'realtime_metrics': realtime_perf,
                    
                    # Integration with Performance Tracker Charts
                    'performance_tracker_data': perf_charts
                }
                
                return jsonify({'status': 'success', 'data': charts})
                
            except Exception as e:
                print(f"Error generating dashboard charts: {e}")
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/browse-folder', methods=['POST'])
        def browse_folder():
            """API endpoint untuk folder browsing"""
            try:
                folder_path = request.json.get('path', '')
                
                if os.path.exists(folder_path):
                    items = []
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        is_dir = os.path.isdir(item_path)
                        
                        if not is_dir:
                            if not item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                continue
                        
                        items.append({
                            'name': item,
                            'path': item_path,
                            'is_directory': is_dir,
                            'size': os.path.getsize(item_path) if not is_dir else 0
                        })
                    
                    return jsonify({
                        'status': 'success',
                        'items': items,
                        'current_path': folder_path
                    })
                else:
                    return jsonify({'error': 'Path not found'}), 404
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/get-drives', methods=['GET'])
        def get_system_drives():
            """Get available system drives"""
            try:
                import string
                drives = []
                
                if os.name == 'nt':  # Windows
                    for letter in string.ascii_uppercase:
                        drive = f"{letter}:\\"
                        if os.path.exists(drive):
                            drives.append({
                                'name': drive,
                                'path': drive,
                                'type': 'drive',
                                'is_directory': True
                            })
                else:  # Linux/Mac
                    drives.append({
                        'name': 'Root',
                        'path': '/',
                        'type': 'drive',
                        'is_directory': True
                    })
                    
                    home = os.path.expanduser('~')
                    drives.append({
                        'name': 'Home',
                        'path': home,
                        'type': 'directory',
                        'is_directory': True
                    })
                
                return jsonify({
                    'status': 'success',
                    'drives': drives
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/copy-file', methods=['POST'])
        def copy_file():
            """Copy file from file system to static uploads folder"""
            try:
                file_path = request.form.get('file_path')
                
                if not file_path or not os.path.exists(file_path):
                    return jsonify({'error': 'File not found'}), 404
                
                if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    return jsonify({'error': 'Unsupported file type'}), 400
                
                upload_dir = 'static/uploads'
                os.makedirs(upload_dir, exist_ok=True)
                
                filename = os.path.basename(file_path)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                destination_path = os.path.join(upload_dir, unique_filename)
                
                shutil.copy2(file_path, destination_path)
                
                return jsonify({
                    'status': 'success',
                    'filename': unique_filename,
                    'image_url': f'/static/uploads/{unique_filename}',
                    'original_path': file_path
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/generate-test-images', methods=['POST'])
        def generate_test_images():
            """Generate test images seperti di run_tests.py"""
            try:
                data = request.json
                output_dir = data.get('output_dir', 'generated_test_images')
                count_per_type = data.get('count_per_type', 3)
                
                os.makedirs(output_dir, exist_ok=True)
                
                test_types = [
                    ("good_product", "GOOD", lambda img: img),
                    ("scratched", "DEFECT", lambda img: add_scratch(img)),
                    ("stained", "DEFECT", lambda img: add_stain(img)),
                    ("damaged", "DEFECT", lambda img: add_damage(img)),
                    ("missing_part", "DEFECT", lambda img: add_missing_part(img)),
                ]
                
                generated_files = []
                
                for i, (name, expected, modifier) in enumerate(test_types):
                    for j in range(count_per_type):
                        base_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
                        cv2.rectangle(base_img, (150, 100), (490, 380), (180, 180, 220), -1)
                        cv2.putText(base_img, f"{name.upper()}", (200, 230), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                        
                        modified_img = modifier(base_img.copy())
                        
                        filename = f"{name}_{j+1}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, modified_img)
                        
                        generated_files.append({
                            'filename': filename,
                            'path': filepath,
                            'expected': expected,
                            'type': name
                        })
                
                return jsonify({
                    'status': 'success',
                    'generated_files': generated_files,
                    'output_directory': output_dir,
                    'total_generated': len(generated_files)
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/camera-capture', methods=['POST'])
        def camera_capture():
            """Camera capture endpoint"""
            try:
                camera_id = request.json.get('camera_id', 0)
                
                cap = cv2.VideoCapture(camera_id)
                
                if not cap.isOpened():
                    return jsonify({'error': f'Could not open camera {camera_id}'}), 400
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return jsonify({'error': 'Failed to capture frame'}), 400
                
                os.makedirs('static/captured', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join('static/captured', filename)
                cv2.imwrite(filepath, frame)
                
                return jsonify({
                    'status': 'success',
                    'filename': filename,
                    'filepath': filepath,
                    'image_url': f'/static/captured/{filename}',
                    'timestamp': timestamp
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/performance-test', methods=['POST'])
        def run_performance_test():
            """Performance testing endpoint"""
            try:
                data = request.json
                iterations = data.get('iterations', 10)
                test_image_size = data.get('image_size', [640, 480])
                
                test_img = np.ones((test_image_size[1], test_image_size[0], 3), dtype=np.uint8) * 200
                temp_path = "temp_perf_test.jpg"
                cv2.imwrite(temp_path, test_img)
                
                times = []
                results = []
                
                for i in range(iterations):
                    start_time = time.time()
                    
                    if self.detector and hasattr(self.detector, 'process_image'):
                        result = self.detector.process_image(temp_path)
                    else:
                        result = self._create_enhanced_mock_result(temp_path, 0, "test_image.jpg")
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    times.append(processing_time)
                    
                    results.append({
                        'iteration': i + 1,
                        'processing_time': processing_time,
                        'decision': result.get('final_decision', 'Unknown') if result else 'Error'
                    })
                
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return jsonify({
                    'status': 'success',
                    'performance_stats': {
                        'iterations': iterations,
                        'average_time': avg_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'std_deviation': std_time,
                        'fps': 1 / avg_time if avg_time > 0 else 0,
                        'total_time': sum(times)
                    },
                    'detailed_results': results,
                    'time_series': times
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # REAL-TIME PROCESSING ENDPOINTS
        @self.app.route('/api/realtime/start-session', methods=['POST'])
        def start_realtime_session():
            """Start new real-time detection session"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                success = self.realtime_processor.start_session()
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'message': 'Real-time session started',
                        'session_id': self.realtime_processor.current_session_id,
                        'start_time': self.realtime_processor.session_start_time.isoformat()
                    })
                else:
                    return jsonify({'error': 'Failed to start session'}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/stop-session', methods=['POST'])
        def stop_realtime_session():
            """Stop current real-time detection session"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                session_report = self.realtime_processor.stop_session()
                
                if session_report:
                    return jsonify({
                        'status': 'success',
                        'message': 'Session stopped successfully',
                        'session_report': session_report
                    })
                else:
                    return jsonify({'error': 'No active session to stop'}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/process-frame', methods=['POST'])
        def process_realtime_frame():
            """Process single frame from real-time camera feed"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                if not self.realtime_processor.session_active:
                    return jsonify({'error': 'No active real-time session'}), 400
                
                # Get frame data
                if 'image' in request.files:
                    image_file = request.files['image']
                    frame_data = image_file.read()
                    
                    # Convert to base64 for processing
                    import base64
                    frame_data = base64.b64encode(frame_data).decode('utf-8')
                    
                elif request.json and 'image_base64' in request.json:
                    frame_data = request.json['image_base64']
                    if frame_data.startswith('data:image'):
                        frame_data = frame_data.split(',')[1]
                else:
                    return jsonify({'error': 'No frame data provided'}), 400
                
                # Get settings
                auto_capture = request.json.get('auto_capture', True) if request.json else True
                
                # Process frame
                result = self.realtime_processor.process_frame(frame_data, auto_capture)
                
                if 'error' in result:
                    return jsonify({'error': result['error']}), 500
                
                # Format response for real-time display
                response = {
                    'status': 'success',
                    'frame_result': {
                        'session_id': result.get('session_id'),
                        'frame_number': result.get('frame_number'),
                        'decision': result['final_decision'],
                        'anomaly_score': result['anomaly_detection']['anomaly_score'],
                        'confidence': self._calculate_confidence_level(result),
                        'processing_time': result.get('realtime_processing_time', 0),
                        'detected_defects': result.get('detected_defect_types', []),
                        'timestamp': result.get('realtime_timestamp')
                    },
                    'session_stats': result.get('session_stats', {}),
                    'screenshot_captured': result.get('screenshot_captured', None)
                }
                
                # Add bounding box data if available
                if result.get('defect_classification') and result['final_decision'] == 'DEFECT':
                    bounding_boxes = []
                    defect_analysis = result['defect_classification'].get('defect_analysis', {})
                    
                    for defect_type, boxes in defect_analysis.get('bounding_boxes', {}).items():
                        for box in boxes:
                            bounding_boxes.append({
                                'defect_type': defect_type,
                                'x': box['x'],
                                'y': box['y'],
                                'width': box['width'],
                                'height': box['height'],
                                'confidence': box.get('confidence', 0.8)
                            })
                    
                    response['frame_result']['bounding_boxes'] = bounding_boxes
                
                return jsonify(response)
                
            except Exception as e:
                print(f"Error processing real-time frame: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/capture-screenshot', methods=['POST'])
        def capture_realtime_screenshot():
            """Manually capture screenshot from real-time feed"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                if not self.realtime_processor.session_active:
                    return jsonify({'error': 'No active real-time session'}), 400
                
                # Get frame data
                if request.json and 'image_base64' in request.json:
                    frame_data = request.json['image_base64']
                    if frame_data.startswith('data:image'):
                        frame_data = frame_data.split(',')[1]
                else:
                    return jsonify({'error': 'No frame data provided'}), 400
                
                # Get optional detection result
                detection_result = request.json.get('detection_result', None)
                
                # Capture screenshot
                screenshot_info = self.realtime_processor.capture_manual_screenshot(
                    frame_data, detection_result
                )
                
                if 'error' in screenshot_info:
                    return jsonify({'error': screenshot_info['error']}), 500
                
                return jsonify({
                    'status': 'success',
                    'message': 'Screenshot captured successfully',
                    'screenshot': screenshot_info
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/session-stats', methods=['GET'])
        def get_realtime_session_stats():
            """Get current real-time session statistics"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                stats = self.realtime_processor.get_session_statistics()
                
                if 'error' in stats:
                    return jsonify({'error': stats['error']}), 400
                
                return jsonify({
                    'status': 'success',
                    'session_stats': stats,
                    'session_active': self.realtime_processor.session_active
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/recent-captures', methods=['GET'])
        def get_recent_realtime_captures():
            """Get recent capture records"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                limit = request.args.get('limit', 10, type=int)
                captures = self.realtime_processor.get_recent_captures(limit)
                
                return jsonify({
                    'status': 'success',
                    'captures': captures
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/session-history', methods=['GET'])
        def get_realtime_session_history():
            """Get real-time session history"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                limit = request.args.get('limit', 20, type=int)
                sessions = self.realtime_processor.get_session_history(limit)
                
                # Format for display
                formatted_sessions = []
                for session in sessions:
                    formatted_sessions.append({
                        'id': session['id'],
                        'start_time': datetime.fromisoformat(session['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': f"{session['duration_seconds'] // 60}m {session['duration_seconds'] % 60}s",
                        'total_frames': session['total_frames'],
                        'defect_rate': f"{session['defect_rate']:.1f}%",
                        'screenshots': session['screenshots_captured'],
                        'avg_time': f"{session['avg_processing_time']:.3f}s"
                    })
                
                return jsonify({
                    'status': 'success',
                    'sessions': formatted_sessions
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/save-capture', methods=['POST'])
        def save_realtime_capture():
            """Save real-time capture data (called from frontend)"""
            try:
                data = request.json
                timestamp = data.get('timestamp')
                image_data = data.get('image_data')
                detection_result = data.get('detection_result')
                
                # Save to static directory for frontend access
                if image_data:
                    import base64
                    
                    # Extract base64 data
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    # Generate filename
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"realtime_capture_{timestamp_str}.jpg"
                    filepath = os.path.join('static', 'realtime_captures', filename)
                    
                    # Create directory if not exists
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(image_data))
                    
                    # Save metadata
                    metadata = {
                        'timestamp': timestamp,
                        'filename': filename,
                        'detection_result': detection_result,
                        'saved_at': datetime.now().isoformat()
                    }
                    
                    metadata_path = filepath.replace('.jpg', '_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'Capture saved successfully',
                        'filename': filename,
                        'filepath': filepath
                    })
                else:
                    return jsonify({'error': 'No image data provided'}), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/save-session', methods=['POST'])
        def save_realtime_session():
            """Save real-time session report (called from frontend)"""
            try:
                data = request.json
                
                # Save session summary to database if real-time processor available
                if self.realtime_processor:
                    # This would be handled by the processor itself
                    pass
                
                # Also save to static directory for frontend access
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_report_{timestamp_str}.json"
                filepath = os.path.join('static', 'session_reports', filename)
                
                # Create directory if not exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save session data
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Session report saved successfully',
                    'filename': filename
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/cleanup', methods=['POST'])
        def cleanup_realtime_data():
            """Cleanup old real-time data"""
            try:
                if not self.realtime_processor:
                    return jsonify({'error': 'Real-time processor not available'}), 503
                
                days_to_keep = request.json.get('days_to_keep', 30) if request.json else 30
                self.realtime_processor.cleanup_old_sessions(days_to_keep)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Cleaned up data older than {days_to_keep} days'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/realtime/status', methods=['GET'])
        def get_realtime_status():
            """Get real-time processor status"""
            try:
                api = self  # Reference to the API instance
                status = {
                    'processor_available': api.realtime_processor is not None,
                    'session_active': False,
                    'current_session_id': None,
                    'session_stats': None
                }
                
                if api.realtime_processor:
                    status.update({
                        'session_active': api.realtime_processor.session_active,
                        'current_session_id': getattr(api.realtime_processor, 'current_session_id', None)
                    })
                    
                    if api.realtime_processor.session_active:
                        status['session_stats'] = api.realtime_processor.get_session_statistics()
                
                return jsonify({
                    'status': 'success',
                    'realtime_status': status
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    # HELPER METHODS FOR CHART CALCULATIONS
    def _calculate_accuracy_from_daily_stats(self, daily_stats):
        """Calculate accuracy percentage from daily stats"""
        if not daily_stats:
            return 95.0
        
        total_analyses = sum(row[1] for row in daily_stats)
        total_good = sum(row[5] for row in daily_stats)
        
        return (total_good / total_analyses * 100) if total_analyses > 0 else 95.0

    def _calculate_avg_throughput(self, perf_data):
        """Calculate average throughput from performance data"""
        if not perf_data:
            return 1.2
        
        avg_times = [row[1] for row in perf_data if row[1] and row[1] > 0]
        if not avg_times:
            return 1.2
        
        avg_time = sum(avg_times) / len(avg_times)
        return 1 / avg_time if avg_time > 0 else 1.2

    def _calculate_defect_rate(self, daily_stats):
        """Calculate defect rate from daily stats"""
        if not daily_stats:
            return 15.2
        
        total_analyses = sum(row[1] for row in daily_stats)
        total_defects = sum(row[2] for row in daily_stats)
        
        return (total_defects / total_analyses * 100) if total_analyses > 0 else 15.2

    def _calculate_avg_confidence(self, defect_dist):
        """Calculate average confidence from defect distribution"""
        if not defect_dist:
            return 0.87
        
        confidences = [row[2] for row in defect_dist if row[2]]
        return sum(confidences) / len(confidences) if confidences else 0.87

    def _calculate_speed_score(self, perf_data):
        """Calculate speed score (0-100)"""
        if not perf_data:
            return 85
        
        avg_times = [row[1] for row in perf_data if row[1]]
        if not avg_times:
            return 85
        
        avg_time = sum(avg_times) / len(avg_times)
        # Convert to score: faster = higher score
        score = max(10, min(100, 100 - (avg_time * 40)))
        return round(score, 1)

    def _calculate_efficiency_score(self, resource_data):
        """Calculate system efficiency score (0-100)"""
        if not resource_data:
            return 82
        
        recent_data = resource_data[-10:] if len(resource_data) >= 10 else resource_data
        
        avg_memory = sum(row[1] for row in recent_data) / len(recent_data)
        avg_cpu = sum(row[2] for row in recent_data) / len(recent_data)
        
        # Lower resource usage = higher efficiency
        efficiency = 100 - ((avg_memory + avg_cpu) / 2)
        return max(0, min(100, round(efficiency, 1)))

    def _calculate_reliability_score(self, daily_stats):
        """Calculate reliability score based on success rate"""
        if not daily_stats:
            return 88
        
        total_analyses = sum(row[1] for row in daily_stats)
        # Assume failed analyses are tracked separately, for now use success rate
        # This is a simplified calculation
        reliability = min(100, 85 + (total_analyses / 10))  # More analyses = more reliable
        return round(reliability, 1)

    def _calculate_consistency_score(self, perf_data):
        """Calculate consistency score based on processing time variance"""
        if not perf_data or len(perf_data) < 2:
            return 78
        
        times = [row[1] for row in perf_data if row[1]]
        if len(times) < 2:
            return 78
        
        import numpy as np
        std_dev = np.std(times)
        mean_time = np.mean(times)
        
        # Lower variance = higher consistency
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 1
        consistency = max(0, min(100, 100 - (coefficient_of_variation * 100)))
        return round(consistency, 1)
    
    def _create_enhanced_mock_result(self, image_path, processing_time, filename):
        """
        Enhanced mock detection with more realistic defect patterns
        """
        import random
        import cv2
        
        # Read image to analyze content
        try:
            img = cv2.imread(image_path)
            if img is not None:
                # Analyze image characteristics
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                
                # More sophisticated defect detection logic
                defect_probability = 0.4  # Base 40% defect rate
                
                # Adjust probability based on image characteristics
                if mean_intensity < 80:  # Very dark images more likely defective
                    defect_probability += 0.3
                elif mean_intensity > 200:  # Very bright images might be overexposed
                    defect_probability += 0.2
                
                if std_intensity < 30:  # Low contrast might indicate issues
                    defect_probability += 0.2
                
                # Check filename for test patterns
                if any(word in filename.lower() for word in ['defect', 'damage', 'scratch', 'stain']):
                    defect_probability = 0.9
                elif any(word in filename.lower() for word in ['good', 'perfect', 'clean']):
                    defect_probability = 0.1
                
                is_defect = random.random() < defect_probability
            else:
                is_defect = random.choice([True, False])
        except:
            is_defect = random.choice([True, False])
        
        # Generate realistic scores
        if is_defect:
            score = random.uniform(0.72, 0.98)  # Higher scores for defects
        else:
            score = random.uniform(0.05, 0.65)   # Lower scores for good products
        
        result = {
            'image_path': image_path,
            'final_decision': 'DEFECT' if is_defect else 'GOOD',
            'processing_time': processing_time + random.uniform(0.1, 0.8),
            'timestamp': datetime.now().isoformat(),
            'anomaly_detection': {
                'anomaly_score': score,
                'decision': 'DEFECT' if is_defect else 'GOOD',
                'threshold_used': 0.7,
                'anomaly_mask': None
            },
            'detected_defect_types': []
        }
        
        # Enhanced defect classification for defective products
        if is_defect:
            available_defects = ['scratch', 'stained', 'damaged', 'missing_component', 'open']
            num_defects = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            result['detected_defect_types'] = random.sample(available_defects, num_defects)
            
            # Generate enhanced defect classification
            result['defect_classification'] = self._generate_enhanced_defect_data(
                result['detected_defect_types'], image_path
            )
            
            # Generate visualization
            result['visualization_path'] = self._create_mock_visualization(
                image_path, result['detected_defect_types']
            )
        
        return result

    def _generate_enhanced_defect_data(self, defect_types, image_path):
        """Generate realistic defect classification data"""
        import random
        
        defect_data = {
            'detected_defects': defect_types,
            'defect_analysis': {
                'defect_statistics': {},
                'class_distribution': {},
                'bounding_boxes': {},
                'spatial_analysis': {}
            }
        }
        
        # Get image dimensions
        try:
            img = cv2.imread(image_path)
            h, w = img.shape[:2] if img is not None else (480, 640)
        except:
            h, w = 480, 640
        
        defect_colors = {
            'scratch': (0, 255, 255),     # Cyan
            'stained': (128, 0, 128),     # Purple  
            'damaged': (255, 0, 0),       # Red
            'missing_component': (255, 255, 0), # Yellow
            'open': (255, 0, 255)         # Magenta
        }
        
        for i, defect in enumerate(defect_types):
            # Realistic area percentages based on defect type
            if defect == 'missing_component':
                area_pct = random.uniform(2.0, 15.0)
            elif defect == 'damaged':
                area_pct = random.uniform(1.5, 12.0) 
            elif defect == 'stained':
                area_pct = random.uniform(0.8, 8.0)
            elif defect == 'scratch':
                area_pct = random.uniform(0.3, 5.0)
            else:
                area_pct = random.uniform(0.5, 6.0)
            
            confidence = random.uniform(0.75, 0.95)
            num_regions = random.randint(1, 4)
            
            # Statistics
            defect_data['defect_analysis']['defect_statistics'][defect] = {
                'avg_confidence': confidence,
                'max_confidence': min(confidence + 0.1, 1.0),
                'num_regions': num_regions,
                'confident_pixels': int(area_pct * 100),
                'total_area': int(area_pct * w * h / 100)
            }
            
            # Class distribution
            defect_data['defect_analysis']['class_distribution'][defect] = {
                'percentage': area_pct,
                'pixel_count': int(area_pct * w * h / 100),
                'class_id': i + 1
            }
            
            # Generate realistic bounding boxes
            bboxes = []
            for j in range(num_regions):
                # Generate box dimensions based on defect type
                if defect == 'scratch':
                    # Scratches tend to be long and thin
                    box_w = random.randint(80, 200)
                    box_h = random.randint(5, 30)
                elif defect == 'missing_component':
                    # Missing components are usually rectangular
                    box_w = random.randint(40, 120)
                    box_h = random.randint(40, 100)
                else:
                    # Other defects are more irregular
                    box_w = random.randint(30, 100)
                    box_h = random.randint(20, 80)
                
                # Ensure boxes fit within image
                max_x = max(50, w - box_w - 50)
                max_y = max(50, h - box_h - 50)
                
                box_x = random.randint(50, max_x)
                box_y = random.randint(50, max_y)
                
                bbox = {
                    'id': j + 1,
                    'x': box_x,
                    'y': box_y,
                    'width': box_w,
                    'height': box_h,
                    'center_x': box_x + box_w // 2,
                    'center_y': box_y + box_h // 2,
                    'area': box_w * box_h,
                    'confidence': confidence + random.uniform(-0.05, 0.05),
                    'severity': self._calculate_mock_severity(area_pct, confidence),
                    'defect_type': defect
                }
                bboxes.append(bbox)
            
            defect_data['defect_analysis']['bounding_boxes'][defect] = bboxes
            
            # Spatial analysis
            defect_data['defect_analysis']['spatial_analysis'][defect] = {
                'quadrant_distribution': {
                    'top_left': random.randint(0, num_regions),
                    'top_right': random.randint(0, num_regions),
                    'bottom_left': random.randint(0, num_regions),
                    'bottom_right': random.randint(0, num_regions)
                },
                'center_of_mass': {
                    'x': random.randint(w//4, 3*w//4),
                    'y': random.randint(h//4, 3*h//4)
                }
            }
        
        return defect_data

    def _calculate_mock_severity(self, area_percentage, confidence):
        """Calculate defect severity for mock data"""
        if area_percentage > 10 and confidence > 0.9:
            return "Critical"
        elif area_percentage > 5 and confidence > 0.8:
            return "High"  
        elif area_percentage > 2 and confidence > 0.7:
            return "Medium"
        else:
            return "Low"

    def _create_mock_visualization(self, image_path, defect_types):
       """Create mock visualization with bounding boxes"""
       try:
           import cv2
           import os
           
           # Read original image
           img = cv2.imread(image_path)
           if img is None:
               return None
           
           # Create visualization with bounding boxes
           vis_img = img.copy()
           
           # Add some mock bounding boxes
           colors = [(0, 255, 255), (128, 0, 128), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
           
           for i, defect in enumerate(defect_types[:3]):  # Max 3 defects for clarity
               color = colors[i % len(colors)]
               
               # Random bounding box
               h, w = img.shape[:2]
               x = random.randint(50, w-150)
               y = random.randint(50, h-100)
               box_w = random.randint(60, 120)
               box_h = random.randint(40, 80)
               
               # Draw bounding box
               cv2.rectangle(vis_img, (x, y), (x + box_w, y + box_h), color, 3)
               
               # Add label
               label = defect.replace('_', ' ').title()
               cv2.putText(vis_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
           
           # Save visualization
           os.makedirs('static/visualizations', exist_ok=True)
           vis_filename = f"vis_{uuid.uuid4().hex}.jpg"
           vis_path = os.path.join('static/visualizations', vis_filename)
           cv2.imwrite(vis_path, vis_img)
           
           return vis_path
           
       except Exception as e:
           print(f"Error creating visualization: {e}")
           return None
   
    def _format_api_response(self, result, analysis_id, filename):
        """
        Format detection result for API response
        
        Args:
            result: Raw detection result
            analysis_id: Database analysis ID
            filename: Processed filename
            
        Returns:
            dict: Formatted API response
        """
        response = {
            'status': 'success',
            'analysis_id': analysis_id,
            'filename': filename,
            'final_decision': result['final_decision'],
            'processing_time': round(result['processing_time'], 3),
            'timestamp': result['timestamp'],
            'anomaly_detection': {
                'anomaly_score': round(result['anomaly_detection']['anomaly_score'], 4),
                'decision': result['anomaly_detection']['decision'],
                'threshold_used': result['anomaly_detection']['threshold_used'],
                'confidence_level': self._calculate_confidence_level(result)
            },
            'detected_defects': result.get('detected_defect_types', []),
            'defect_count': len(result.get('detected_defect_types', [])),
            'image_url': f"/static/uploads/{filename}"
        }
        
        # Add defect details if available
        if result.get('defect_classification') and result['final_decision'] == 'DEFECT':
            defect_class = result['defect_classification']
            response['defect_details'] = {
                'total_defect_types': len(defect_class['detected_defects']),
                'defect_statistics': defect_class.get('defect_analysis', {}).get('defect_statistics', {}),
                'bounding_boxes_count': sum(
                    len(boxes) for boxes in defect_class.get('defect_analysis', {}).get('bounding_boxes', {}).values()
                )
            }
        
        return response
    
    def _setup_web_routes(self):
        """Setup web interface routes for serving HTML pages"""
        
        @self.app.route('/')
        def dashboard():
            """Serve dashboard HTML page"""
            return send_from_directory('templates', 'dashboard.html')
        
        @self.app.route('/analysis')
        def new_analysis():
            """Serve new analysis HTML page"""
            return send_from_directory('templates', 'analysis.html')
        
        @self.app.route('/history')
        def history():
            """Serve analysis history HTML page"""
            return send_from_directory('templates', 'history.html')
        
        @self.app.route('/realtime')
        def realtime_analysis():
            """Serve real-time analysis HTML page"""
            return send_from_directory('templates', 'realtime_analysis.html')
        
        @self.app.route('/settings')
        def settings():
            """Serve settings HTML page"""
            return send_from_directory('templates', 'settings.html')
        
        @self.app.route('/analysis/<int:analysis_id>')
        def analysis_details(analysis_id):
            """Serve analysis details HTML page"""
            return send_from_directory('templates', 'analysis_details.html')
    
    def run(self, debug=False):
        """
        Start the enhanced API server
        
        Args:
            debug: Enable Flask debug mode
        """
        print(" Starting Enhanced Defect Detection Server")
        print("=" * 60)
        print(f" Dashboard:     http://{self.host}:{self.port}")
        print(f" New Analysis:  http://{self.host}:{self.port}/analysis")
        print(f" History:       http://{self.host}:{self.port}/history")
        print(f"  Settings:      http://{self.host}:{self.port}/settings")
        print(f" Real-time:     http://{self.host}:{self.port}/realtime")
        print("=" * 60)
        print(f"   Health Check:  http://{self.host}:{self.port}/api/health")
        print(f"   Detect Image:  http://{self.host}:{self.port}/api/detect-image")
        print(f"   Dashboard:     http://{self.host}:{self.port}/api/dashboard-stats")
        print(f"   History:       http://{self.host}:{self.port}/api/analysis-history")
        print(f"   Charts:        http://{self.host}:{self.port}/api/dashboard-charts")
        print("=" * 60)
        print("   Real-time Endpoints:")
        print(f"   Start Session: http://{self.host}:{self.port}/api/realtime/start-session")
        print(f"   Process Frame: http://{self.host}:{self.port}/api/realtime/process-frame")
        print(f"   Session Stats: http://{self.host}:{self.port}/api/realtime/session-stats")
        print(f"   Stop Session:  http://{self.host}:{self.port}/api/realtime/stop-session")
        print("=" * 60)
        print(f" Static Files:  http://{self.host}:{self.port}/static/")
        print(f" Database:      {self.db_path}")
        print(f" Detector:      {getattr(self, 'detector_status', 'unknown')}")
        print(f" Performance:   {' Available' if self.has_performance_tracker() else ' Unavailable'}")
        print(f" Real-time:     {' Available' if self.realtime_processor else ' Unavailable'}")
        print("=" * 60)
        
        self.app.run(host=self.host, port=self.port, debug=True, threaded=True, use_reloader=False)

class MockRealtimeProcessor:
    """Mock real-time processor for development/testing"""
    
    def __init__(self):
        self.session_active = False
        self.current_session_id = None
        self.session_start_time = None
        
    def start_session(self):
        """Start mock session"""
        self.session_active = True
        self.current_session_id = f"mock_session_{int(time.time())}"
        self.session_start_time = datetime.now()
        return True
        
    def stop_session(self):
        """Stop mock session"""
        if self.session_active:
            self.session_active = False
            report = {
                'session_id': self.current_session_id,
                'duration': 'Mock session ended',
                'frames_processed': 0,
                'defects_detected': 0
            }
            self.current_session_id = None
            self.session_start_time = None
            return report
        return None
        
    def process_frame(self, frame_data, auto_capture=True):
        """Process mock frame"""
        return {
            'final_decision': 'GOOD',
            'anomaly_detection': {'anomaly_score': 0.1},
            'processing_time': 0.05,
            'detected_defect_types': [],
            'session_id': self.current_session_id,
            'frame_number': 1,
            'realtime_timestamp': datetime.now().isoformat(),
            'realtime_processing_time': 0.05,
            'session_stats': {
                'total_frames': 1,
                'defects_detected': 0,
                'avg_processing_time': 0.05
            }
        }
        
    def get_session_statistics(self):
        """Get mock session stats"""
        if self.session_active:
            return {
                'session_id': self.current_session_id,
                'duration_seconds': 60,
                'total_frames': 10,
                'defects_detected': 1,
                'avg_processing_time': 0.05
            }
        return {'error': 'No active session'}
        
    def capture_manual_screenshot(self, frame_data, detection_result=None):
        """Mock screenshot capture"""
        return {
            'screenshot_id': f"mock_screenshot_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'filename': 'mock_screenshot.jpg'
        }
        
    def get_recent_captures(self, limit=10):
        """Mock recent captures"""
        return []
        
    def get_session_history(self, limit=20):
        """Mock session history"""
        return []
        
    def cleanup_old_sessions(self, days_to_keep=30):
        """Mock cleanup"""
        pass
    
def add_scratch(img):
    """Add scratch defect to image"""
    cv2.line(img, (200, 150), (400, 200), (50, 50, 50), 3)
    cv2.line(img, (220, 160), (380, 190), (60, 60, 60), 2)
    return img

def add_stain(img):
    """Add stain defect to image"""
    cv2.circle(img, (300, 200), 25, (80, 60, 40), -1)
    cv2.circle(img, (320, 180), 15, (90, 70, 50), -1)
    return img

def add_damage(img):
    """Add damage defect to image"""
    pts = np.array([[250, 150], [280, 140], [320, 170], [300, 200], [260, 190]], np.int32)
    cv2.fillPoly(img, [pts], (20, 20, 20))
    return img

def add_missing_part(img):
    """Add missing part defect to image"""
    cv2.rectangle(img, (350, 180), (420, 220), (0, 0, 0), -1)
    cv2.putText(img, "MISSING", (355, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

# Convenience function for easy server creation
def create_enhanced_api_server(host='0.0.0.0', port=5000):
    """
    Create and return enhanced API server instance
    
    Args:
        host: Server host address
        port: Server port number
        
    Returns:
        EnhancedDefectDetectionAPI: Server instance
    """
    return EnhancedDefectDetectionAPI(host=host, port=port)

# Main execution
if __name__ == "__main__":
    # Create and start the enhanced API server
    api_server = create_enhanced_api_server()
    # Run with debug mode for development
    api_server.run(debug=True)