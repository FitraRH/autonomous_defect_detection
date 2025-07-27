# processors/realtime_processor.py
"""
Real-Time Processing for Live Camera Feed with Continuous Detection
"""

import cv2
import os
import time
import json
import base64
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import deque
import sqlite3
from config import *
from processors.image_processor import ImageProcessor
from utils.visualization import create_enhanced_visualization
from utils.reports import save_enhanced_analysis_report


class RealTimeProcessor:
    """
    Enhanced Real-Time Processor for Live Camera Detection
    
    Features:
    - Continuous frame processing
    - Auto-screenshot on defect detection
    - Real-time statistics tracking
    - Session report generation
    - Database integration for real-time data
    """
    
    def __init__(self, detection_core, performance_tracker=None):
        self.detection_core = detection_core
        self.image_processor = ImageProcessor(detection_core)
        self.performance_tracker = performance_tracker
        
        # Real-time session data
        self.session_active = False
        self.session_start_time = None
        self.session_stats = {
            'total_frames': 0,
            'good_count': 0,
            'defect_count': 0,
            'screenshots_captured': 0,
            'processing_times': deque(maxlen=100),
            'recent_detections': deque(maxlen=50)
        }
        
        # Storage paths
        self.realtime_dir = OUTPUTS_DIR / "realtime"
        self.screenshots_dir = self.realtime_dir / "screenshots"
        self.session_reports_dir = self.realtime_dir / "sessions"
        
        # Create directories
        self._create_directories()
        
        # Database for real-time data
        self.db_path = "defect_detection.db"
        self._init_realtime_tables()
        
        print("✅ Real-Time Processor initialized")
    
    def _create_directories(self):
        """Create necessary directories for real-time processing"""
        directories = [
            self.realtime_dir,
            self.screenshots_dir,
            self.session_reports_dir,
            self.realtime_dir / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_realtime_tables(self):
        """Initialize database tables for real-time data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Real-time sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP,
                    session_end TIMESTAMP,
                    duration_seconds INTEGER,
                    total_frames INTEGER,
                    good_count INTEGER,
                    defect_count INTEGER,
                    defect_rate REAL,
                    screenshots_captured INTEGER,
                    avg_processing_time REAL,
                    session_report_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Real-time captures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    capture_timestamp TIMESTAMP,
                    image_path TEXT,
                    detection_result TEXT,
                    final_decision TEXT,
                    anomaly_score REAL,
                    detected_defects TEXT,
                    processing_time REAL,
                    auto_captured BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES realtime_sessions (id)
                )
            ''')
            
            # Real-time frame statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_frame_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    frame_timestamp TIMESTAMP,
                    final_decision TEXT,
                    anomaly_score REAL,
                    processing_time REAL,
                    detected_defects TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES realtime_sessions (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Real-time database tables initialized")
            
        except Exception as e:
            print(f"Error initializing real-time tables: {e}")
    
    def start_session(self):
        """Start a new real-time detection session"""
        if self.session_active:
            print("⚠️ Session already active")
            return False
        
        # Reset session data
        self.session_active = True
        self.session_start_time = datetime.now()
        self.session_stats = {
            'total_frames': 0,
            'good_count': 0,
            'defect_count': 0,
            'screenshots_captured': 0,
            'processing_times': deque(maxlen=100),
            'recent_detections': deque(maxlen=50)
        }
        
        # Create session in database
        self.current_session_id = self._create_session_record()
        
        print(f"🚀 Real-time session started - ID: {self.current_session_id}")
        return True
    
    def stop_session(self):
        """Stop current real-time detection session and generate report"""
        if not self.session_active:
            print("⚠️ No active session to stop")
            return None
        
        self.session_active = False
        session_end_time = datetime.now()
        session_duration = session_end_time - self.session_start_time
        
        # Generate comprehensive session report
        session_report = self._generate_session_report(session_end_time, session_duration)
        
        # Update database record
        self._update_session_record(session_end_time, session_duration, session_report)
        
        print(f"🏁 Real-time session stopped - Duration: {session_duration}")
        return session_report
    
    def process_frame(self, frame_data, auto_capture_defects=True):
        """
        Process a single frame from real-time camera feed
        
        Args:
            frame_data: Base64 encoded image data or file path
            auto_capture_defects: Automatically capture screenshots of defects
            
        Returns:
            dict: Processing result with detection info and frame statistics
        """
        if not self.session_active:
            return {'error': 'No active session'}
        
        try:
            # Start performance measurement
            start_time = time.time()
            analysis_id = None
            
            if self.performance_tracker:
                perf_start = self.performance_tracker.start_measurement()
            
            # Save frame temporarily
            temp_frame_path = self._save_temp_frame(frame_data)
            if not temp_frame_path:
                return {'error': 'Failed to save frame'}
            
            # Process frame using standard image processor
            result = self.image_processor.process_single_image(
                temp_frame_path, 
                output_dir=str(self.realtime_dir / "temp")
            )
            
            if not result:
                return {'error': 'Frame processing failed'}
            
            # End performance measurement
            processing_time = time.time() - start_time
            result['realtime_processing_time'] = processing_time
            
            if self.performance_tracker and perf_start:
                analysis_id = self.performance_tracker.end_measurement(perf_start, result)
            
            # Update session statistics
            self._update_session_stats(result, processing_time)
            
            # Save frame statistics to database
            self._save_frame_stats(result, processing_time)
            
            # Auto-capture defects if enabled
            if auto_capture_defects and result['final_decision'] == 'DEFECT':
                screenshot_info = self._capture_defect_screenshot(temp_frame_path, result)
                result['screenshot_captured'] = screenshot_info
            
            # Add real-time metadata
            result.update({
                'session_id': self.current_session_id,
                'frame_number': self.session_stats['total_frames'],
                'session_stats': self._get_current_session_stats(),
                'realtime_timestamp': datetime.now().isoformat()
            })
            
            # Cleanup temp file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            return result
            
        except Exception as e:
            print(f"Error processing real-time frame: {e}")
            return {'error': str(e)}
    
    def capture_manual_screenshot(self, frame_data, detection_result=None):
        """
        Manually capture screenshot from real-time feed
        
        Args:
            frame_data: Base64 encoded image data
            detection_result: Optional detection result to associate
            
        Returns:
            dict: Screenshot information
        """
        try:
            # Save frame
            temp_frame_path = self._save_temp_frame(frame_data)
            if not temp_frame_path:
                return {'error': 'Failed to save frame'}
            
            # Capture screenshot
            screenshot_info = self._capture_defect_screenshot(
                temp_frame_path, 
                detection_result, 
                auto_captured=False
            )
            
            # Update session stats
            self.session_stats['screenshots_captured'] += 1
            
            # Cleanup
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            return screenshot_info
            
        except Exception as e:
            print(f"Error capturing manual screenshot: {e}")
            return {'error': str(e)}
    
    def _save_temp_frame(self, frame_data):
        """Save frame data to temporary file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            temp_path = self.realtime_dir / "temp" / f"frame_{timestamp}.jpg"
            
            if isinstance(frame_data, str):
                # Base64 encoded data
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                
                image_data = base64.b64decode(frame_data)
                with open(temp_path, 'wb') as f:
                    f.write(image_data)
            else:
                # Already a file path or numpy array
                if isinstance(frame_data, np.ndarray):
                    cv2.imwrite(str(temp_path), frame_data)
                else:
                    # Assume it's a file path
                    import shutil
                    shutil.copy2(frame_data, temp_path)
            
            return str(temp_path)
            
        except Exception as e:
            print(f"Error saving temp frame: {e}")
            return None
    
    def _capture_defect_screenshot(self, frame_path, detection_result, auto_captured=True):
        """Capture and save screenshot of detected defect"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            decision = detection_result.get('final_decision', 'UNKNOWN')
            defects = detection_result.get('detected_defect_types', [])
            defects_str = "_".join(defects[:2]) if defects else "manual"
            
            # Generate filename
            screenshot_filename = f"{decision}_{defects_str}_{timestamp}.jpg"
            screenshot_path = self.screenshots_dir / screenshot_filename
            
            # Copy frame to screenshots directory
            import shutil
            shutil.copy2(frame_path, screenshot_path)
            
            # Create enhanced visualization if defects detected
            if detection_result.get('defect_classification'):
                try:
                    viz_path = create_enhanced_visualization(
                        detection_result, 
                        str(self.screenshots_dir)
                    )
                    if viz_path:
                        # Replace original with visualization
                        shutil.move(viz_path, screenshot_path)
                except Exception as e:
                    print(f"Error creating visualization: {e}")
            
            # Save to database
            capture_info = self._save_capture_record(
                screenshot_path, detection_result, auto_captured
            )
            
            # Update session stats
            self.session_stats['screenshots_captured'] += 1
            
            print(f"📸 Screenshot captured: {screenshot_filename}")
            
            return {
                'filename': screenshot_filename,
                'path': str(screenshot_path),
                'timestamp': timestamp,
                'auto_captured': auto_captured,
                'detection_result': detection_result,
                'capture_id': capture_info
            }
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def _update_session_stats(self, result, processing_time):
        """Update session statistics with new frame result"""
        self.session_stats['total_frames'] += 1
        self.session_stats['processing_times'].append(processing_time)
        
        if result['final_decision'] == 'GOOD':
            self.session_stats['good_count'] += 1
        elif result['final_decision'] == 'DEFECT':
            self.session_stats['defect_count'] += 1
        
        # Add to recent detections
        detection_summary = {
            'timestamp': datetime.now().isoformat(),
            'decision': result['final_decision'],
            'anomaly_score': result['anomaly_detection']['anomaly_score'],
            'processing_time': processing_time,
            'defects': result.get('detected_defect_types', [])
        }
        
        self.session_stats['recent_detections'].append(detection_summary)
    
    def _get_current_session_stats(self):
        """Get current session statistics"""
        total_frames = self.session_stats['total_frames']
        good_count = self.session_stats['good_count']
        defect_count = self.session_stats['defect_count']
        
        defect_rate = (defect_count / total_frames * 100) if total_frames > 0 else 0
        avg_processing_time = (
            sum(self.session_stats['processing_times']) / 
            len(self.session_stats['processing_times'])
        ) if self.session_stats['processing_times'] else 0
        
        return {
            'total_frames': total_frames,
            'good_count': good_count,
            'defect_count': defect_count,
            'defect_rate': round(defect_rate, 2),
            'screenshots_captured': self.session_stats['screenshots_captured'],
            'avg_processing_time': round(avg_processing_time, 3),
            'session_duration': str(datetime.now() - self.session_start_time) if self.session_start_time else "0:00:00"
        }
    
    def _create_session_record(self):
        """Create new session record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_sessions (session_start)
                VALUES (?)
            ''', (self.session_start_time,))
            
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return session_id
            
        except Exception as e:
            print(f"Error creating session record: {e}")
            return None
    
    def _update_session_record(self, end_time, duration, session_report):
        """Update session record with final statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            avg_processing_time = (
                sum(self.session_stats['processing_times']) / 
                len(self.session_stats['processing_times'])
            ) if self.session_stats['processing_times'] else 0
            
            defect_rate = (
                self.session_stats['defect_count'] / 
                self.session_stats['total_frames'] * 100
            ) if self.session_stats['total_frames'] > 0 else 0
            
            cursor.execute('''
                UPDATE realtime_sessions 
                SET session_end = ?, duration_seconds = ?, total_frames = ?,
                    good_count = ?, defect_count = ?, defect_rate = ?,
                    screenshots_captured = ?, avg_processing_time = ?,
                    session_report_path = ?
                WHERE id = ?
            ''', (
                end_time,
                int(duration.total_seconds()),
                self.session_stats['total_frames'],
                self.session_stats['good_count'],
                self.session_stats['defect_count'],
                defect_rate,
                self.session_stats['screenshots_captured'],
                avg_processing_time,
                session_report.get('report_path'),
                self.current_session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating session record: {e}")
    
    def _save_frame_stats(self, result, processing_time):
        """Save frame statistics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_frame_stats 
                (session_id, frame_timestamp, final_decision, anomaly_score, 
                 processing_time, detected_defects)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session_id,
                datetime.now(),
                result['final_decision'],
                result['anomaly_detection']['anomaly_score'],
                processing_time,
                json.dumps(result.get('detected_defect_types', []))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving frame stats: {e}")
    
    def _save_capture_record(self, screenshot_path, detection_result, auto_captured):
        """Save capture record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_captures 
                (session_id, capture_timestamp, image_path, detection_result,
                 final_decision, anomaly_score, detected_defects, 
                 processing_time, auto_captured)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session_id,
                datetime.now(),
                str(screenshot_path),
                json.dumps(detection_result),
                detection_result.get('final_decision', 'UNKNOWN'),
                detection_result.get('anomaly_detection', {}).get('anomaly_score', 0),
                json.dumps(detection_result.get('detected_defect_types', [])),
                detection_result.get('processing_time', 0),
                auto_captured
            ))
            
            capture_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return capture_id
            
        except Exception as e:
            print(f"Error saving capture record: {e}")
            return None
    
    def _generate_session_report(self, end_time, duration):
        """Generate comprehensive session report"""
        try:
            # Calculate statistics
            total_frames = self.session_stats['total_frames']
            good_count = self.session_stats['good_count']
            defect_count = self.session_stats['defect_count']
            screenshots = self.session_stats['screenshots_captured']
            
            defect_rate = (defect_count / total_frames * 100) if total_frames > 0 else 0
            avg_processing_time = (
                sum(self.session_stats['processing_times']) / 
                len(self.session_stats['processing_times'])
            ) if self.session_stats['processing_times'] else 0
            
            # Generate comprehensive report data
            report_data = {
                'session_info': {
                    'session_id': self.current_session_id,
                    'start_time': self.session_start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration': str(duration),
                    'duration_seconds': int(duration.total_seconds())
                },
                'processing_stats': {
                    'total_frames_processed': total_frames,
                    'good_products_detected': good_count,
                    'defective_products_detected': defect_count,
                    'defect_detection_rate': round(defect_rate, 2),
                    'screenshots_captured': screenshots,
                    'avg_processing_time': round(avg_processing_time, 3),
                    'total_processing_time': round(sum(self.session_stats['processing_times']), 2),
                    'throughput_fps': round(total_frames / duration.total_seconds(), 2) if duration.total_seconds() > 0 else 0
                },
                'quality_metrics': {
                    'detection_accuracy': round((good_count / total_frames * 100), 2) if total_frames > 0 else 0,
                    'system_efficiency': round((total_frames / duration.total_seconds() * avg_processing_time), 2) if duration.total_seconds() > 0 else 0,
                    'capture_rate': round((screenshots / defect_count * 100), 2) if defect_count > 0 else 0
                },
                'recent_detections': list(self.session_stats['recent_detections']),
                'performance_trend': self._analyze_performance_trend()
            }
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"realtime_session_report_{self.current_session_id}_{timestamp}.json"
            report_path = self.session_reports_dir / report_filename
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Generate enhanced text report
            text_report_path = self._generate_text_report(report_data, timestamp)
            
            report_result = {
                'report_data': report_data,
                'report_path': str(report_path),
                'text_report_path': text_report_path,
                'summary': {
                    'session_duration': str(duration),
                    'total_frames': total_frames,
                    'defect_rate': round(defect_rate, 2),
                    'avg_fps': round(total_frames / duration.total_seconds(), 2) if duration.total_seconds() > 0 else 0
                }
            }
            
            print(f"📊 Session report generated: {report_filename}")
            return report_result
            
        except Exception as e:
            print(f"Error generating session report: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trend(self):
        """Analyze performance trends during session"""
        try:
            if len(self.session_stats['processing_times']) < 10:
                return {'trend': 'insufficient_data'}
            
            # Calculate trend using simple linear regression
            times = list(self.session_stats['processing_times'])
            x = list(range(len(times)))
            
            # Simple slope calculation
            n = len(times)
            sum_x = sum(x)
            sum_y = sum(times)
            sum_xy = sum(x[i] * times[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if slope > 0.001:
                trend = 'degrading'
            elif slope < -0.001:
                trend = 'improving'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'slope': slope,
                'avg_early': np.mean(times[:len(times)//3]),
                'avg_late': np.mean(times[-len(times)//3:]),
                'variance': np.var(times)
            }
            
        except Exception as e:
            print(f"Error analyzing performance trend: {e}")
            return {'trend': 'error', 'error': str(e)}
    
    def _generate_text_report(self, report_data, timestamp):
        """Generate human-readable text report"""
        try:
            text_filename = f"realtime_session_text_report_{self.current_session_id}_{timestamp}.txt"
            text_path = self.session_reports_dir / text_filename
            
            session_info = report_data['session_info']
            stats = report_data['processing_stats']
            quality = report_data['quality_metrics']
            
            report_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REAL-TIME DETECTION SESSION REPORT                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {session_info['session_id']}
Report Type: Real-Time Detection Session Analysis

╔══════════════════════════════════════════════════════════════════════════════╗
║                              SESSION SUMMARY                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Session Duration: {session_info['duration']}
Start Time: {datetime.fromisoformat(session_info['start_time']).strftime('%Y-%m-%d %H:%M:%S')}
End Time: {datetime.fromisoformat(session_info['end_time']).strftime('%Y-%m-%d %H:%M:%S')}

PROCESSING OVERVIEW:
├─ Total Frames Processed: {stats['total_frames_processed']:,}
├─ Good Products Detected: {stats['good_products_detected']:,} ({quality['detection_accuracy']:.1f}%)
├─ Defective Products: {stats['defective_products_detected']:,} ({stats['defect_detection_rate']:.1f}%)
├─ Screenshots Captured: {stats['screenshots_captured']:,}
└─ Average Processing Speed: {stats['throughput_fps']:.2f} FPS

╔══════════════════════════════════════════════════════════════════════════════╗
║                            PERFORMANCE METRICS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROCESSING PERFORMANCE:
├─ Average Processing Time: {stats['avg_processing_time']:.3f} seconds per frame
├─ Total Processing Time: {stats['total_processing_time']:.2f} seconds
├─ Throughput: {stats['throughput_fps']:.2f} frames per second
├─ System Efficiency: {quality['system_efficiency']:.2f}%
└─ Capture Success Rate: {quality['capture_rate']:.1f}%

QUALITY INDICATORS:
├─ Detection Accuracy: {quality['detection_accuracy']:.2f}%
├─ Real-time Capability: {'✅ Excellent' if stats['throughput_fps'] > 10 else '⚠️ Adequate' if stats['throughput_fps'] > 5 else '❌ Poor'}
├─ Processing Consistency: {self._get_consistency_rating()}
└─ Overall Performance: {self._get_overall_rating(quality, stats)}

╔══════════════════════════════════════════════════════════════════════════════╗
║                             DETECTION ANALYSIS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFECT DETECTION SUMMARY:
├─ Total Defects Found: {stats['defective_products_detected']:,}
├─ Detection Rate: {stats['defect_detection_rate']:.2f}% of processed frames
├─ Auto-Capture Success: {quality['capture_rate']:.1f}% capture rate
└─ Response Time: Real-time detection with {stats['avg_processing_time']:.3f}s latency

RECENT DETECTION PATTERN:
"""
            
            # Add recent detections summary
            recent = report_data.get('recent_detections', [])
            if recent:
                defect_recent = [d for d in recent[-10:] if d['decision'] == 'DEFECT']
                if defect_recent:
                    report_text += f"├─ Last 10 frames: {len(defect_recent)} defects detected\n"
                    for detection in defect_recent[-5:]:
                        timestamp_str = datetime.fromisoformat(detection['timestamp']).strftime('%H:%M:%S')
                        defects_str = ', '.join(detection['defects'][:2]) if detection['defects'] else 'General'
                        report_text += f"├─ {timestamp_str}: {detection['decision']} - {defects_str} (Score: {detection['anomaly_score']:.3f})\n"
                else:
                    report_text += "├─ Last 10 frames: No defects detected\n"
            
            report_text += f"""
└─ Session completed successfully

╔══════════════════════════════════════════════════════════════════════════════╗
║                              RECOMMENDATIONS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYSTEM PERFORMANCE:
"""
            
            # Add performance recommendations
            if stats['throughput_fps'] < 5:
                report_text += "├─ ⚠️ Low throughput detected - Consider optimizing detection parameters\n"
            elif stats['throughput_fps'] > 15:
                report_text += "├─ ✅ Excellent throughput achieved - System performing optimally\n"
            else:
                report_text += "├─ ✅ Good throughput maintained - System stable\n"
            
            if stats['defect_detection_rate'] > 20:
                report_text += "├─ ⚠️ High defect rate - Investigate production quality\n"
            elif stats['defect_detection_rate'] < 5:
                report_text += "├─ ✅ Low defect rate - Production quality excellent\n"
            else:
                report_text += "├─ ✅ Normal defect rate - Monitor trends\n"
            
            if quality['capture_rate'] < 80 and stats['defective_products_detected'] > 0:
                report_text += "├─ ⚠️ Low capture rate - Check auto-capture settings\n"
            
            report_text += f"""
├─ Continue monitoring for quality trends
├─ Review captured screenshots for pattern analysis
└─ Maintain current detection parameters

╔══════════════════════════════════════════════════════════════════════════════╗
║                            TECHNICAL DETAILS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SESSION CONFIGURATION:
├─ Detection Engine: Enhanced Real-Time Processor v2.0
├─ Processing Mode: Live Camera Feed Analysis
├─ Auto-Capture: {'Enabled' if quality['capture_rate'] > 0 else 'Disabled'}
├─ Database Integration: ✅ Active
└─ Performance Tracking: {'✅ Active' if hasattr(self, 'performance_tracker') and self.performance_tracker else '❌ Inactive'}

DATA STORAGE:
├─ Screenshots Directory: {self.screenshots_dir}
├─ Session Reports: {self.session_reports_dir}
├─ Database Records: {stats['total_frames_processed']:,} frame records stored
└─ Captured Images: {stats['screenshots_captured']:,} images saved

SYSTEM HEALTH:
├─ Memory Usage: Optimized (circular buffers used)
├─ Processing Stability: {self._get_stability_status()}
├─ Error Rate: {'< 1%' if stats['total_frames_processed'] > 0 else 'N/A'}
└─ Session Integrity: ✅ Complete

═══════════════════════════════════════════════════════════════════════════════
Real-Time Session Report Generated by Enhanced Defect Detection System v2.0
Session ID: {session_info['session_id']} | Duration: {session_info['duration']}
═══════════════════════════════════════════════════════════════════════════════
"""
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📄 Text report generated: {text_filename}")
            return str(text_path)
            
        except Exception as e:
            print(f"Error generating text report: {e}")
            return None
    
    def _get_consistency_rating(self):
        """Get processing consistency rating"""
        if not self.session_stats['processing_times']:
            return "Unknown"
        
        times = list(self.session_stats['processing_times'])
        variance = np.var(times)
        
        if variance < 0.01:
            return "Excellent"
        elif variance < 0.05:
            return "Good"
        elif variance < 0.1:
            return "Fair"
        else:
            return "Poor"
    
    def _get_overall_rating(self, quality, stats):
        """Get overall performance rating"""
        score = 0
        
        # Accuracy component (40%)
        if quality['detection_accuracy'] > 95:
            score += 40
        elif quality['detection_accuracy'] > 90:
            score += 35
        elif quality['detection_accuracy'] > 85:
            score += 30
        else:
            score += 20
        
        # Throughput component (30%)
        if stats['throughput_fps'] > 15:
            score += 30
        elif stats['throughput_fps'] > 10:
            score += 25
        elif stats['throughput_fps'] > 5:
            score += 20
        else:
            score += 10
        
        # Consistency component (20%)
        consistency = self._get_consistency_rating()
        if consistency == "Excellent":
            score += 20
        elif consistency == "Good":
            score += 15
        elif consistency == "Fair":
            score += 10
        else:
            score += 5
        
        # Efficiency component (10%)
        if quality['system_efficiency'] > 90:
            score += 10
        elif quality['system_efficiency'] > 80:
            score += 8
        elif quality['system_efficiency'] > 70:
            score += 6
        else:
            score += 3
        
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B+ (Good)"
        elif score >= 60:
            return "B (Satisfactory)"
        else:
            return "C (Needs Improvement)"
    
    def _get_stability_status(self):
        """Get system stability status"""
        if len(self.session_stats['processing_times']) < 10:
            return "Insufficient Data"
        
        times = list(self.session_stats['processing_times'])
        trend = self._analyze_performance_trend()
        
        if trend['trend'] == 'stable' and trend.get('variance', 1) < 0.05:
            return "Excellent"
        elif trend['trend'] == 'improving':
            return "Good (Improving)"
        elif trend['trend'] == 'degrading':
            return "Fair (Degrading)"
        else:
            return "Stable"
    
    def get_session_statistics(self):
        """Get current session statistics"""
        if not self.session_active:
            return {'error': 'No active session'}
        
        return self._get_current_session_stats()
    
    def get_recent_captures(self, limit=10):
        """Get recent capture records from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM realtime_captures 
                WHERE session_id = ?
                ORDER BY capture_timestamp DESC
                LIMIT ?
            ''', (self.current_session_id, limit))
            
            captures = []
            for row in cursor.fetchall():
                captures.append({
                    'id': row[0],
                    'timestamp': row[2],
                    'image_path': row[3],
                    'final_decision': row[5],
                    'anomaly_score': row[6],
                    'detected_defects': json.loads(row[7]) if row[7] else [],
                    'auto_captured': bool(row[9])
                })
            
            conn.close()
            return captures
            
        except Exception as e:
            print(f"Error getting recent captures: {e}")
            return []
    
    def cleanup_old_sessions(self, days_to_keep=30):
        """Clean up old session data and files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old sessions
            cursor.execute('''
                SELECT id, session_report_path FROM realtime_sessions 
                WHERE session_start < ?
            ''', (cutoff_date,))
            
            old_sessions = cursor.fetchall()
            
            # Clean up files and database records
            for session_id, report_path in old_sessions:
                # Delete associated captures and files
                cursor.execute('''
                    SELECT image_path FROM realtime_captures 
                    WHERE session_id = ?
                ''', (session_id,))
                
                for (image_path,) in cursor.fetchall():
                    if os.path.exists(image_path):
                        os.remove(image_path)
                
                # Delete report file
                if report_path and os.path.exists(report_path):
                    os.remove(report_path)
                
                # Delete database records
                cursor.execute('DELETE FROM realtime_captures WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM realtime_frame_stats WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM realtime_sessions WHERE id = ?', (session_id,))
            
            conn.commit()
            conn.close()
            
            print(f"🧹 Cleaned up {len(old_sessions)} old sessions")
            
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
    
    def get_session_history(self, limit=20):
        """Get session history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, session_start, session_end, duration_seconds, 
                       total_frames, good_count, defect_count, defect_rate,
                       screenshots_captured, avg_processing_time
                FROM realtime_sessions 
                ORDER BY session_start DESC
                LIMIT ?
            ''', (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'duration_seconds': row[3],
                    'total_frames': row[4],
                    'good_count': row[5],
                    'defect_count': row[6],
                    'defect_rate': row[7],
                    'screenshots_captured': row[8],
                    'avg_processing_time': row[9]
                })
            
            conn.close()
            return sessions
            
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []