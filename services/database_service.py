# services/database_service.py
"""
Database Service - Data Access Layer
Handles all database operations and data persistence
Uses existing database schema and structure
"""

import sqlite3
import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


class DatabaseService:
    """
    Database Service for handling all data persistence operations
    Integrates with existing database schema from api_server.py
    """
    
    def __init__(self, db_path="defect_detection.db"):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and tables exist using existing schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use existing table creation from your api_server.py
            self._create_existing_tables(cursor)
            
            conn.commit()
            conn.close()
            print("Database initialized with existing schema")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def _create_existing_tables(self, cursor):
        """Create tables using existing schema from your project"""
        
        # Main analyses table (from your existing api_server.py)
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
        
        # Defect statistics table (from your existing api_server.py)
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
        
        # System statistics table (from your existing api_server.py)
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
        
        # User settings table (from your existing api_server.py)
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
        
        # Performance metrics table (from your existing api_server.py)
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
    
    def save_analysis(self, result):
        """Save analysis result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            image_name = os.path.basename(result.get('image_path', 'unknown'))
            image_size = result.get('image_size', 'unknown')
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
                str(image_size),
                result.get('final_decision', 'Unknown'),
                result.get('anomaly_detection', {}).get('anomaly_score', 0.0),
                confidence_level,
                json.dumps(result.get('detected_defect_types', [])),
                result.get('processing_time', 0.0),
                'Completed',
                result.get('visualization_path', '')
            ))
            
            analysis_id = cursor.lastrowid
            
            # Save defect statistics if available
            if result.get('defect_classification') and result['final_decision'] == 'DEFECT':
                self._save_defect_statistics(cursor, analysis_id, result['defect_classification'])
            
            conn.commit()
            conn.close()
            
            return analysis_id
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return None
    
    def save_batch_results(self, results):
        """Save batch processing results"""
        try:
            batch_id = f"batch_{int(datetime.now().timestamp())}"
            
            for result in results:
                if result:
                    # Add batch identifier to result
                    result['batch_id'] = batch_id
                    self.save_analysis(result)
            
            return batch_id
            
        except Exception as e:
            print(f"Error saving batch results: {e}")
            return None
    
    def save_video_analysis(self, result):
        """Save video analysis results"""
        try:
            video_id = f"video_{int(datetime.now().timestamp())}"
            
            # Save video summary as special analysis record
            video_summary = {
                'image_path': result.get('video_path', ''),
                'final_decision': 'VIDEO_ANALYSIS',
                'anomaly_detection': {'anomaly_score': 0.0},
                'detected_defect_types': [],
                'processing_time': result.get('summary', {}).get('total_processing_time', 0),
                'video_id': video_id
            }
            
            analysis_id = self.save_analysis(video_summary)
            
            return video_id
            
        except Exception as e:
            print(f"Error saving video analysis: {e}")
            return None
    
    def get_analysis_history(self, page=1, per_page=20, filters=None):
        """Get paginated analysis history with optional filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with filters
            base_query = '''
                SELECT id, image_name, analysis_date, detected_defects, final_decision, 
                       status, anomaly_score, processing_time, confidence_level
                FROM analyses
            '''
            
            where_conditions = []
            params = []
            
            if filters:
                if filters.get('decision'):
                    where_conditions.append('final_decision = ?')
                    params.append(filters['decision'])
                
                if filters.get('date_from'):
                    where_conditions.append('DATE(analysis_date) >= ?')
                    params.append(filters['date_from'])
                
                if filters.get('date_to'):
                    where_conditions.append('DATE(analysis_date) <= ?')
                    params.append(filters['date_to'])
                
                if filters.get('defect_type'):
                    where_conditions.append('detected_defects LIKE ?')
                    params.append(f'%{filters["defect_type"]}%')
            
            if where_conditions:
                base_query += ' WHERE ' + ' AND '.join(where_conditions)
            
            # Add ordering and pagination
            base_query += ' ORDER BY analysis_date DESC LIMIT ? OFFSET ?'
            params.extend([per_page, (page - 1) * per_page])
            
            cursor.execute(base_query, params)
            
            analyses = []
            for row in cursor.fetchall():
                defects = json.loads(row[3]) if row[3] else []
                analyses.append({
                    'id': row[0],
                    'image_name': row[1],
                    'analysis_date': row[2],
                    'defects': defects,
                    'defect_count': len(defects),
                    'final_decision': row[4],
                    'status': row[5],
                    'anomaly_score': row[6],
                    'processing_time': row[7],
                    'confidence_level': row[8]
                })
            
            # Get total count for pagination
            count_query = 'SELECT COUNT(*) FROM analyses'
            if where_conditions:
                count_query += ' WHERE ' + ' AND '.join(where_conditions)
            
            cursor.execute(count_query, params[:-2])  # Remove LIMIT/OFFSET params
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'analyses': analyses,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total_count,
                    'pages': (total_count + per_page - 1) // per_page
                }
            }
            
        except Exception as e:
            print(f"Error getting analysis history: {e}")
            return {'analyses': [], 'pagination': {'page': 1, 'per_page': per_page, 'total': 0, 'pages': 0}}
    
    def get_analysis_by_id(self, analysis_id):
        """Get specific analysis by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                analysis = dict(zip(columns, row))
                conn.close()
                return analysis
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"Error getting analysis by ID: {e}")
            return None
    
    def get_defect_statistics(self, analysis_id):
        """Get defect statistics for specific analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM defect_statistics WHERE analysis_id = ?', (analysis_id,))
            rows = cursor.fetchall()
            
            statistics = []
            if rows:
                columns = [description[0] for description in cursor.description]
                for row in rows:
                    statistics.append(dict(zip(columns, row)))
            
            conn.close()
            return statistics
            
        except Exception as e:
            print(f"Error getting defect statistics: {e}")
            return []
    
    def delete_analysis(self, analysis_id):
        """Delete analysis and related records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete related records first
            cursor.execute('DELETE FROM defect_statistics WHERE analysis_id = ?', (analysis_id,))
            cursor.execute('DELETE FROM performance_metrics WHERE analysis_id = ?', (analysis_id,))
            
            # Delete main analysis record
            cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return success
            
        except Exception as e:
            print(f"Error deleting analysis: {e}")
            return False
    
    def get_comprehensive_statistics(self):
        """Get comprehensive system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects_detected,
                    SUM(CASE WHEN final_decision = 'GOOD' THEN 1 ELSE 0 END) as good_products,
                    AVG(processing_time) as avg_processing_time,
                    AVG(anomaly_score) as avg_anomaly_score
                FROM analyses
                WHERE analysis_date >= DATE('now', '-30 days')
            ''')
            
            basic_stats = cursor.fetchone()
            
            # Defect type distribution
            cursor.execute('''
                SELECT defect_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM defect_statistics ds
                JOIN analyses a ON ds.analysis_id = a.id
                WHERE a.analysis_date >= DATE('now', '-30 days')
                GROUP BY defect_type
                ORDER BY count DESC
            ''')
            
            defect_distribution = cursor.fetchall()
            
            # Daily statistics for last 7 days
            cursor.execute('''
                SELECT 
                    DATE(analysis_date) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN final_decision = 'DEFECT' THEN 1 ELSE 0 END) as defects,
                    AVG(processing_time) as avg_time
                FROM analyses
                WHERE analysis_date >= DATE('now', '-7 days')
                GROUP BY DATE(analysis_date)
                ORDER BY date DESC
            ''')
            
            daily_stats = cursor.fetchall()
            
            conn.close()
            
            # Format response
            statistics = {
                'overview': {
                    'total_analyses': basic_stats[0] or 0,
                    'defects_detected': basic_stats[1] or 0,
                    'good_products': basic_stats[2] or 0,
                    'defect_rate': (basic_stats[1] / basic_stats[0] * 100) if basic_stats[0] > 0 else 0,
                    'avg_processing_time': basic_stats[3] or 0,
                    'avg_anomaly_score': basic_stats[4] or 0
                },
                'defect_distribution': [
                    {
                        'defect_type': row[0],
                        'count': row[1],
                        'avg_confidence': row[2] or 0
                    }
                    for row in defect_distribution
                ],
                'daily_trends': [
                    {
                        'date': row[0],
                        'total_analyses': row[1],
                        'defects_found': row[2],
                        'avg_processing_time': row[3] or 0,
                        'defect_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                    }
                    for row in daily_stats
                ],
                'generated_at': datetime.now().isoformat(),
                'period': 'last_30_days'
            }
            
            return statistics
            
        except Exception as e:
            print(f"Error getting comprehensive statistics: {e}")
            return {
                'overview': {},
                'defect_distribution': [],
                'daily_trends': [],
                'error': str(e)
            }
    
    def export_analysis_data(self, format='json', date_range=None, include_details=False):
        """Export analysis data in specified format"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on parameters
            query = 'SELECT * FROM analyses'
            params = []
            
            if date_range:
                conditions = []
                if date_range.get('start_date'):
                    conditions.append('DATE(analysis_date) >= ?')
                    params.append(date_range['start_date'])
                if date_range.get('end_date'):
                    conditions.append('DATE(analysis_date) <= ?')
                    params.append(date_range['end_date'])
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY analysis_date DESC'
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            # Include detailed defect statistics if requested
            if include_details and not df.empty:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ds.*, a.id as analysis_id 
                    FROM defect_statistics ds
                    JOIN analyses a ON ds.analysis_id = a.id
                ''')
                
                defect_stats = cursor.fetchall()
                # Process defect statistics (implementation depends on specific needs)
            
            conn.close()
            
            # Generate export file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            if format == 'json':
                filename = f"analysis_export_{timestamp}.json"
                filepath = export_dir / filename
                df.to_json(filepath, orient='records', date_format='iso')
                
            elif format == 'csv':
                filename = f"analysis_export_{timestamp}.csv"
                filepath = export_dir / filename
                df.to_csv(filepath, index=False)
                
            elif format == 'xlsx':
                filename = f"analysis_export_{timestamp}.xlsx"
                filepath = export_dir / filename
                df.to_excel(filepath, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Get file size
            file_size = filepath.stat().st_size
            
            return {
                'file_path': str(filepath),
                'record_count': len(df),
                'file_size': file_size,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error exporting analysis data: {e}")
            raise
    
    def get_system_settings(self):
        """Get current system settings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT anomaly_threshold, defect_threshold, notification_enabled, 
                       auto_save_results, preferred_format, updated_at
                FROM user_settings WHERE user_id = ?
            ''', ('default',))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'anomaly_threshold': row[0],
                    'defect_threshold': row[1],
                    'notification_enabled': bool(row[2]),
                    'auto_save_results': bool(row[3]),
                    'preferred_format': row[4],
                    'last_updated': row[5]
                }
            else:
                # Return default settings
                return {
                    'anomaly_threshold': 0.7,
                    'defect_threshold': 0.85,
                    'notification_enabled': True,
                    'auto_save_results': True,
                    'preferred_format': 'JSON',
                    'last_updated': None
                }
                
        except Exception as e:
            print(f"Error getting system settings: {e}")
            return {}
    
    def update_system_settings(self, new_settings):
        """Update system settings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_settings 
                (user_id, anomaly_threshold, defect_threshold, notification_enabled, 
                 auto_save_results, preferred_format, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                'default',
                new_settings.get('anomaly_threshold', 0.7),
                new_settings.get('defect_threshold', 0.85),
                new_settings.get('notification_enabled', True),
                new_settings.get('auto_save_results', True),
                new_settings.get('preferred_format', 'JSON')
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error updating system settings: {e}")
            return False
    
    def _save_defect_statistics(self, cursor, analysis_id, defect_classification):
        """Save detailed defect statistics (from existing api_server.py logic)"""
        try:
            defect_stats = defect_classification.get('defect_analysis', {})
            
            for defect_type, stats in defect_stats.get('defect_statistics', {}).items():
                # Calculate severity based on area and confidence
                class_dist = defect_stats.get('class_distribution', {}).get(defect_type, {})
                area_pct = class_dist.get('percentage', 0.0)
                confidence = stats.get('avg_confidence', 0.0)
                severity = self._calculate_severity(area_pct, confidence)
                
                # Get location data
                bboxes = defect_stats.get('bounding_boxes', {}).get(defect_type, [])
                location_data = json.dumps([{
                    'x': bbox.get('x', 0), 'y': bbox.get('y', 0), 
                    'width': bbox.get('width', 0), 'height': bbox.get('height', 0)
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
                
        except Exception as e:
            print(f"Error saving defect statistics: {e}")
    
    def _calculate_confidence_level(self, result):
        """Calculate confidence level from result (from existing api_server.py logic)"""
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
    
    def _calculate_severity(self, area_percentage, confidence):
        """Calculate defect severity level (from existing api_server.py logic)"""
        if area_percentage > 10 and confidence > 0.8:
            return "Critical"
        elif area_percentage > 5 and confidence > 0.7:
            return "High"
        elif area_percentage > 2 and confidence > 0.6:
            return "Medium"
        else:
            return "Low"