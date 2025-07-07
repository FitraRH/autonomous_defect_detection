import time
import psutil
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sqlite3
from collections import defaultdict, deque

class EnhancedPerformanceTracker:
    """Enhanced performance tracking with comprehensive metrics and chart data generation"""
    
    def __init__(self, db_path="defect_detection.db"):
        self.db_path = db_path
        self.reset()
        self.init_performance_tables()
    
    def reset(self):
        """Reset all metrics"""
        self.start_times = []
        self.end_times = []
        self.processing_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.decisions = []
        self.anomaly_scores = []
        self.defect_types = []
        self.image_sizes = []
        self.confidence_scores = []
        
        # Real-time metrics (circular buffers for efficiency)
        self.realtime_processing_times = deque(maxlen=100)
        self.realtime_memory = deque(maxlen=100)
        self.realtime_cpu = deque(maxlen=100)
        self.realtime_timestamps = deque(maxlen=100)
    
    def init_performance_tables(self):
        """Initialize performance tracking tables in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    processing_time REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    image_size TEXT,
                    anomaly_score REAL,
                    decision TEXT,
                    defect_types TEXT,
                    confidence_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id)
                )
            ''')
            
            # Daily performance summary
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_analyses INTEGER,
                    avg_processing_time REAL,
                    max_processing_time REAL,
                    min_processing_time REAL,
                    avg_memory_usage REAL,
                    avg_cpu_usage REAL,
                    defect_rate REAL,
                    avg_confidence REAL,
                    throughput_per_hour REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System performance history
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
            print("✅ Performance tracking tables initialized")
            
        except Exception as e:
            print(f"Error initializing performance tables: {e}")
    
    def start_measurement(self, analysis_id=None):
        """Start timing measurement with enhanced system monitoring"""
        start_time = time.time()
        self.start_times.append(start_time)
        
        # Record enhanced system metrics
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            self.memory_usage.append(memory.percent)
            self.cpu_usage.append(cpu_percent)
            
            # Real-time tracking
            self.realtime_memory.append(memory.percent)
            self.realtime_cpu.append(cpu_percent)
            self.realtime_timestamps.append(start_time)
            
            # Log system performance to database
            self._log_system_performance(memory, cpu_percent)
            
        except Exception as e:
            print(f"Error recording system metrics: {e}")
            self.memory_usage.append(0)
            self.cpu_usage.append(0)
        
        return start_time
    
    def end_measurement(self, start_time, result=None, analysis_id=None):
        """End timing measurement with comprehensive result logging"""
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.end_times.append(end_time)
        self.processing_times.append(processing_time)
        self.realtime_processing_times.append(processing_time)
        
        # Enhanced result processing
        if result:
            decision = result.get('final_decision', 'Unknown')
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0)
            defect_types = result.get('detected_defect_types', [])
            
            self.decisions.append(decision)
            self.anomaly_scores.append(anomaly_score)
            self.defect_types.append(defect_types)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(result)
            self.confidence_scores.append(confidence)
            
            # Log to database
            self._log_performance_metric(analysis_id, processing_time, anomaly_score, 
                                       decision, defect_types, confidence)
        
        return processing_time
    
    def _calculate_confidence_score(self, result):
        """Calculate overall confidence score from analysis result"""
        try:
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0)
            
            if result.get('final_decision') == 'GOOD':
                # For good products, higher confidence when anomaly score is low
                confidence = max(0.7, 1.0 - anomaly_score)
            else:
                # For defects, confidence based on anomaly score and defect classification
                confidence = min(0.95, anomaly_score + 0.1)
                
                # Adjust based on number of defects found
                defect_count = len(result.get('detected_defect_types', []))
                if defect_count > 0:
                    confidence = min(0.98, confidence + (defect_count * 0.05))
            
            return confidence
            
        except Exception:
            return 0.8  # Default confidence
    
    def _log_system_performance(self, memory, cpu_percent):
        """Log system performance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_performance 
                (memory_total, memory_used, memory_percent, cpu_percent, 
                 disk_usage, active_processes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                memory.total / (1024**3),  # GB
                memory.used / (1024**3),   # GB
                memory.percent,
                cpu_percent,
                psutil.disk_usage('/').percent,
                len(psutil.pids())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging system performance: {e}")
    
    def _log_performance_metric(self, analysis_id, processing_time, anomaly_score, 
                              decision, defect_types, confidence):
        """Log individual performance metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (analysis_id, processing_time, memory_usage, cpu_usage, 
                 anomaly_score, decision, defect_types, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                processing_time,
                self.memory_usage[-1] if self.memory_usage else 0,
                self.cpu_usage[-1] if self.cpu_usage else 0,
                anomaly_score,
                decision,
                json.dumps(defect_types),
                confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging performance metric: {e}")
    
    def get_enhanced_metrics(self):
        """Get comprehensive performance metrics with statistical analysis"""
        if not self.processing_times:
            return self._get_empty_metrics()
        
        times = np.array(self.processing_times)
        
        # Basic metrics
        metrics = {
            'basic_stats': {
                'total_tests': len(times),
                'avg_processing_time': float(np.mean(times)),
                'min_processing_time': float(np.min(times)),
                'max_processing_time': float(np.max(times)),
                'std_processing_time': float(np.std(times)),
                'median_processing_time': float(np.median(times)),
                'percentile_95': float(np.percentile(times, 95)),
                'percentile_99': float(np.percentile(times, 99))
            },
            
            'throughput_metrics': {
                'avg_throughput_fps': float(1 / np.mean(times)),
                'max_throughput_fps': float(1 / np.min(times)),
                'total_processing_time': float(np.sum(times)),
                'total_runtime': float(max(self.end_times) - min(self.start_times)) if self.start_times else 0
            },
            
            'system_metrics': {
                'avg_memory_usage': float(np.mean(self.memory_usage)) if self.memory_usage else 0,
                'max_memory_usage': float(np.max(self.memory_usage)) if self.memory_usage else 0,
                'avg_cpu_usage': float(np.mean(self.cpu_usage)) if self.cpu_usage else 0,
                'max_cpu_usage': float(np.max(self.cpu_usage)) if self.cpu_usage else 0
            },
            
            'quality_metrics': {
                'good_products': self.decisions.count('GOOD'),
                'defective_products': self.decisions.count('DEFECT'),
                'error_count': self.decisions.count('ERROR'),
                'avg_anomaly_score': float(np.mean(self.anomaly_scores)) if self.anomaly_scores else 0,
                'avg_confidence': float(np.mean(self.confidence_scores)) if self.confidence_scores else 0,
                'defect_detection_rate': len([d for d in self.defect_types if d]) / len(self.defect_types) if self.defect_types else 0
            }
        }
        
        # Calculate rates
        total = metrics['basic_stats']['total_tests']
        if total > 0:
            metrics['quality_metrics']['good_rate'] = metrics['quality_metrics']['good_products'] / total * 100
            metrics['quality_metrics']['defect_rate'] = metrics['quality_metrics']['defective_products'] / total * 100
            metrics['quality_metrics']['error_rate'] = metrics['quality_metrics']['error_count'] / total * 100
        
        # Performance trends
        metrics['trends'] = self._calculate_performance_trends()
        
        return metrics
    
    def _calculate_performance_trends(self):
        """Calculate performance trends over time"""
        trends = {
            'processing_time_trend': 'stable',
            'memory_trend': 'stable',
            'quality_trend': 'stable',
            'improvement_suggestions': []
        }
        
        if len(self.processing_times) >= 10:
            # Calculate trends using linear regression
            x = np.arange(len(self.processing_times))
            
            # Processing time trend
            time_slope = np.polyfit(x, self.processing_times, 1)[0]
            if time_slope > 0.01:
                trends['processing_time_trend'] = 'degrading'
                trends['improvement_suggestions'].append('Processing time increasing - consider model optimization')
            elif time_slope < -0.01:
                trends['processing_time_trend'] = 'improving'
            
            # Memory trend
            if self.memory_usage:
                memory_slope = np.polyfit(x[-len(self.memory_usage):], self.memory_usage, 1)[0]
                if memory_slope > 1:
                    trends['memory_trend'] = 'increasing'
                    trends['improvement_suggestions'].append('Memory usage increasing - check for memory leaks')
                elif memory_slope < -1:
                    trends['memory_trend'] = 'decreasing'
            
            # Quality trend (confidence scores)
            if self.confidence_scores:
                conf_slope = np.polyfit(x[-len(self.confidence_scores):], self.confidence_scores, 1)[0]
                if conf_slope > 0.01:
                    trends['quality_trend'] = 'improving'
                elif conf_slope < -0.01:
                    trends['quality_trend'] = 'degrading'
                    trends['improvement_suggestions'].append('Detection confidence decreasing - model retraining may be needed')
        
        return trends
    
    def get_chart_data_for_dashboard(self):
        """Generate chart data specifically for dashboard display"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            chart_data = {}
            
            # 1. Performance trend over last 30 days
            cursor.execute('''
                SELECT DATE(timestamp) as date, 
                       AVG(processing_time) as avg_time,
                       COUNT(*) as count,
                       AVG(memory_usage) as avg_memory,
                       AVG(cpu_usage) as avg_cpu
                FROM performance_metrics 
                WHERE timestamp >= DATE('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''')
            
            perf_data = cursor.fetchall()
            chart_data['performance_trend'] = {
                'labels': [row[0] for row in perf_data],
                'datasets': [
                    {
                        'label': 'Avg Processing Time (s)',
                        'data': [row[1] for row in perf_data],
                        'borderColor': '#3b82f6',
                        'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                        'tension': 0.4,
                        'yAxisID': 'y'
                    },
                    {
                        'label': 'Memory Usage (%)',
                        'data': [row[3] for row in perf_data],
                        'borderColor': '#ef4444',
                        'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                        'tension': 0.4,
                        'yAxisID': 'y1'
                    }
                ]
            }
            
            # 2. Hourly performance pattern (last 24 hours)
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour,
                       AVG(processing_time) as avg_time,
                       COUNT(*) as count
                FROM performance_metrics 
                WHERE timestamp >= DATETIME('now', '-24 hours')
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''')
            
            hourly_data = cursor.fetchall()
            chart_data['hourly_pattern'] = {
                'labels': [f"{row[0]}:00" for row in hourly_data],
                'datasets': [{
                    'label': 'Analyses per Hour',
                    'data': [row[2] for row in hourly_data],
                    'backgroundColor': 'rgba(16, 185, 129, 0.8)',
                    'borderColor': '#10b981',
                    'borderWidth': 2
                }]
            }
            
            # 3. Quality metrics distribution
            cursor.execute('''
                SELECT decision, COUNT(*) as count,
                       AVG(confidence_score) as avg_confidence
                FROM performance_metrics 
                WHERE timestamp >= DATE('now', '-7 days')
                GROUP BY decision
            ''')
            
            quality_data = cursor.fetchall()
            chart_data['quality_distribution'] = {
                'labels': [row[0] for row in quality_data],
                'datasets': [{
                    'data': [row[1] for row in quality_data],
                    'backgroundColor': [
                        '#10b981' if row[0] == 'GOOD' else '#ef4444' if row[0] == 'DEFECT' else '#6b7280'
                        for row in quality_data
                    ]
                }]
            }
            
            # 4. System resource usage trend
            cursor.execute('''
                SELECT datetime(timestamp) as time,
                       memory_percent,
                       cpu_percent
                FROM system_performance 
                WHERE timestamp >= DATETIME('now', '-2 hours')
                ORDER BY timestamp
            ''')
            
            resource_data = cursor.fetchall()
            chart_data['resource_usage'] = {
                'labels': [row[0][-8:] for row in resource_data],  # Last 8 chars (HH:MM:SS)
                'datasets': [
                    {
                        'label': 'Memory %',
                        'data': [row[1] for row in resource_data],
                        'borderColor': '#8b5cf6',
                        'backgroundColor': 'rgba(139, 92, 246, 0.1)',
                        'tension': 0.4
                    },
                    {
                        'label': 'CPU %',
                        'data': [row[2] for row in resource_data],
                        'borderColor': '#f59e0b',
                        'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                        'tension': 0.4
                    }
                ]
            }
            
            # 5. Performance summary metrics
            cursor.execute('''
                SELECT 
                    AVG(processing_time) as avg_time,
                    MIN(processing_time) as min_time,
                    MAX(processing_time) as max_time,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) as total_analyses,
                    SUM(CASE WHEN decision = 'DEFECT' THEN 1 ELSE 0 END) as defects
                FROM performance_metrics 
                WHERE timestamp >= DATE('now', '-7 days')
            ''')
            
            summary = cursor.fetchone()
            chart_data['summary_metrics'] = {
                'avg_processing_time': summary[0] or 0,
                'min_processing_time': summary[1] or 0,
                'max_processing_time': summary[2] or 0,
                'avg_confidence': summary[3] or 0,
                'total_analyses': summary[4] or 0,
                'defect_rate': (summary[5] / summary[4] * 100) if summary[4] > 0 else 0,
                'throughput_fps': 1 / summary[0] if summary[0] > 0 else 0
            }
            
            conn.close()
            return chart_data
            
        except Exception as e:
            print(f"Error generating chart data: {e}")
            return self._get_empty_chart_data()
    
    def get_realtime_metrics(self):
        """Get real-time metrics for live dashboard updates"""
        if not self.realtime_processing_times:
            return {}
        
        return {
            'current_throughput': 1 / list(self.realtime_processing_times)[-1] if self.realtime_processing_times else 0,
            'avg_processing_time_5min': float(np.mean(list(self.realtime_processing_times)[-10:])),
            'current_memory_usage': list(self.realtime_memory)[-1] if self.realtime_memory else 0,
            'current_cpu_usage': list(self.realtime_cpu)[-1] if self.realtime_cpu else 0,
            'trend_processing_time': self._calculate_short_term_trend(list(self.realtime_processing_times)),
            'trend_memory': self._calculate_short_term_trend(list(self.realtime_memory)),
            'samples_count': len(self.realtime_processing_times)
        }
    
    def _calculate_short_term_trend(self, data):
        """Calculate short-term trend direction"""
        if len(data) < 5:
            return 'stable'
        
        recent = data[-5:]
        earlier = data[-10:-5] if len(data) >= 10 else data[:-5]
        
        if not earlier:
            return 'stable'
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        change_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100
        
        if change_percent > 10:
            return 'increasing'
        elif change_percent < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def update_daily_summary(self):
        """Update daily performance summary in database"""
        try:
            today = datetime.now().date()
            metrics = self.get_enhanced_metrics()
            
            if not metrics['basic_stats']['total_tests']:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance 
                (date, total_analyses, avg_processing_time, max_processing_time, 
                 min_processing_time, avg_memory_usage, avg_cpu_usage, defect_rate, 
                 avg_confidence, throughput_per_hour, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                today,
                metrics['basic_stats']['total_tests'],
                metrics['basic_stats']['avg_processing_time'],
                metrics['basic_stats']['max_processing_time'],
                metrics['basic_stats']['min_processing_time'],
                metrics['system_metrics']['avg_memory_usage'],
                metrics['system_metrics']['avg_cpu_usage'],
                metrics['quality_metrics']['defect_rate'],
                metrics['quality_metrics']['avg_confidence'],
                metrics['throughput_metrics']['avg_throughput_fps'] * 3600  # per hour
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating daily summary: {e}")
    
    def _get_empty_metrics(self):
        """Return empty metrics structure"""
        return {
            'basic_stats': {
                'total_tests': 0,
                'avg_processing_time': 0,
                'min_processing_time': 0,
                'max_processing_time': 0,
                'std_processing_time': 0,
                'median_processing_time': 0
            },
            'throughput_metrics': {
                'avg_throughput_fps': 0,
                'total_processing_time': 0
            },
            'system_metrics': {
                'avg_memory_usage': 0,
                'avg_cpu_usage': 0
            },
            'quality_metrics': {
                'good_products': 0,
                'defective_products': 0,
                'error_count': 0,
                'good_rate': 0,
                'defect_rate': 0,
                'error_rate': 0
            }
        }
    
    def _get_empty_chart_data(self):
        """Return empty chart data structure"""
        return {
            'performance_trend': {'labels': [], 'datasets': []},
            'hourly_pattern': {'labels': [], 'datasets': []},
            'quality_distribution': {'labels': [], 'datasets': []},
            'resource_usage': {'labels': [], 'datasets': []},
            'summary_metrics': {
                'avg_processing_time': 0,
                'min_processing_time': 0,
                'max_processing_time': 0,
                'avg_confidence': 0,
                'total_analyses': 0,
                'defect_rate': 0,
                'throughput_fps': 0
            }
        }
    
    def create_enhanced_performance_charts(self, output_dir="outputs"):
        """Create comprehensive performance charts with enhanced analytics"""
        try:
            if not self.processing_times:
                print("No performance data to chart")
                return None
            
            Path(output_dir).mkdir(exist_ok=True)
            
            # Create enhanced dashboard with multiple charts
            plt.ioff()
            fig, axes = plt.subplots(3, 3, figsize=(20, 16), facecolor='white')
            fig.suptitle('Comprehensive Performance Analytics Dashboard', fontsize=20, fontweight='bold')
            
            # Chart 1: Processing time trend with moving average
            self._create_processing_trend_chart(axes[0, 0])
            
            # Chart 2: System resource usage
            self._create_resource_usage_chart(axes[0, 1])
            
            # Chart 3: Quality metrics distribution
            self._create_quality_distribution_chart(axes[0, 2])
            
            # Chart 4: Throughput analysis
            self._create_throughput_chart(axes[1, 0])
            
            # Chart 5: Performance heatmap
            self._create_performance_heatmap(axes[1, 1])
            
            # Chart 6: Confidence score analysis
            self._create_confidence_analysis_chart(axes[1, 2])
            
            # Chart 7: Error rate analysis
            self._create_error_analysis_chart(axes[2, 0])
            
            # Chart 8: Processing time distribution
            self._create_time_distribution_chart(axes[2, 1])
            
            # Chart 9: Performance summary metrics
            self._create_summary_metrics_chart(axes[2, 2])
            
            plt.tight_layout()
            
            # Save enhanced performance chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = Path(output_dir) / f"enhanced_performance_analysis_{timestamp}.png"
            
            fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"Enhanced performance charts saved: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating enhanced performance charts: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _create_processing_trend_chart(self, ax):
        """Create processing time trend with moving average"""
        iterations = list(range(1, len(self.processing_times) + 1))
        ax.plot(iterations, self.processing_times, 'b-', alpha=0.6, linewidth=1, label='Processing Time')
        
        # Add moving average
        if len(self.processing_times) > 5:
            window = min(10, len(self.processing_times) // 3)
            moving_avg = np.convolve(self.processing_times, np.ones(window)/window, mode='valid')
            ma_iterations = iterations[window-1:]
            ax.plot(ma_iterations, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
        
        ax.set_title('Processing Time Trend Analysis', fontweight='bold')
        ax.set_xlabel('Analysis #')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_resource_usage_chart(self, ax):
        """Create system resource usage chart"""
        if self.memory_usage and self.cpu_usage:
            iterations = list(range(1, len(self.memory_usage) + 1))
            
            ax2 = ax.twinx()
            line1 = ax.plot(iterations, self.memory_usage, 'g-', linewidth=2, label='Memory %')
            line2 = ax2.plot(iterations, self.cpu_usage, 'r-', linewidth=2, label='CPU %')
            
            ax.set_xlabel('Analysis #')
            ax.set_ylabel('Memory Usage (%)', color='g')
            ax2.set_ylabel('CPU Usage (%)', color='r')
            ax.set_title('System Resource Usage', fontweight='bold')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No resource data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Resource Usage', fontweight='bold')
    
    def _create_quality_distribution_chart(self, ax):
        """Create quality metrics distribution"""
        if self.decisions:
            decision_counts = {}
            for decision in self.decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            colors = []
            for decision in decision_counts.keys():
                if decision == 'GOOD':
                    colors.append('#10b981')
                elif decision == 'DEFECT':
                    colors.append('#ef4444')
                else:
                    colors.append('#6b7280')
            
            wedges, texts, autotexts = ax.pie(
                decision_counts.values(), 
                labels=decision_counts.keys(),
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors
            )
            ax.set_title('Quality Distribution', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Quality Distribution', fontweight='bold')
    
    def _create_throughput_chart(self, ax):
        """Create throughput analysis chart"""
        if self.processing_times:
            throughput = [1/t for t in self.processing_times]
            iterations = list(range(1, len(throughput) + 1))
            
            ax.plot(iterations, throughput, 'purple', linewidth=2, marker='o', markersize=4)
            ax.axhline(y=np.mean(throughput), color='orange', linestyle='--', 
                      label=f'Avg: {np.mean(throughput):.2f} FPS')
            ax.set_title('Processing Throughput', fontweight='bold')
            ax.set_xlabel('Analysis #')
            ax.set_ylabel('Throughput (FPS)')