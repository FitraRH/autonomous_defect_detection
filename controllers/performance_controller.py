# controllers/performance_controller.py
"""
Performance Controller for JSON API
Handles performance monitoring and analytics endpoints
"""

from flask import jsonify
from datetime import datetime, timedelta


class PerformanceController:
    """
    Controller for performance-related API endpoints
    Manages performance metrics, trends, and system monitoring
    """
    
    def __init__(self, database_service):
        self.database_service = database_service
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        try:
            # Get performance data from database
            metrics = self.database_service.get_performance_overview()
            
            # Add real-time system metrics
            system_metrics = self._get_current_system_metrics()
            
            response_data = {
                'performance_overview': metrics,
                'system_metrics': system_metrics,
                'health_indicators': self._calculate_health_indicators(metrics, system_metrics),
                'last_updated': datetime.now().isoformat()
            }
            
            return jsonify({
                'status': 'success',
                'data': response_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_performance_trends(self, request):
        """Get performance trends with time period filtering"""
        try:
            # Extract time period parameters
            period = request.args.get('period', '7d')  # 1h, 24h, 7d, 30d
            metric_type = request.args.get('metric', 'all')  # processing_time, throughput, accuracy, etc.
            
            # Validate period
            valid_periods = ['1h', '24h', '7d', '30d']
            if period not in valid_periods:
                return jsonify({
                    'status': 'error',
                    'error': f'Invalid period. Valid options: {valid_periods}',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Get trend data
            trends = self.database_service.get_performance_trends(period, metric_type)
            
            # Calculate trend analysis
            trend_analysis = self._analyze_trends(trends, period)
            
            response_data = {
                'period': period,
                'metric_type': metric_type,
                'trends': trends,
                'trend_analysis': trend_analysis,
                'summary': self._generate_trend_summary(trends, trend_analysis)
            }
            
            return jsonify({
                'status': 'success',
                'data': response_data,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_performance_charts(self):
        """Get performance chart data for visualization"""
        try:
            # Get various chart data from database
            chart_data = {
                'processing_time_chart': self.database_service.get_processing_time_chart_data(),
                'throughput_chart': self.database_service.get_throughput_chart_data(),
                'accuracy_trend': self.database_service.get_accuracy_trend_data(),
                'defect_rate_chart': self.database_service.get_defect_rate_chart_data(),
                'system_resource_chart': self.database_service.get_system_resource_chart_data(),
                'hourly_activity': self.database_service.get_hourly_activity_data()
            }
            
            # Add chart metadata
            chart_metadata = {
                'generated_at': datetime.now().isoformat(),
                'data_period': '30 days',
                'chart_types': list(chart_data.keys()),
                'total_data_points': sum(len(data.get('data_points', [])) for data in chart_data.values())
            }
            
            return jsonify({
                'status': 'success',
                'data': {
                    'charts': chart_data,
                    'metadata': chart_metadata
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def _get_current_system_metrics(self):
        """Get current system resource metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (basic)
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                    'status': 'normal' if cpu_percent < 80 else 'high' if cpu_percent < 95 else 'critical'
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'usage_percent': memory.percent,
                    'status': 'normal' if memory.percent < 80 else 'high' if memory.percent < 95 else 'critical'
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'usage_percent': round((disk.used / disk.total) * 100, 2),
                    'status': 'normal' if disk.used / disk.total < 0.8 else 'high' if disk.used / disk.total < 0.95 else 'critical'
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_received': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_received': network.packets_recv
                },
                'system_load': {
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                    'process_count': len(psutil.pids()),
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
                }
            }
            
        except ImportError:
            return {
                'cpu': {'usage_percent': 0, 'status': 'monitoring_unavailable'},
                'memory': {'usage_percent': 0, 'status': 'monitoring_unavailable'},
                'disk': {'usage_percent': 0, 'status': 'monitoring_unavailable'},
                'network': {'status': 'monitoring_unavailable'},
                'system_load': {'status': 'monitoring_unavailable'}
            }
        except Exception as e:
            return {
                'error': f'Failed to get system metrics: {str(e)}',
                'status': 'error'
            }
    
    def _calculate_health_indicators(self, performance_metrics, system_metrics):
        """Calculate overall system health indicators"""
        try:
            health_score = 100
            issues = []
            recommendations = []
            
            # Check performance metrics
            if performance_metrics.get('avg_processing_time', 0) > 2.0:
                health_score -= 15
                issues.append('High processing time detected')
                recommendations.append('Consider optimizing detection algorithms')
            
            if performance_metrics.get('error_rate', 0) > 5:
                health_score -= 20
                issues.append('High error rate detected')
                recommendations.append('Investigate error causes and implement fixes')
            
            # Check system metrics
            cpu_status = system_metrics.get('cpu', {}).get('status', 'normal')
            if cpu_status == 'high':
                health_score -= 10
                issues.append('High CPU usage')
                recommendations.append('Monitor CPU-intensive processes')
            elif cpu_status == 'critical':
                health_score -= 25
                issues.append('Critical CPU usage')
                recommendations.append('Immediate CPU optimization required')
            
            memory_status = system_metrics.get('memory', {}).get('status', 'normal')
            if memory_status == 'high':
                health_score -= 10
                issues.append('High memory usage')
                recommendations.append('Monitor memory leaks and optimize memory usage')
            elif memory_status == 'critical':
                health_score -= 25
                issues.append('Critical memory usage')
                recommendations.append('Immediate memory optimization required')
            
            disk_status = system_metrics.get('disk', {}).get('status', 'normal')
            if disk_status == 'high':
                health_score -= 5
                issues.append('High disk usage')
                recommendations.append('Clean up old files and logs')
            elif disk_status == 'critical':
                health_score -= 15
                issues.append('Critical disk usage')
                recommendations.append('Immediate disk cleanup required')
            
            # Determine overall health level
            if health_score >= 90:
                health_level = 'excellent'
            elif health_score >= 80:
                health_level = 'good'
            elif health_score >= 70:
                health_level = 'fair'
            elif health_score >= 60:
                health_level = 'poor'
            else:
                health_level = 'critical'
            
            return {
                'overall_health_score': max(0, health_score),
                'health_level': health_level,
                'issues_detected': issues,
                'recommendations': recommendations,
                'status_indicators': {
                    'processing_performance': 'good' if performance_metrics.get('avg_processing_time', 0) < 1.5 else 'needs_attention',
                    'system_resources': 'good' if all(status in ['normal'] for status in [cpu_status, memory_status, disk_status]) else 'needs_attention',
                    'error_rate': 'good' if performance_metrics.get('error_rate', 0) < 2 else 'needs_attention'
                }
            }
            
        except Exception as e:
            return {
                'overall_health_score': 0,
                'health_level': 'unknown',
                'issues_detected': [f'Health calculation error: {str(e)}'],
                'recommendations': ['Check system monitoring configuration'],
                'status_indicators': {}
            }
    
    def _analyze_trends(self, trends, period):
        """Analyze performance trends and detect patterns"""
        try:
            analysis = {
                'trend_direction': 'stable',
                'volatility': 'low',
                'significant_changes': [],
                'pattern_detection': {},
                'forecasting': {}
            }
            
            if not trends or len(trends) < 2:
                return analysis
            
            # Processing time trend analysis
            if 'processing_times' in trends:
                processing_times = trends['processing_times']
                if len(processing_times) >= 2:
                    recent_avg = sum(processing_times[-3:]) / min(3, len(processing_times))
                    older_avg = sum(processing_times[:3]) / min(3, len(processing_times))
                    
                    if recent_avg > older_avg * 1.1:
                        analysis['trend_direction'] = 'degrading'
                        analysis['significant_changes'].append('Processing time increasing')
                    elif recent_avg < older_avg * 0.9:
                        analysis['trend_direction'] = 'improving'
                        analysis['significant_changes'].append('Processing time decreasing')
                    
                    # Calculate volatility
                    if len(processing_times) > 1:
                        import numpy as np
                        std_dev = np.std(processing_times)
                        mean_time = np.mean(processing_times)
                        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
                        
                        if coefficient_of_variation > 0.3:
                            analysis['volatility'] = 'high'
                        elif coefficient_of_variation > 0.15:
                            analysis['volatility'] = 'medium'
            
            # Throughput trend analysis
            if 'throughput' in trends:
                throughput_data = trends['throughput']
                if len(throughput_data) >= 2:
                    recent_throughput = sum(throughput_data[-3:]) / min(3, len(throughput_data))
                    older_throughput = sum(throughput_data[:3]) / min(3, len(throughput_data))
                    
                    if recent_throughput > older_throughput * 1.1:
                        analysis['significant_changes'].append('Throughput increasing')
                    elif recent_throughput < older_throughput * 0.9:
                        analysis['significant_changes'].append('Throughput decreasing')
            
            # Pattern detection
            analysis['pattern_detection'] = self._detect_patterns(trends, period)
            
            # Simple forecasting
            analysis['forecasting'] = self._generate_forecast(trends, period)
            
            return analysis
            
        except Exception as e:
            return {
                'trend_direction': 'unknown',
                'volatility': 'unknown',
                'significant_changes': [f'Analysis error: {str(e)}'],
                'pattern_detection': {},
                'forecasting': {}
            }
    
    def _detect_patterns(self, trends, period):
        """Detect patterns in performance data"""
        patterns = {
            'cyclical_patterns': False,
            'peak_hours': [],
            'low_activity_periods': [],
            'anomalies_detected': []
        }
        
        try:
            # Detect hourly patterns if we have hourly data
            if 'hourly_data' in trends:
                hourly_data = trends['hourly_data']
                if hourly_data:
                    # Find peak hours (simple approach)
                    max_activity = max(hourly_data)
                    peak_threshold = max_activity * 0.8
                    
                    for i, activity in enumerate(hourly_data):
                        if activity >= peak_threshold:
                            patterns['peak_hours'].append(f'{i:02d}:00')
                        elif activity < max_activity * 0.2:
                            patterns['low_activity_periods'].append(f'{i:02d}:00')
            
            # Detect anomalies in processing times
            if 'processing_times' in trends:
                processing_times = trends['processing_times']
                if len(processing_times) > 3:
                    import numpy as np
                    mean_time = np.mean(processing_times)
                    std_time = np.std(processing_times)
                    
                    for i, time_val in enumerate(processing_times):
                        if abs(time_val - mean_time) > 2 * std_time:
                            patterns['anomalies_detected'].append({
                                'index': i,
                                'value': time_val,
                                'type': 'processing_time_anomaly'
                            })
            
            return patterns
            
        except Exception as e:
            patterns['error'] = str(e)
            return patterns
    
    def _generate_forecast(self, trends, period):
        """Generate simple performance forecast"""
        forecast = {
            'next_period_prediction': 'stable',
            'confidence': 'low',
            'expected_range': {},
            'recommendations': []
        }
        
        try:
            if 'processing_times' in trends and len(trends['processing_times']) >= 3:
                processing_times = trends['processing_times']
                recent_trend = processing_times[-3:]
                
                import numpy as np
                trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
                
                if trend_slope > 0.01:
                    forecast['next_period_prediction'] = 'increasing_processing_time'
                    forecast['recommendations'].append('Monitor for performance degradation')
                elif trend_slope < -0.01:
                    forecast['next_period_prediction'] = 'improving_processing_time'
                    forecast['recommendations'].append('Performance optimization working well')
                
                # Simple range prediction
                recent_avg = np.mean(recent_trend)
                recent_std = np.std(recent_trend)
                
                forecast['expected_range'] = {
                    'min_processing_time': max(0, recent_avg - recent_std),
                    'max_processing_time': recent_avg + recent_std,
                    'most_likely': recent_avg
                }
                
                forecast['confidence'] = 'medium' if len(processing_times) >= 10 else 'low'
            
            return forecast
            
        except Exception as e:
            forecast['error'] = str(e)
            return forecast
    
    def _generate_trend_summary(self, trends, trend_analysis):
        """Generate human-readable trend summary"""
        try:
            summary = {
                'overall_performance': 'stable',
                'key_insights': [],
                'alerts': [],
                'positive_trends': [],
                'areas_for_improvement': []
            }
            
            # Analyze trend direction
            if trend_analysis['trend_direction'] == 'improving':
                summary['overall_performance'] = 'improving'
                summary['positive_trends'].append('Processing performance is improving')
            elif trend_analysis['trend_direction'] == 'degrading':
                summary['overall_performance'] = 'declining'
                summary['areas_for_improvement'].append('Processing performance needs attention')
            
            # Analyze volatility
            if trend_analysis['volatility'] == 'high':
                summary['alerts'].append('High performance volatility detected')
                summary['areas_for_improvement'].append('Investigate causes of performance instability')
            
            # Add significant changes
            for change in trend_analysis['significant_changes']:
                if 'increasing' in change.lower() and 'time' in change.lower():
                    summary['areas_for_improvement'].append(change)
                elif 'increasing' in change.lower() and 'throughput' in change.lower():
                    summary['positive_trends'].append(change)
                elif 'decreasing' in change.lower() and 'time' in change.lower():
                    summary['positive_trends'].append(change)
            
            # Add pattern insights
            patterns = trend_analysis.get('pattern_detection', {})
            if patterns.get('peak_hours'):
                summary['key_insights'].append(f"Peak activity hours: {', '.join(patterns['peak_hours'][:3])}")
            
            if patterns.get('anomalies_detected'):
                summary['alerts'].append(f"{len(patterns['anomalies_detected'])} performance anomalies detected")
            
            # Add forecasting insights
            forecast = trend_analysis.get('forecasting', {})
            if forecast.get('next_period_prediction') and forecast['next_period_prediction'] != 'stable':
                summary['key_insights'].append(f"Forecast: {forecast['next_period_prediction'].replace('_', ' ')}")
            
            return summary
            
        except Exception as e:
            return {
                'overall_performance': 'unknown',
                'key_insights': [],
                'alerts': [f'Summary generation error: {str(e)}'],
                'positive_trends': [],
                'areas_for_improvement': []
            }