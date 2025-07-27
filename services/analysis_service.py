# services/analysis_service.py
"""
Analysis Service - Business Logic for Analysis Management
Handles analysis statistics, trends, and data processing
Integrates with existing utils and performance tracking
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import existing utility modules
try:
    from utils.performance_tracker import EnhancedPerformanceTracker
    from utils.reports import save_enhanced_analysis_report, generate_enhanced_batch_report
    from utils.visualization import create_enhanced_visualization
    from utils.pdf_generator import PDFReportGenerator
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Some utils not available: {e}")
    UTILS_AVAILABLE = False


class AnalysisService:
    """
    Analysis Service - Business logic for analysis management and statistics
    Integrates with existing utility modules and performance tracking
    """
    
    def __init__(self):
        # Initialize performance tracker if available
        if UTILS_AVAILABLE:
            try:
                self.performance_tracker = EnhancedPerformanceTracker()
                self.pdf_generator = PDFReportGenerator()
                print("Analysis service initialized with full utils support")
            except Exception as e:
                print(f"Error initializing utils: {e}")
                self.performance_tracker = None
                self.pdf_generator = None
        else:
            self.performance_tracker = None
            self.pdf_generator = None
        
        # Service configuration
        self.reports_dir = Path("outputs/reports")
        self.exports_dir = Path("exports")
        self.temp_dir = Path("temp")
        
        # Create directories
        for directory in [self.reports_dir, self.exports_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def calculate_comprehensive_statistics(self, database_service):
        """Calculate comprehensive analysis statistics using existing logic"""
        try:
            # Get basic statistics from database
            basic_stats = database_service.get_comprehensive_statistics()
            
            # Enhance with performance tracker data if available
            if self.performance_tracker:
                performance_metrics = self.performance_tracker.get_enhanced_metrics()
                chart_data = self.performance_tracker.get_chart_data_for_dashboard()
                realtime_metrics = self.performance_tracker.get_realtime_metrics()
                
                # Combine database stats with performance metrics
                enhanced_stats = self._merge_statistics(basic_stats, performance_metrics, chart_data)
            else:
                enhanced_stats = basic_stats
            
            # Add calculated insights
            enhanced_stats['insights'] = self._generate_insights(enhanced_stats)
            enhanced_stats['recommendations'] = self._generate_recommendations(enhanced_stats)
            
            return enhanced_stats
            
        except Exception as e:
            print(f"Error calculating comprehensive statistics: {e}")
            # Return fallback statistics
            return {
                'overview': {'error': str(e)},
                'defect_distribution': [],
                'daily_trends': [],
                'insights': [],
                'recommendations': []
            }
    
    def generate_analysis_report(self, analysis_data, report_format='json'):
        """Generate comprehensive analysis report using existing report utilities"""
        try:
            if not UTILS_AVAILABLE:
                raise RuntimeError("Report generation utilities not available")
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_analysis',
                'analysis_data': analysis_data
            }
            
            # Generate report using existing utilities
            if report_format.lower() == 'pdf':
                if self.pdf_generator:
                    report_path = self.pdf_generator.generate_single_image_report(
                        analysis_data, str(self.reports_dir)
                    )
                else:
                    raise RuntimeError("PDF generator not available")
            
            elif report_format.lower() == 'enhanced_text':
                report_path = save_enhanced_analysis_report(
                    analysis_data, str(self.reports_dir)
                )
            
            else:  # JSON format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"analysis_report_{timestamp}.json"
                report_path = self.reports_dir / report_filename
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                'report_path': str(report_path),
                'report_format': report_format,
                'file_size': os.path.getsize(report_path) if os.path.exists(report_path) else 0,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating analysis report: {e}")
            raise
    
    def generate_batch_report(self, batch_data, report_format='json'):
        """Generate batch analysis report using existing batch report utilities"""
        try:
            if not UTILS_AVAILABLE:
                raise RuntimeError("Report generation utilities not available")
            
            # Use existing batch report generator
            if report_format.lower() == 'enhanced_text':
                report_path = generate_enhanced_batch_report(
                    batch_data, str(self.reports_dir)
                )
            
            elif report_format.lower() == 'pdf':
                if self.pdf_generator:
                    report_path = self.pdf_generator.generate_batch_report(
                        batch_data, str(self.reports_dir)
                    )
                else:
                    raise RuntimeError("PDF generator not available")
            
            else:  # JSON format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"batch_report_{timestamp}.json"
                report_path = self.reports_dir / report_filename
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                'report_path': str(report_path),
                'report_format': report_format,
                'batch_summary': self._calculate_batch_summary(batch_data),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating batch report: {e}")
            raise
    
    def create_visualization(self, analysis_result):
        """Create visualization using existing visualization utilities"""
        try:
            if not UTILS_AVAILABLE:
                raise RuntimeError("Visualization utilities not available")
            
            # Use existing visualization creator
            viz_path = create_enhanced_visualization(
                analysis_result, str(self.reports_dir)
            )
            
            if viz_path and os.path.exists(viz_path):
                return {
                    'visualization_path': viz_path,
                    'file_size': os.path.getsize(viz_path),
                    'format': 'png',
                    'created_at': datetime.now().isoformat()
                }
            else:
                raise RuntimeError("Visualization creation failed")
                
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def analyze_trends(self, historical_data, period_days=30):
        """Analyze trends in historical analysis data"""
        try:
            if not historical_data or len(historical_data) < 2:
                return {
                    'trend_analysis': 'insufficient_data',
                    'insights': ['Not enough data for trend analysis'],
                    'recommendations': ['Collect more analysis data']
                }
            
            # Extract time series data
            dates = [item.get('analysis_date') for item in historical_data if item.get('analysis_date')]
            processing_times = [item.get('processing_time', 0) for item in historical_data]
            anomaly_scores = [item.get('anomaly_score', 0) for item in historical_data]
            defect_counts = [len(item.get('defects', [])) for item in historical_data]
            
            # Calculate trends
            trends = {
                'processing_time_trend': self._calculate_trend(processing_times),
                'anomaly_score_trend': self._calculate_trend(anomaly_scores),
                'defect_detection_trend': self._calculate_trend(defect_counts),
                'volume_trend': self._calculate_volume_trend(dates, period_days)
            }
            
            # Generate insights
            insights = self._generate_trend_insights(trends, historical_data)
            
            # Performance analysis
            performance_analysis = self._analyze_performance_trends(processing_times, anomaly_scores)
            
            return {
                'period_analyzed': f'{period_days} days',
                'data_points': len(historical_data),
                'trends': trends,
                'insights': insights,
                'performance_analysis': performance_analysis,
                'quality_indicators': self._calculate_quality_indicators(historical_data)
            }
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return {
                'error': str(e),
                'trend_analysis': 'error',
                'insights': ['Trend analysis failed'],
                'recommendations': ['Check data format and availability']
            }
    
    def calculate_quality_metrics(self, analysis_data):
        """Calculate comprehensive quality metrics"""
        try:
            total_analyses = len(analysis_data)
            if total_analyses == 0:
                return self._get_empty_quality_metrics()
            
            # Basic counts
            good_count = sum(1 for item in analysis_data if item.get('final_decision') == 'GOOD')
            defect_count = sum(1 for item in analysis_data if item.get('final_decision') == 'DEFECT')
            error_count = total_analyses - good_count - defect_count
            
            # Processing metrics
            processing_times = [item.get('processing_time', 0) for item in analysis_data if item.get('processing_time')]
            anomaly_scores = [item.get('anomaly_score', 0) for item in analysis_data if item.get('anomaly_score')]
            
            # Calculate quality metrics
            quality_metrics = {
                'overview': {
                    'total_analyses': total_analyses,
                    'good_products': good_count,
                    'defective_products': defect_count,
                    'processing_errors': error_count,
                    'success_rate': (good_count + defect_count) / total_analyses * 100,
                    'defect_rate': defect_count / total_analyses * 100 if total_analyses > 0 else 0
                },
                'performance': {
                    'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                    'min_processing_time': np.min(processing_times) if processing_times else 0,
                    'max_processing_time': np.max(processing_times) if processing_times else 0,
                    'processing_time_std': np.std(processing_times) if processing_times else 0,
                    'throughput_per_minute': 60 / np.mean(processing_times) if processing_times and np.mean(processing_times) > 0 else 0
                },
                'accuracy': {
                    'avg_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0,
                    'anomaly_score_std': np.std(anomaly_scores) if anomaly_scores else 0,
                    'confidence_distribution': self._calculate_confidence_distribution(analysis_data),
                    'detection_consistency': self._calculate_detection_consistency(analysis_data)
                },
                'defect_analysis': self._analyze_defect_patterns(analysis_data),
                'quality_score': self._calculate_overall_quality_score(good_count, defect_count, processing_times, anomaly_scores)
            }
            
            return quality_metrics
            
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return self._get_empty_quality_metrics()
    
    def export_analysis_data(self, analysis_data, export_format='json', filters=None):
        """Export analysis data with filtering and formatting"""
        try:
            # Apply filters if provided
            if filters:
                analysis_data = self._apply_filters(analysis_data, filters)
            
            # Generate export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"analysis_export_{timestamp}.{export_format.lower()}"
            export_path = self.exports_dir / export_filename
            
            # Export based on format
            if export_format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'export_info': {
                            'generated_at': datetime.now().isoformat(),
                            'total_records': len(analysis_data),
                            'filters_applied': filters or {},
                            'format': 'json'
                        },
                        'data': analysis_data
                    }, f, indent=2, ensure_ascii=False, default=str)
            
            elif export_format.lower() == 'csv':
                import pandas as pd
                df = self._convert_to_dataframe(analysis_data)
                df.to_csv(export_path, index=False, encoding='utf-8')
            
            elif export_format.lower() == 'xlsx':
                import pandas as pd
                df = self._convert_to_dataframe(analysis_data)
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Analysis Data', index=False)
                    
                    # Add summary sheet
                    summary_df = self._create_summary_dataframe(analysis_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Get file info
            file_size = os.path.getsize(export_path)
            
            return {
                'export_path': str(export_path),
                'filename': export_filename,
                'format': export_format,
                'file_size': file_size,
                'record_count': len(analysis_data),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error exporting analysis data: {e}")
            raise
    
    def cleanup_old_files(self, days_to_keep=30):
        """Clean up old report and export files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cleaned_files = []
            total_size_freed = 0
            
            # Clean up directories
            for directory in [self.reports_dir, self.exports_dir, self.temp_dir]:
                if directory.exists():
                    for file_path in directory.iterdir():
                        if file_path.is_file():
                            # Check file age
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                                total_size_freed += file_size
            
            return {
                'files_cleaned': len(cleaned_files),
                'size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
                'cutoff_date': cutoff_date.isoformat(),
                'cleaned_files': cleaned_files[:10]  # Show first 10 files
            }
            
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            return {
                'files_cleaned': 0,
                'size_freed_mb': 0,
                'error': str(e)
            }
    
    def _merge_statistics(self, basic_stats, performance_metrics, chart_data):
        """Merge database statistics with performance tracker data"""
        try:
            merged = basic_stats.copy()
            
            # Add performance metrics
            if performance_metrics:
                merged['performance_metrics'] = performance_metrics
                merged['system_metrics'] = performance_metrics.get('system_metrics', {})
                merged['quality_metrics_enhanced'] = performance_metrics.get('quality_metrics', {})
            
            # Add chart data
            if chart_data:
                merged['chart_data'] = chart_data
            
            return merged
            
        except Exception as e:
            print(f"Error merging statistics: {e}")
            return basic_stats
    
    def _generate_insights(self, statistics):
        """Generate insights from statistics data"""
        insights = []
        
        try:
            overview = statistics.get('overview', {})
            
            # Defect rate insights
            defect_rate = overview.get('defect_rate', 0)
            if defect_rate == 0:
                insights.append("Excellent quality: No defects detected in recent analyses")
            elif defect_rate < 5:
                insights.append(f"Good quality: Low defect rate of {defect_rate:.1f}%")
            elif defect_rate < 15:
                insights.append(f"Quality concern: Moderate defect rate of {defect_rate:.1f}%")
            else:
                insights.append(f"Quality alert: High defect rate of {defect_rate:.1f}%")
            
            # Performance insights
            avg_time = overview.get('avg_processing_time', 0)
            if avg_time > 0:
                if avg_time < 1.0:
                    insights.append(f"Excellent performance: Fast processing at {avg_time:.2f}s per image")
                elif avg_time < 2.0:
                    insights.append(f"Good performance: Processing at {avg_time:.2f}s per image")
                else:
                    insights.append(f"Performance concern: Slow processing at {avg_time:.2f}s per image")
            
            # Volume insights
            total_analyses = overview.get('total_analyses', 0)
            if total_analyses > 1000:
                insights.append(f"High volume: {total_analyses:,} analyses processed")
            elif total_analyses > 100:
                insights.append(f"Moderate volume: {total_analyses:,} analyses processed")
            else:
                insights.append(f"Low volume: {total_analyses:,} analyses processed")
            
            # Defect pattern insights
            defect_distribution = statistics.get('defect_distribution', [])
            if defect_distribution:
                most_common = defect_distribution[0]
                insights.append(f"Most common defect: {most_common['defect_type']} ({most_common['count']} occurrences)")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _generate_recommendations(self, statistics):
        """Generate recommendations based on statistics"""
        recommendations = []
        
        try:
            overview = statistics.get('overview', {})
            
            # Defect rate recommendations
            defect_rate = overview.get('defect_rate', 0)
            if defect_rate > 10:
                recommendations.append("Implement immediate quality control measures")
                recommendations.append("Investigate root causes of high defect rate")
            elif defect_rate > 5:
                recommendations.append("Review production processes for quality improvement")
                recommendations.append("Increase inspection frequency")
            
            # Performance recommendations
            avg_time = overview.get('avg_processing_time', 0)
            if avg_time > 2.0:
                recommendations.append("Consider optimizing detection algorithms")
                recommendations.append("Review system resources and hardware capabilities")
            
            # Volume recommendations
            total_analyses = overview.get('total_analyses', 0)
            if total_analyses < 50:
                recommendations.append("Increase analysis volume for better statistical reliability")
            
            # General recommendations
            recommendations.append("Monitor trends regularly for early detection of issues")
            recommendations.append("Maintain consistent data collection practices")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _calculate_trend(self, data):
        """Calculate trend direction for numerical data"""
        if len(data) < 2:
            return 'insufficient_data'
        
        try:
            # Simple linear regression slope
            x = list(range(len(data)))
            slope = np.polyfit(x, data, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _calculate_volume_trend(self, dates, period_days):
        """Calculate volume trend over time period"""
        try:
            if not dates or len(dates) < 2:
                return 'insufficient_data'
            
            # Convert dates and count by day
            from collections import defaultdict
            daily_counts = defaultdict(int)
            
            for date_str in dates:
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        day_key = date_obj.date()
                        daily_counts[day_key] += 1
                    except:
                        continue
            
            if len(daily_counts) < 2:
                return 'insufficient_data'
            
            # Calculate trend in daily volumes
            sorted_days = sorted(daily_counts.keys())
            volumes = [daily_counts[day] for day in sorted_days]
            
            return self._calculate_trend(volumes)
            
        except Exception:
            return 'unknown'
    
    def _generate_trend_insights(self, trends, historical_data):
        """Generate insights from trend analysis"""
        insights = []
        
        try:
            # Processing time trends
            if trends['processing_time_trend'] == 'increasing':
                insights.append("Processing time is increasing - investigate performance bottlenecks")
            elif trends['processing_time_trend'] == 'decreasing':
                insights.append("Processing time is improving - optimizations are working")
            
            # Defect detection trends
            if trends['defect_detection_trend'] == 'increasing':
                insights.append("Defect detection rate is increasing - quality may be declining")
            elif trends['defect_detection_trend'] == 'decreasing':
                insights.append("Defect detection rate is decreasing - quality is improving")
            
            # Volume trends
            if trends['volume_trend'] == 'increasing':
                insights.append("Analysis volume is increasing - system usage growing")
            elif trends['volume_trend'] == 'decreasing':
                insights.append("Analysis volume is decreasing - monitor system utilization")
            
            # Data quality insights
            if len(historical_data) < 10:
                insights.append("Limited historical data - collect more data for better trend analysis")
            
        except Exception as e:
            insights.append(f"Error generating trend insights: {str(e)}")
        
        return insights
    
    def _analyze_performance_trends(self, processing_times, anomaly_scores):
        """Analyze performance trends in detail"""
        try:
            if not processing_times or not anomaly_scores:
                return {'error': 'Insufficient performance data'}
            
            analysis = {
                'processing_performance': {
                    'average_time': np.mean(processing_times),
                    'time_variability': np.std(processing_times),
                    'fastest_time': np.min(processing_times),
                    'slowest_time': np.max(processing_times),
                    'performance_consistency': 'high' if np.std(processing_times) < 0.2 else 'medium' if np.std(processing_times) < 0.5 else 'low'
                },
                'detection_performance': {
                    'average_anomaly_score': np.mean(anomaly_scores),
                    'score_variability': np.std(anomaly_scores),
                    'detection_sensitivity': 'high' if np.mean(anomaly_scores) > 0.5 else 'low',
                    'score_distribution': {
                        'low_scores': sum(1 for score in anomaly_scores if score < 0.3),
                        'medium_scores': sum(1 for score in anomaly_scores if 0.3 <= score < 0.7),
                        'high_scores': sum(1 for score in anomaly_scores if score >= 0.7)
                    }
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_quality_indicators(self, historical_data):
        """Calculate quality indicators from historical data"""
        try:
            total = len(historical_data)
            if total == 0:
                return {}
            
            # Decision distribution
            decisions = [item.get('final_decision', 'UNKNOWN') for item in historical_data]
            good_count = decisions.count('GOOD')
            defect_count = decisions.count('DEFECT')
            
            # Confidence distribution
            confidence_levels = [item.get('confidence_level', 'unknown') for item in historical_data]
            confidence_dist = {
                'high_confidence': sum(1 for c in confidence_levels if 'high' in c.lower()),
                'medium_confidence': sum(1 for c in confidence_levels if 'medium' in c.lower()),
                'low_confidence': sum(1 for c in confidence_levels if 'low' in c.lower())
            }
            
            return {
                'decision_accuracy': (good_count + defect_count) / total * 100,
                'quality_consistency': confidence_dist['high_confidence'] / total * 100,
                'detection_reliability': defect_count / total * 100 if total > 0 else 0,
                'overall_quality_score': self._calculate_overall_quality_score(
                    good_count, defect_count, 
                    [item.get('processing_time', 0) for item in historical_data],
                    [item.get('anomaly_score', 0) for item in historical_data]
                )
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_overall_quality_score(self, good_count, defect_count, processing_times, anomaly_scores):
        """Calculate overall quality score (0-100)"""
        try:
            total = good_count + defect_count
            if total == 0:
                return 0
            
            # Base score from accuracy
            accuracy_score = ((good_count + defect_count) / total) * 40
            
            # Performance score
            avg_time = np.mean(processing_times) if processing_times else 2.0
            performance_score = max(0, min(30, 30 - (avg_time - 1.0) * 15))
            
            # Consistency score
            time_std = np.std(processing_times) if processing_times else 1.0
            consistency_score = max(0, min(20, 20 - time_std * 10))
            
            # Detection quality score
            if anomaly_scores:
                score_consistency = 1 - np.std(anomaly_scores)
                detection_score = max(0, min(10, score_consistency * 10))
            else:
                detection_score = 5
            
            total_score = accuracy_score + performance_score + consistency_score + detection_score
            return round(min(100, max(0, total_score)), 1)
            
        except Exception:
            return 0
    
    def _get_empty_quality_metrics(self):
        """Return empty quality metrics structure"""
        return {
            'overview': {
                'total_analyses': 0,
                'good_products': 0,
                'defective_products': 0,
                'processing_errors': 0,
                'success_rate': 0,
                'defect_rate': 0
            },
            'performance': {
                'avg_processing_time': 0,
                'min_processing_time': 0,
                'max_processing_time': 0,
                'processing_time_std': 0,
                'throughput_per_minute': 0
            },
            'accuracy': {
                'avg_anomaly_score': 0,
                'anomaly_score_std': 0,
                'confidence_distribution': {},
                'detection_consistency': 0
            },
            'defect_analysis': {},
            'quality_score': 0
        }
    
    def _calculate_batch_summary(self, batch_data):
        """Calculate summary for batch data"""
        try:
            results = batch_data.get('results', [])
            total = len(results)
            
            if total == 0:
                return {'total_images': 0, 'summary': 'No results to summarize'}
            
            good_count = sum(1 for r in results if r.get('final_decision') == 'GOOD')
            defect_count = sum(1 for r in results if r.get('final_decision') == 'DEFECT')
            
            return {
                'total_images': total,
                'good_products': good_count,
                'defective_products': defect_count,
                'defect_rate': (defect_count / total * 100) if total > 0 else 0,
                'success_rate': ((good_count + defect_count) / total * 100) if total > 0 else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _apply_filters(self, data, filters):
        """Apply filters to data"""
        try:
            filtered_data = data.copy()
            
            # Date filters
            if filters.get('date_from'):
                filtered_data = [
                    item for item in filtered_data 
                    if item.get('analysis_date', '') >= filters['date_from']
                ]
            
            if filters.get('date_to'):
                filtered_data = [
                    item for item in filtered_data 
                    if item.get('analysis_date', '') <= filters['date_to']
                ]
            
            # Decision filter
            if filters.get('decision'):
                filtered_data = [
                    item for item in filtered_data 
                    if item.get('final_decision') == filters['decision']
                ]
            
            # Defect type filter
            if filters.get('defect_type'):
                filtered_data = [
                    item for item in filtered_data 
                    if filters['defect_type'] in item.get('defects', [])
                ]
            
            return filtered_data
            
        except Exception as e:
            print(f"Error applying filters: {e}")
            return data
    
    def _convert_to_dataframe(self, data):
        """Convert analysis data to pandas DataFrame"""
        try:
            import pandas as pd
            
            # Flatten data for CSV/Excel export
            flattened_data = []
            for item in data:
                flat_item = {
                    'id': item.get('id'),
                    'image_name': item.get('image_name'),
                    'analysis_date': item.get('analysis_date'),
                    'final_decision': item.get('final_decision'),
                    'anomaly_score': item.get('anomaly_score'),
                    'processing_time': item.get('processing_time'),
                    'confidence_level': item.get('confidence_level'),
                    'defect_count': item.get('defect_count', 0),
                    'status': item.get('status'),
                    'defects': ', '.join(item.get('defects', [])) if item.get('defects') else 'None'
                }
                flattened_data.append(flat_item)
            
            return pd.DataFrame(flattened_data)
            
        except ImportError:
            raise RuntimeError("pandas not available for DataFrame conversion")
        except Exception as e:
            raise RuntimeError(f"Error converting to DataFrame: {e}")
    
    def _create_summary_dataframe(self, data):
        """Create summary DataFrame for Excel export"""
        try:
            import pandas as pd
            
            # Calculate summary statistics
            total = len(data)
            good_count = sum(1 for item in data if item.get('final_decision') == 'GOOD')
            defect_count = sum(1 for item in data if item.get('final_decision') == 'DEFECT')
            
            processing_times = [item.get('processing_time', 0) for item in data if item.get('processing_time')]
            anomaly_scores = [item.get('anomaly_score', 0) for item in data if item.get('anomaly_score')]
            
            summary_data = [
                ['Total Analyses', total],
                ['Good Products', good_count],
                ['Defective Products', defect_count],
                ['Defect Rate (%)', (defect_count / total * 100) if total > 0 else 0],
                ['Avg Processing Time (s)', np.mean(processing_times) if processing_times else 0],
                ['Avg Anomaly Score', np.mean(anomaly_scores) if anomaly_scores else 0],
                ['Generated At', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            
        except ImportError:
            raise RuntimeError("pandas not available for summary DataFrame creation")
        except Exception as e:
            raise RuntimeError(f"Error creating summary DataFrame: {e}")
    
    def _calculate_confidence_distribution(self, data):
        """Calculate confidence level distribution"""
        try:
            confidence_levels = [item.get('confidence_level', 'unknown') for item in data]
            distribution = {}
            
            for level in confidence_levels:
                level_key = level.lower() if level else 'unknown'
                distribution[level_key] = distribution.get(level_key, 0) + 1
            
            # Convert to percentages
            total = len(data)
            if total > 0:
                for key in distribution:
                    distribution[key] = (distribution[key] / total) * 100
            
            return distribution
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_detection_consistency(self, data):
        """Calculate detection consistency score"""
        try:
            # Analyze consistency between anomaly scores and decisions
            consistent_count = 0
            total_with_scores = 0
            
            for item in data:
                anomaly_score = item.get('anomaly_score')
                decision = item.get('final_decision')
                
                if anomaly_score is not None and decision:
                    total_with_scores += 1
                    # Check if score aligns with decision
                    if (decision == 'GOOD' and anomaly_score < 0.7) or \
                       (decision == 'DEFECT' and anomaly_score >= 0.7):
                        consistent_count += 1
            
            if total_with_scores > 0:
                return (consistent_count / total_with_scores) * 100
            else:
                return 0
                
        except Exception as e:
            return 0
    
    def _analyze_defect_patterns(self, data):
        """Analyze patterns in defect detection"""
        try:
            defect_items = [item for item in data if item.get('final_decision') == 'DEFECT']
            
            if not defect_items:
                return {'total_defects': 0, 'patterns': 'No defects to analyze'}
            
            # Count defect types
            defect_type_counts = {}
            for item in defect_items:
                defects = item.get('defects', [])
                for defect in defects:
                    defect_type_counts[defect] = defect_type_counts.get(defect, 0) + 1
            
            # Analyze patterns
            patterns = {
                'total_defects': len(defect_items),
                'defect_types_found': len(defect_type_counts),
                'defect_type_distribution': defect_type_counts,
                'most_common_defect': max(defect_type_counts.items(), key=lambda x: x[1]) if defect_type_counts else None,
                'multi_defect_rate': sum(1 for item in defect_items if len(item.get('defects', [])) > 1) / len(defect_items) * 100
            }
            
            return patterns
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_service_info(self):
        """Get analysis service information"""
        return {
            'service_type': 'AnalysisService',
            'utils_available': UTILS_AVAILABLE,
            'capabilities': {
                'statistics_calculation': True,
                'trend_analysis': True,
                'report_generation': UTILS_AVAILABLE,
                'visualization_creation': UTILS_AVAILABLE,
                'data_export': True,
                'quality_metrics': True,
                'performance_tracking': self.performance_tracker is not None
            },
            'supported_formats': {
                'export_formats': ['json', 'csv', 'xlsx'],
                'report_formats': ['json', 'enhanced_text', 'pdf'] if UTILS_AVAILABLE else ['json'],
                'visualization_formats': ['png'] if UTILS_AVAILABLE else []
            },
            'directories': {
                'reports_dir': str(self.reports_dir),
                'exports_dir': str(self.exports_dir),
                'temp_dir': str(self.temp_dir)
            }
        }