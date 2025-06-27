# utils/performance_tracker.py - Enhanced performance tracking
"""
Enhanced performance tracking and metrics
"""

import time
import psutil
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.start_times = []
        self.end_times = []
        self.processing_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.decisions = []
        self.anomaly_scores = []
        
    def start_measurement(self):
        """Start timing measurement"""
        start_time = time.time()
        self.start_times.append(start_time)
        
        # Record system metrics
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
        
        return start_time
    
    def end_measurement(self, start_time, result=None):
        """End timing measurement"""
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.end_times.append(end_time)
        self.processing_times.append(processing_time)
        
        if result:
            self.decisions.append(result.get('final_decision', 'Unknown'))
            anomaly_detection = result.get('anomaly_detection', {})
            self.anomaly_scores.append(anomaly_detection.get('anomaly_score', 0))
        
        return processing_time
    
    def get_metrics(self):
        """Get comprehensive performance metrics"""
        if not self.processing_times:
            return {}
        
        times = np.array(self.processing_times)
        
        metrics = {
            'total_tests': len(times),
            'avg_processing_time': float(np.mean(times)),
            'min_processing_time': float(np.min(times)),
            'max_processing_time': float(np.max(times)),
            'std_processing_time': float(np.std(times)),
            'median_processing_time': float(np.median(times)),
            'throughput_fps': float(1 / np.mean(times)),
            'total_processing_time': float(np.sum(times)),
            
            'avg_memory_usage': float(np.mean(self.memory_usage)) if self.memory_usage else 0,
            'avg_cpu_usage': float(np.mean(self.cpu_usage)) if self.cpu_usage else 0,
            
            'good_products': self.decisions.count('GOOD'),
            'defective_products': self.decisions.count('DEFECT'),
            'error_count': self.decisions.count('ERROR'),
            
            'avg_anomaly_score': float(np.mean(self.anomaly_scores)) if self.anomaly_scores else 0,
            'anomaly_score_std': float(np.std(self.anomaly_scores)) if self.anomaly_scores else 0,
        }
        
        # Calculate quality metrics
        if metrics['total_tests'] > 0:
            metrics['good_rate'] = metrics['good_products'] / metrics['total_tests'] * 100
            metrics['defect_rate'] = metrics['defective_products'] / metrics['total_tests'] * 100
            metrics['error_rate'] = metrics['error_count'] / metrics['total_tests'] * 100
        
        return metrics
    
    def create_performance_charts(self, output_dir="outputs"):
        """Create comprehensive performance charts"""
        try:
            if not self.processing_times:
                print("No performance data to chart")
                return None
            
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Create 2x3 subplot layout for comprehensive analysis
            plt.ioff()
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
            fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
            
            # Chart 1: Processing time trend
            iterations = list(range(1, len(self.processing_times) + 1))
            axes[0, 0].plot(iterations, self.processing_times, 'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_title('Processing Time Trend', fontweight='bold')
            axes[0, 0].set_xlabel('Test #')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(self.processing_times) > 2:
                z = np.polyfit(iterations, self.processing_times, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(iterations, p(iterations), "r--", alpha=0.8, label='Trend')
                axes[0, 0].legend()
            
            # Chart 2: Processing time distribution
            axes[0, 1].hist(self.processing_times, bins=max(5, len(self.processing_times)//3), 
                           alpha=0.7, color='skyblue', edgecolor='black')
            mean_time = np.mean(self.processing_times)
            axes[0, 1].axvline(mean_time, color='red', linestyle='--', 
                              label=f'Mean: {mean_time:.3f}s')
            axes[0, 1].set_title('Processing Time Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Chart 3: System resource usage
            if self.memory_usage and self.cpu_usage:
                ax3_twin = axes[0, 2].twinx()
                line1 = axes[0, 2].plot(iterations, self.memory_usage, 'g-o', 
                                       label='Memory %', linewidth=2, markersize=4)
                line2 = ax3_twin.plot(iterations, self.cpu_usage, 'r-s', 
                                     label='CPU %', linewidth=2, markersize=4)
                
                axes[0, 2].set_xlabel('Test #')
                axes[0, 2].set_ylabel('Memory Usage (%)', color='g')
                ax3_twin.set_ylabel('CPU Usage (%)', color='r')
                axes[0, 2].set_title('System Resource Usage', fontweight='bold')
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[0, 2].legend(lines, labels, loc='upper left')
                axes[0, 2].grid(True, alpha=0.3)
            else:
                axes[0, 2].text(0.5, 0.5, 'System metrics\nnot available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('System Resource Usage', fontweight='bold')
            
            # Chart 4: Quality distribution
            if self.decisions:
                decision_counts = {}
                for decision in self.decisions:
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
                
                colors = []
                for decision in decision_counts.keys():
                    if decision == 'GOOD':
                        colors.append('lightgreen')
                    elif decision == 'DEFECT':
                        colors.append('lightcoral')
                    else:
                        colors.append('lightgray')
                
                wedges, texts, autotexts = axes[1, 0].pie(
                    decision_counts.values(), 
                    labels=decision_counts.keys(),
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors
                )
                axes[1, 0].set_title('Quality Distribution', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No quality data\navailable', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Quality Distribution', fontweight='bold')
            
            # Chart 5: Performance metrics summary
            metrics = self.get_metrics()
            metric_names = ['Avg Time', 'Min Time', 'Max Time', 'Throughput\n(FPS)']
            metric_values = [
                metrics['avg_processing_time'],
                metrics['min_processing_time'], 
                metrics['max_processing_time'],
                metrics['throughput_fps']
            ]
            
            bars = axes[1, 1].bar(metric_names, metric_values, 
                                 color=['blue', 'green', 'red', 'orange'], alpha=0.7)
            axes[1, 1].set_title('Performance Metrics', fontweight='bold')
            axes[1, 1].set_ylabel('Time (s) / FPS')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.02,
                               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Chart 6: Anomaly score analysis
            if self.anomaly_scores:
                axes[1, 2].scatter(iterations, self.anomaly_scores, 
                                  c=['red' if d == 'DEFECT' else 'green' for d in self.decisions],
                                  alpha=0.7, s=50)
                axes[1, 2].axhline(y=0.7, color='black', linestyle='--', 
                                  label='Threshold (0.7)')
                axes[1, 2].set_title('Anomaly Score Analysis', fontweight='bold')
                axes[1, 2].set_xlabel('Test #')
                axes[1, 2].set_ylabel('Anomaly Score')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No anomaly score\ndata available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Anomaly Score Analysis', fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = Path(output_dir) / f"performance_analysis_{timestamp}.png"
            
            fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"Performance charts saved: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            print(f"Error creating performance charts: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def save_metrics_report(self, output_dir="outputs"):
        """Save detailed metrics report"""
        try:
            metrics = self.get_metrics()
            
            # Create detailed report
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': metrics,
                'raw_data': {
                    'processing_times': self.processing_times,
                    'memory_usage': self.memory_usage,
                    'cpu_usage': self.cpu_usage,
                    'decisions': self.decisions,
                    'anomaly_scores': self.anomaly_scores
                }
            }
            
            # Save JSON report
            Path(output_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(output_dir) / f"performance_metrics_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"Metrics report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"Error saving metrics report: {e}")
            return None

# Global performance tracker instance
performance_tracker = PerformanceTracker()
