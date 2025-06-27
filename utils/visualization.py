# utils/visualization.py - FIXED VERSION
"""
Visualization utilities for analysis results - FIXED
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
import base64
from io import BytesIO

def create_visualization(result, output_dir):
    """Create comprehensive visualization of analysis results"""
    try:
        # Load original image
        image = cv2.imread(result['image_path'])
        if image is None:
            print(f"Could not load image: {result['image_path']}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots - FIXED: Better layout
        plt.ioff()  # Turn off interactive mode
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle(f'Unified Defect Detection Results\n{os.path.basename(result["image_path"])}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Product Image', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Plot 2: Anomaly detection result
        _plot_anomaly_detection(axes[0, 1], image_rgb, result['anomaly_detection'])
        
        # Plot 3: Defect classification (if available)
        _plot_defect_classification(axes[1, 0], image_rgb, result)
        
        # Plot 4: Combined result with bounding boxes
        _plot_combined_result(axes[1, 1], image_rgb, result)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        # Save visualization with high DPI
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        viz_filename = f"analysis_{timestamp}_{image_name}.png"
        viz_path = os.path.join(output_dir, viz_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with high quality
        fig.savefig(viz_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # IMPORTANT: Close figure to free memory
        plt.close(fig)
        
        print(f"Visualization saved: {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None

def _plot_anomaly_detection(ax, image_rgb, anomaly_result):
    """Plot anomaly detection results with enhanced visualization"""
    # Create overlay image
    overlay = image_rgb.copy()
    
    if anomaly_result['decision'] == 'GOOD':
        # Green tint for good products
        green_overlay = np.zeros_like(overlay)
        green_overlay[:, :, 1] = 50  # Green channel
        overlay = cv2.addWeighted(overlay, 0.8, green_overlay, 0.2, 0)
        title_color = 'green'
        border_color = 'lightgreen'
    else:
        # Red tint for defective products
        red_overlay = np.zeros_like(overlay)
        red_overlay[:, :, 0] = 50  # Red channel
        overlay = cv2.addWeighted(overlay, 0.8, red_overlay, 0.2, 0)
        title_color = 'red'
        border_color = 'lightcoral'
    
    ax.imshow(overlay)
    ax.set_title(f'Anomaly Detection: {anomaly_result["decision"]}\n'
                f'Score: {anomaly_result["anomaly_score"]:.3f} '
                f'(Threshold: {anomaly_result["threshold_used"]:.2f})', 
                fontweight='bold', fontsize=11, color=title_color)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
    
    ax.axis('off')

def _plot_defect_classification(ax, image_rgb, result):
    """Plot defect classification results with detailed info"""
    if result.get('defect_classification') and result['final_decision'] == 'DEFECT':
        defect_result = result['defect_classification']
        
        # Create colored defect mask overlay
        if 'predicted_mask' in defect_result:
            predicted_mask = defect_result['predicted_mask']
            
            # Resize mask to match image if needed
            if predicted_mask.shape[:2] != image_rgb.shape[:2]:
                predicted_mask = cv2.resize(predicted_mask, 
                                          (image_rgb.shape[1], image_rgb.shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
            
            # Create colored overlay
            colored_mask = np.zeros_like(image_rgb)
            for class_id, color in DEFECT_COLORS.items():
                if class_id > 0:  # Skip background
                    class_pixels = (predicted_mask == class_id)
                    colored_mask[class_pixels] = color
            
            # Blend with original image
            if np.any(colored_mask):
                blended = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
            else:
                blended = image_rgb
            
            ax.imshow(blended)
        else:
            ax.imshow(image_rgb)
        
        # Create title with detected defects
        detected_defects = defect_result.get('detected_defects', [])
        if detected_defects:
            defects_text = ', '.join([d.replace('_', ' ').title() for d in detected_defects[:3]])
            if len(detected_defects) > 3:
                defects_text += f' (+{len(detected_defects)-3} more)'
        else:
            defects_text = 'Processing...'
            
        ax.set_title(f'Defect Classification\n{defects_text}', 
                    fontweight='bold', fontsize=11, color='darkred')
        
        # Add red border for defects
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    else:
        # Show original image for good products
        ax.imshow(image_rgb)
        ax.set_title('Defect Classification\nNo Defects Detected', 
                    fontweight='bold', fontsize=11, color='green')
        
        # Add green border for good products
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
    
    ax.axis('off')

def _plot_combined_result(ax, image_rgb, result):
    """Plot combined result with detailed bounding boxes"""
    result_image = image_rgb.copy()
    
    # Add product-level border
    if result['final_decision'] == 'DEFECT':
        # Red border for defective products
        border_color = (255, 0, 0)
        border_thickness = 8
    else:
        # Green border for good products  
        border_color = (0, 255, 0)
        border_thickness = 6
    
    h, w = result_image.shape[:2]
    cv2.rectangle(result_image, (5, 5), (w-5, h-5), border_color, border_thickness)
    
    # Add defect-specific bounding boxes with enhanced visualization
    if (result['final_decision'] == 'DEFECT' and 
        result.get('defect_classification') and 
        'bounding_boxes' in result['defect_classification']):
        
        bboxes = result['defect_classification']['bounding_boxes']
        colors_used = []
        
        for defect_type, bbox_list in bboxes.items():
            if defect_type in result['defect_classification']['class_distribution']:
                class_id = result['defect_classification']['class_distribution'][defect_type]['class_id']
                color = DEFECT_COLORS.get(class_id, (255, 255, 255))
                colors_used.append((defect_type, color))
                
                for i, bbox in enumerate(bbox_list):
                    # Draw bounding box with thick border
                    cv2.rectangle(result_image, 
                                (bbox['x'], bbox['y']), 
                                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                                color, 4)
                    
                    # Add defect label with background
                    label = f"{defect_type.replace('_', ' ').title()}"
                    if len(bbox_list) > 1:
                        label += f" #{i+1}"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Label background
                    cv2.rectangle(result_image, 
                                (bbox['x'], bbox['y'] - label_size[1] - 10),
                                (bbox['x'] + label_size[0] + 10, bbox['y']),
                                color, -1)
                    
                    # Label text
                    cv2.putText(result_image, label,
                              (bbox['x'] + 5, bbox['y'] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    ax.imshow(result_image)
    
    # Enhanced title with processing info
    title = f'Final Result: {result["final_decision"]}\n'
    title += f'Processing Time: {result["processing_time"]:.2f}s'
    
    if result.get('detected_defect_types'):
        title += f'\nDefects: {len(result["detected_defect_types"])} types'
    
    title_color = 'darkred' if result['final_decision'] == 'DEFECT' else 'darkgreen'
    ax.set_title(title, fontweight='bold', fontsize=11, color=title_color)
    ax.axis('off')

def create_performance_charts(performance_data):
    """Create performance analysis charts"""
    try:
        plt.ioff()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
        fig.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Chart 1: Processing time trend
        iterations = list(range(1, len(performance_data['times']) + 1))
        ax1.plot(iterations, performance_data['times'], 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Processing Time per Iteration', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Processing time distribution
        ax2.hist(performance_data['times'], bins=8, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(performance_data['times']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(performance_data["times"]):.3f}s')
        ax2.set_title('Processing Time Distribution', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Performance metrics
        metrics = ['Min', 'Mean', 'Max', 'Std Dev']
        values = [
            min(performance_data['times']),
            np.mean(performance_data['times']),
            max(performance_data['times']),
            np.std(performance_data['times'])
        ]
        
        bars = ax3.bar(metrics, values, color=['green', 'blue', 'red', 'orange'], alpha=0.7)
        ax3.set_title('Performance Metrics Summary', fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Throughput analysis
        throughput = [1/t for t in performance_data['times']]
        ax4.plot(iterations, throughput, 'g-s', linewidth=2, markersize=6)
        ax4.set_title('Processing Throughput', fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Images per Second')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save performance chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f"outputs/performance_analysis_{timestamp}.png"
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating performance charts: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None

def create_test_results_charts(test_results):
    """Create comprehensive test results charts"""
    try:
        plt.ioff()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
        fig.suptitle('Test Results Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data from test results
        decisions = [r.get('final_decision', 'Unknown') for r in test_results]
        scores = [r.get('anomaly_detection', {}).get('anomaly_score', 0) for r in test_results]
        times = [r.get('processing_time', 0) for r in test_results]
        
        # Chart 1: Decision distribution
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        colors = ['lightgreen' if d == 'GOOD' else 'lightcoral' for d in decision_counts.keys()]
        wedges, texts, autotexts = ax1.pie(decision_counts.values(), labels=decision_counts.keys(), 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Product Quality Distribution', fontweight='bold')
        
        # Chart 2: Anomaly score distribution
        ax2.hist(scores, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
        ax2.set_title('Anomaly Score Distribution', fontweight='bold')
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Processing time vs Decision
        good_times = [times[i] for i, d in enumerate(decisions) if d == 'GOOD']
        defect_times = [times[i] for i, d in enumerate(decisions) if d == 'DEFECT']
        
        box_data = []
        labels = []
        if good_times:
            box_data.append(good_times)
            labels.append('GOOD')
        if defect_times:
            box_data.append(defect_times)
            labels.append('DEFECT')
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                
        ax3.set_title('Processing Time by Decision', fontweight='bold')
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Test summary metrics
        total_tests = len(test_results)
        good_count = sum(1 for d in decisions if d == 'GOOD')
        defect_count = sum(1 for d in decisions if d == 'DEFECT')
        avg_time = np.mean(times) if times else 0
        
        metrics = ['Total\nTests', 'Good\nProducts', 'Defective\nProducts', 'Avg Time\n(seconds)']
        values = [total_tests, good_count, defect_count, avg_time]
        colors = ['blue', 'green', 'red', 'orange']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Test Summary Metrics', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if isinstance(value, float):
                label = f'{value:.2f}'
            else:
                label = str(value)
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save test results chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f"outputs/test_results_{timestamp}.png"
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating test results charts: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None
