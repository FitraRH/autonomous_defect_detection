# utils/pdf_generator.py - Enhanced PDF report generation
"""
Professional PDF report generation with auto-download capability
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os
from datetime import datetime
from pathlib import Path
import numpy as np

class PDFReportGenerator:
    """Professional PDF report generator"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkred
        )
        
        # Subheader style
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkgreen
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Metrics style
        self.metrics_style = ParagraphStyle(
            'MetricsStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=4,
            leftIndent=20
        )
    
    def generate_single_image_report(self, result, output_dir="outputs"):
        """Generate PDF report for single image analysis"""
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
            pdf_filename = f"detection_report_{image_name}_{timestamp}.pdf"
            pdf_path = Path(output_dir) / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=72, 
                                  leftMargin=72, topMargin=72, bottomMargin=18)
            
            # Build story (content)
            story = []
            
            # Title
            story.append(Paragraph("Unified Defect Detection Report", self.title_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.header_style))
            
            decision_color = "red" if result['final_decision'] == 'DEFECT' else "green"
            summary_text = f"""
            <para>
            <b>Image:</b> {os.path.basename(result['image_path'])}<br/>
            <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Final Decision:</b> <font color="{decision_color}"><b>{result['final_decision']}</b></font><br/>
            <b>Processing Time:</b> {result['processing_time']:.2f} seconds<br/>
            <b>System:</b> Unified Defect Detection System v1.0
            </para>
            """
            story.append(Paragraph(summary_text, self.body_style))
            story.append(Spacer(1, 20))
            
            # Anomaly Detection Results
            story.append(Paragraph("Stage 1: Anomaly Detection", self.header_style))
            
            anomaly_data = result['anomaly_detection']
            anomaly_text = f"""
            <para>
            <b>Model:</b> Mock Anomalib (Rule-based Analysis)<br/>
            <b>Anomaly Score:</b> {anomaly_data['anomaly_score']:.4f}<br/>
            <b>Threshold Used:</b> {anomaly_data['threshold_used']:.2f}<br/>
            <b>Decision:</b> {anomaly_data['decision']}<br/>
            <b>Has Anomaly Mask:</b> {'Yes' if anomaly_data.get('anomaly_mask') is not None else 'No'}
            </para>
            """
            story.append(Paragraph(anomaly_text, self.body_style))
            story.append(Spacer(1, 15))
            
            # Defect Classification Results
            if result['final_decision'] == 'DEFECT' and result.get('defect_classification'):
                story.append(Paragraph("Stage 2: Defect Classification", self.header_style))
                
                defect_result = result['defect_classification']
                detected_defects = defect_result.get('detected_defects', [])
                
                classification_text = f"""
                <para>
                <b>Model:</b> Enhanced HRNet Segmentation<br/>
                <b>Detected Defect Types:</b> {len(detected_defects)}<br/>
                <b>Defects Found:</b> {', '.join([d.replace('_', ' ').title() for d in detected_defects]) if detected_defects else 'None'}
                </para>
                """
                story.append(Paragraph(classification_text, self.body_style))
                
                # Detailed defect analysis
                if detected_defects:
                    story.append(Paragraph("Detailed Defect Analysis:", self.subheader_style))
                    
                    for i, defect_type in enumerate(detected_defects, 1):
                        if 'defect_analysis' in defect_result:
                            stats = defect_result['defect_analysis'].get('defect_statistics', {}).get(defect_type, {})
                            distribution = defect_result['defect_analysis'].get('class_distribution', {}).get(defect_type, {})
                            
                            defect_text = f"""
                            <para>
                            <b>{i}. {defect_type.replace('_', ' ').title()}</b><br/>
                            • Coverage: {distribution.get('percentage', 0):.2f}% ({distribution.get('pixel_count', 0):,} pixels)<br/>
                            • Average Confidence: {stats.get('avg_confidence', 0):.4f}<br/>
                            • Number of Regions: {stats.get('num_regions', 0)}
                            </para>
                            """
                            story.append(Paragraph(defect_text, self.metrics_style))
                
                story.append(Spacer(1, 15))
            else:
                story.append(Paragraph("Stage 2: Defect Classification", self.header_style))
                story.append(Paragraph("Status: Skipped (Product classified as GOOD)", self.body_style))
                story.append(Paragraph("No defect classification needed.", self.body_style))
                story.append(Spacer(1, 15))
            
            # Add visualization if available
            if result.get('visualization_path') and os.path.exists(result['visualization_path']):
                story.append(Paragraph("Visual Analysis", self.header_style))
                try:
                    # Add image with proper sizing
                    img = Image(result['visualization_path'])
                    img.drawHeight = 6 * inch
                    img.drawWidth = 8 * inch
                    story.append(img)
                    story.append(Spacer(1, 10))
                except Exception as e:
                    story.append(Paragraph(f"Visualization not available: {e}", self.body_style))
                    story.append(Spacer(1, 10))
            
            # Recommendations
            story.append(Paragraph("Recommendations", self.header_style))
            
            if result['final_decision'] == 'DEFECT':
                recommendations = """
                <para>
                <b>DEFECTIVE PRODUCT DETECTED</b><br/>
                • Immediate quality control review required<br/>
                • Investigate root cause of detected defects<br/>
                • Implement corrective actions<br/>
                • Update quality control procedures if needed<br/>
                • Consider stopping production until issues resolved
                </para>
                """
                
                # Add critical defect warnings
                if result.get('detected_defect_types'):
                    severe_defects = [d for d in result['detected_defect_types'] 
                                    if d in ['damaged', 'missing_component']]
                    if severe_defects:
                        recommendations += f"""
                        <para>
                        <b><font color="red">CRITICAL:</font></b> {', '.join(severe_defects)} defects require immediate attention
                        </para>
                        """
            else:
                recommendations = """
                <para>
                <b>GOOD PRODUCT</b><br/>
                • Product meets quality standards<br/>
                • No defects detected<br/>
                • Continue current production processes<br/>
                • Maintain quality control standards<br/>
                • Regular monitoring recommended
                </para>
                """
            
            story.append(Paragraph(recommendations, self.body_style))
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"""
            <para alignment="center">
            <i>Report generated by Unified Defect Detection System v1.0<br/>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
            </para>
            """
            story.append(Paragraph(footer_text, self.body_style))
            
            # Build PDF
            doc.build(story)
            
            print(f"PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
    
    def generate_batch_report(self, batch_results, output_dir="outputs"):
        """Generate comprehensive batch analysis PDF report"""
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"batch_analysis_report_{timestamp}.pdf"
            pdf_path = Path(output_dir) / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=72, 
                                  leftMargin=72, topMargin=72, bottomMargin=18)
            
            story = []
            results = batch_results['results']
            summary = batch_results['summary']
            
            # Title
            story.append(Paragraph("Batch Defect Detection Analysis Report", self.title_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.header_style))
            
            total_images = summary['total_images']
            good_products = summary['good_products']
            defective_products = summary['defective_products']
            defect_rate = (defective_products / total_images * 100) if total_images > 0 else 0
            
            exec_summary = f"""
            <para>
            <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Total Images Processed:</b> {total_images}<br/>
            <b>Good Products:</b> {good_products} ({(good_products/total_images*100):.1f}%)<br/>
            <b>Defective Products:</b> {defective_products} ({defect_rate:.1f}%)<br/>
            <b>Average Processing Time:</b> {summary['avg_processing_time']:.2f} seconds per image<br/>
            <b>Total Processing Duration:</b> {sum(summary['processing_times']):.2f} seconds
            </para>
            """
            story.append(Paragraph(exec_summary, self.body_style))
            story.append(Spacer(1, 20))
            
            # Performance Metrics
            story.append(Paragraph("Performance Metrics", self.header_style))
            
            throughput = total_images / sum(summary['processing_times']) if sum(summary['processing_times']) > 0 else 0
            
            perf_metrics = f"""
            <para>
            <b>Throughput:</b> {throughput:.2f} images per second<br/>
            <b>Minimum Processing Time:</b> {min(summary['processing_times']):.2f} seconds<br/>
            <b>Maximum Processing Time:</b> {max(summary['processing_times']):.2f} seconds<br/>
            <b>Processing Time Std Dev:</b> {np.std(summary['processing_times']):.2f} seconds
            </para>
            """
            story.append(Paragraph(perf_metrics, self.body_style))
            story.append(Spacer(1, 15))
            
            # Defect Analysis
            if summary.get('defect_types_found'):
                story.append(Paragraph("Defect Analysis", self.header_style))
                
                # Count defect occurrences
                defect_counts = {}
                for result in results:
                    if 'detected_defect_types' in result:
                        for defect in result['detected_defect_types']:
                            defect_counts[defect] = defect_counts.get(defect, 0) + 1
                
                defect_text = "<para><b>Defect Type Distribution:</b><br/>"
                for defect_type, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / defective_products) * 100 if defective_products > 0 else 0
                    defect_text += f"• {defect_type.replace('_', ' ').title()}: {count} occurrences ({percentage:.1f}%)<br/>"
                defect_text += "</para>"
                
                story.append(Paragraph(defect_text, self.body_style))
                story.append(Spacer(1, 15))
            
            # Quality Assessment
            story.append(Paragraph("Quality Assessment", self.header_style))
            
            if defect_rate == 0:
                quality_assessment = """
                <para>
                <b><font color="green">EXCELLENT QUALITY</font></b><br/>
                • Zero defects detected across all samples<br/>
                • Current quality control processes are highly effective<br/>
                • Continue monitoring to maintain standards<br/>
                • Consider this batch as quality benchmark
                </para>
                """
            elif defect_rate < 5:
                quality_assessment = f"""
                <para>
                <b><font color="orange">ACCEPTABLE QUALITY</font></b> (Defect Rate: {defect_rate:.1f}%)<br/>
                • Low defect rate within acceptable limits<br/>
                • Monitor common defect patterns<br/>
                • Investigate root causes of detected defects<br/>
                • Implement preventive measures for identified issues
                </para>
                """
            elif defect_rate < 15:
                quality_assessment = f"""
                <para>
                <b><font color="red">MODERATE CONCERNS</font></b> (Defect Rate: {defect_rate:.1f}%)<br/>
                • Defect rate above optimal levels<br/>
                • Immediate investigation required<br/>
                • Review quality control procedures<br/>
                • Implement corrective actions for most common defects
                </para>
                """
            else:
                quality_assessment = f"""
                <para>
                <b><font color="red">CRITICAL QUALITY ISSUES</font></b> (Defect Rate: {defect_rate:.1f}%)<br/>
                • High defect rate requires immediate attention<br/>
                • Stop production until issues are resolved<br/>
                • Comprehensive review of entire quality system<br/>
                • Implement emergency quality measures
                </para>
                """
            
            story.append(Paragraph(quality_assessment, self.body_style))
            story.append(Spacer(1, 20))
            
            # Detailed Results Table (first 10 items)
            story.append(Paragraph("Detailed Results (Sample)", self.header_style))
            
            # Create table data
            table_data = [['#', 'Image Name', 'Decision', 'Score', 'Time (s)', 'Defects']]
            
            for i, result in enumerate(results[:10], 1):  # Show first 10 results
                image_name = os.path.basename(result['image_path'])[:20] + "..." if len(os.path.basename(result['image_path'])) > 20 else os.path.basename(result['image_path'])
                decision = result['final_decision']
                score = f"{result['anomaly_detection']['anomaly_score']:.3f}"
                time_taken = f"{result['processing_time']:.2f}"
                defects = ', '.join(result.get('detected_defect_types', [])[:2])
                if len(result.get('detected_defect_types', [])) > 2:
                    defects += "..."
                
                table_data.append([str(i), image_name, decision, score, time_taken, defects])
            
            # Create and style table
            table = Table(table_data, colWidths=[0.5*inch, 2*inch, 1*inch, 0.8*inch, 0.8*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"""
            <para alignment="center">
            <i>Batch Analysis Report generated by Unified Defect Detection System v1.0<br/>
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            Total of {total_images} images analyzed</i>
            </para>
            """
            story.append(Paragraph(footer_text, self.body_style))
            
            # Build PDF
            doc.build(story)
            
            print(f"Batch PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            print(f"Error generating batch PDF report: {e}")
            return None

# Global PDF generator instance
pdf_generator = PDFReportGenerator()
