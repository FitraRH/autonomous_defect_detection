# run_tests_gui.py
"""
GUI Testing Interface for Unified Defect Detection System
Run: python run_tests_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import time
import cv2
import numpy as np
import threading
from datetime import datetime
import json
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')  # Fix threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'

try:
    from main import UnifiedDefectDetector, create_detector, quick_detect
except ImportError as e:
    print(f"Warning: Could not import main modules: {e}")
    # Create dummy functions for testing
    def create_detector():
        return None
    def quick_detect(image_path):
        return {"status": "error", "message": "Main module not available"}


class DefectDetectionGUI:
    """GUI Testing Interface for Backend"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Defect Detection - Testing Interface")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.detector = None
        self.test_images = []
        self.current_result = None
        self.test_results = {}
        self.is_testing = False
        self.selected_image_path = None
        
        # Create GUI components
        self.create_widgets()
        self.setup_layout()
        
        # Initialize detector after GUI is fully created
        self.root.after(100, self.initialize_detector)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="[DETECT] Unified Defect Detection - Testing Interface", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: System Status
        self.create_system_tab()
        
        # Tab 2: Single Image Testing
        self.create_single_image_tab()
        
        # Tab 3: Batch Testing
        self.create_batch_testing_tab()
        
        # Tab 4: Performance Testing
        self.create_performance_tab()
        
        # Tab 5: Test Results
        self.create_results_tab()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, bg='#ecf0f1')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_system_tab(self):
        """Create system status tab"""
        self.system_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.system_frame, text="[CONFIG] System Status")
        
        # System info frame
        info_frame = ttk.LabelFrame(self.system_frame, text="System Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        # System status labels
        self.system_status_label = tk.Label(info_frame, text="System Status: Initializing...", 
                                           font=('Arial', 12, 'bold'), fg='orange')
        self.system_status_label.pack(anchor='w')
        
        self.device_label = tk.Label(info_frame, text="Device: Unknown", font=('Arial', 10))
        self.device_label.pack(anchor='w')
        
        self.models_label = tk.Label(info_frame, text="Models Loaded: Unknown", font=('Arial', 10))
        self.models_label.pack(anchor='w')
        
        self.anomaly_threshold_label = tk.Label(info_frame, text="Anomaly Threshold: Unknown", font=('Arial', 10))
        self.anomaly_threshold_label.pack(anchor='w')
        
        self.defect_threshold_label = tk.Label(info_frame, text="Defect Threshold: Unknown", font=('Arial', 10))
        self.defect_threshold_label.pack(anchor='w')
        
        # Control buttons
        control_frame = ttk.LabelFrame(self.system_frame, text="System Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="[REFRESH] Refresh System Info", 
                  command=self.refresh_system_info).pack(side='left', padx=5)
        ttk.Button(control_frame, text="⚙️ Reload Models", 
                  command=self.reload_models).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🧪 Run All Tests", 
                  command=self.run_all_tests).pack(side='left', padx=5)
        
        # Log display
        log_frame = ttk.LabelFrame(self.system_frame, text="System Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill='both', expand=True)
        
        # Add initial log message
        self.log_message("System started. Initializing detector...")
    
    def create_single_image_tab(self):
        """Create single image testing tab"""
        self.single_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.single_frame, text="[IMG] Single Image Test")
        
        # Left panel for controls
        left_panel = ttk.Frame(self.single_frame, width=350)
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Image selection
        image_frame = ttk.LabelFrame(left_panel, text="Image Selection", padding=10)
        image_frame.pack(fill='x', pady=5)
        
        ttk.Button(image_frame, text="[FOLDER] Select Image", 
                  command=self.select_image).pack(fill='x', pady=2)
        ttk.Button(image_frame, text="[TARGET] Generate Test Image", 
                  command=self.generate_test_image).pack(fill='x', pady=2)
        ttk.Button(image_frame, text="[CAMERA] Capture from Camera", 
                  command=self.capture_from_camera).pack(fill='x', pady=2)
        
        self.selected_image_label = tk.Label(image_frame, text="No image selected", 
                                           wraplength=300, justify='left', bg='white', relief='sunken')
        self.selected_image_label.pack(fill='x', pady=5, ipady=10)
        
        # Image preview in selection area
        self.preview_label = tk.Label(image_frame, text="Image Preview", 
                                    bg='lightgray', width=25, height=8, relief='sunken')
        self.preview_label.pack(fill='x', pady=5)
        
        # Detection controls
        detection_frame = ttk.LabelFrame(left_panel, text="Detection Controls", padding=10)
        detection_frame.pack(fill='x', pady=5)
        
        self.detect_button = ttk.Button(detection_frame, text="[DETECT] Run Detection", 
                                       command=self.run_single_detection, state='disabled')
        self.detect_button.pack(fill='x', pady=2)
        
        # Progress bar
        self.single_progress = ttk.Progressbar(detection_frame, mode='indeterminate')
        self.single_progress.pack(fill='x', pady=2)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(left_panel, text="Detection Settings", padding=10)
        settings_frame.pack(fill='x', pady=5)
        
        # Sensitivity slider
        tk.Label(settings_frame, text="Sensitivity:").pack(anchor='w')
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = tk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   resolution=0.1, orient='horizontal', variable=self.sensitivity_var)
        sensitivity_scale.pack(fill='x')
        
        # Show processing steps
        self.show_steps_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Show processing steps", 
                      variable=self.show_steps_var).pack(anchor='w')
        
        # Save results
        self.save_results_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Save results to file", 
                      variable=self.save_results_var).pack(anchor='w')
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(results_frame, height=12, width=35)
        self.result_text.pack(fill='both', expand=True)
        
        # Right panel for image display
        right_panel = ttk.Frame(self.single_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Image display frame
        image_display_frame = ttk.LabelFrame(right_panel, text="Image Display", padding=10)
        image_display_frame.pack(fill='both', expand=True)
        
        # Create image display area with tabs
        image_notebook = ttk.Notebook(image_display_frame)
        image_notebook.pack(fill='both', expand=True)
        
        # Original image tab
        original_frame = ttk.Frame(image_notebook)
        image_notebook.add(original_frame, text="Original")
        
        self.original_image_label = tk.Label(original_frame, text="No image loaded", 
                                           bg='white', relief='sunken')
        self.original_image_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Result visualization tab
        result_frame = ttk.Frame(image_notebook)
        image_notebook.add(result_frame, text="Detection Result")
        
        self.result_image_label = tk.Label(result_frame, text="No results yet", 
                                         bg='white', relief='sunken')
        self.result_image_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Processing steps tab
        steps_frame = ttk.Frame(image_notebook)
        image_notebook.add(steps_frame, text="Processing Steps")
        
        self.steps_image_label = tk.Label(steps_frame, text="Processing steps will appear here", 
                                        bg='white', relief='sunken')
        self.steps_image_label.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_batch_testing_tab(self):
        """Create batch testing tab"""
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="[FOLDER] Batch Testing")
        
        # Controls frame
        controls_frame = ttk.LabelFrame(self.batch_frame, text="Batch Testing Controls", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Folder selection
        folder_frame = ttk.Frame(controls_frame)
        folder_frame.pack(fill='x', pady=5)
        
        ttk.Label(folder_frame, text="Input Folder:").pack(side='left')
        self.folder_path_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_path_var, width=50).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.select_folder).pack(side='left')
        
        # Output folder selection
        output_folder_frame = ttk.Frame(controls_frame)
        output_folder_frame.pack(fill='x', pady=5)
        
        ttk.Label(output_folder_frame, text="Output Folder:").pack(side='left')
        self.output_folder_var = tk.StringVar(value="batch_results")
        ttk.Entry(output_folder_frame, textvariable=self.output_folder_var, width=50).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(output_folder_frame, text="Browse", command=self.select_output_folder).pack(side='left')
        
        # Batch controls
        batch_controls_frame = ttk.Frame(controls_frame)
        batch_controls_frame.pack(fill='x', pady=5)
        
        self.batch_test_button = ttk.Button(batch_controls_frame, text="[RUN] Start Batch Test", 
                                           command=self.run_batch_test, state='disabled')
        self.batch_test_button.pack(side='left', padx=5)
        
        ttk.Button(batch_controls_frame, text="[CHART] Generate Test Images", 
                  command=self.generate_batch_test_images).pack(side='left', padx=5)
        
        ttk.Button(batch_controls_frame, text="🛑 Stop Testing", 
                  command=self.stop_batch_test).pack(side='left', padx=5)
        
        # Progress
        self.batch_progress = ttk.Progressbar(controls_frame, mode='determinate')
        self.batch_progress.pack(fill='x', pady=5)
        
        self.batch_status_label = tk.Label(controls_frame, text="Ready for batch testing")
        self.batch_status_label.pack()
        
        # Results display
        results_frame = ttk.LabelFrame(self.batch_frame, text="Batch Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Results tree
        columns = ('Image', 'Expected', 'Actual', 'Score', 'Time', 'Status')
        self.batch_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.batch_tree.heading(col, text=col)
            self.batch_tree.column(col, width=100)
        
        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.batch_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # Summary frame
        summary_frame = ttk.LabelFrame(self.batch_frame, text="Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=4, width=80)
        self.summary_text.pack(fill='x')
    
    def create_performance_tab(self):
        """Create performance testing tab"""
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="[PERF] Performance")
        
        # Performance controls
        perf_controls = ttk.LabelFrame(self.performance_frame, text="Performance Testing", padding=10)
        perf_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(perf_controls, text="[PERF] Run Performance Test", 
                  command=self.run_performance_test).pack(side='left', padx=5)
        ttk.Button(perf_controls, text="[CHART] Benchmark Different Sizes", 
                  command=self.run_size_benchmark).pack(side='left', padx=5)
        ttk.Button(perf_controls, text="[REFRESH] Memory Usage Test", 
                  command=self.run_memory_test).pack(side='left', padx=5)
        
        # Performance results
        perf_results = ttk.LabelFrame(self.performance_frame, text="Performance Metrics", padding=10)
        perf_results.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for performance charts
        self.perf_fig, (self.perf_ax1, self.perf_ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.perf_fig.patch.set_facecolor('white')
        
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_results)
        self.perf_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Performance text results
        self.perf_text = scrolledtext.ScrolledText(perf_results, height=8)
        self.perf_text.pack(fill='x', pady=5)
    
    def create_results_tab(self):
        """Create test results tab"""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="[CHART] Test Results")
        
        # Results controls
        results_controls = ttk.LabelFrame(self.results_frame, text="Results Management", padding=10)
        results_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(results_controls, text="[REPORT] Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(results_controls, text="[SAVE] Save Results", 
                  command=self.save_results).pack(side='left', padx=5)
        ttk.Button(results_controls, text="[FOLDER] Open Output Folder", 
                  command=self.open_output_folder).pack(side='left', padx=5)
        ttk.Button(results_controls, text="[REFRESH] Clear Results", 
                  command=self.clear_all_results).pack(side='left', padx=5)
        
        # Overall results display
        overall_frame = ttk.LabelFrame(self.results_frame, text="Overall Test Results", padding=10)
        overall_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for results visualization
        self.results_fig, ((self.results_ax1, self.results_ax2), 
                          (self.results_ax3, self.results_ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.results_fig.patch.set_facecolor('white')
        
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, overall_frame)
        self.results_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Detailed results text
        details_frame = ttk.LabelFrame(self.results_frame, text="Detailed Results", padding=10)
        details_frame.pack(fill='x', padx=10, pady=5)
        
        self.detailed_results_text = scrolledtext.ScrolledText(details_frame, height=8)
        self.detailed_results_text.pack(fill='both', expand=True)
    
    def setup_layout(self):
        """Setup additional layout configurations"""
        # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def initialize_detector(self):
        """Initialize the detector in a separate thread"""
        def init_thread():
            try:
                self.log_message("Initializing detector...")
                self.detector = create_detector()
                
                if self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready():
                    self.log_message("[OK] Detector initialized successfully!")
                    self.update_system_status("[OK] Ready", "green")
                elif self.detector:
                    self.log_message("[!] Detector created but models not loaded")
                    self.update_system_status("[!] Models Not Loaded", "orange")
                else:
                    self.log_message("[!] Detector not available (main module may be missing)")
                    self.update_system_status("[!] No Detector", "orange")
                
                self.refresh_system_info()
                
            except Exception as e:
                self.log_message(f"[X] Detector initialization failed: {e}")
                self.update_system_status("[X] Failed", "red")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def update_system_status(self, status, color):
        """Update system status label"""
        if hasattr(self, 'system_status_label'):
            self.system_status_label.config(text=f"System Status: {status}", fg=color)
        else:
            print(f"System Status: {status}")
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Check if log_text exists before using it
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
        
        # Check if status_bar exists before using it
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=message)
        
        # Always print to console as backup
        print(f"[{timestamp}] {message}")
        
        # Update GUI
        self.root.update_idletasks()
    
    def refresh_system_info(self):
        """Refresh system information display"""
        if self.detector and hasattr(self.detector, 'get_system_info'):
            try:
                info = self.detector.get_system_info()
                self.device_label.config(text=f"Device: {info.get('device', 'Unknown')}")
                self.models_label.config(text=f"Models Loaded: {'[OK] Yes' if info.get('models_loaded', False) else '[X] No'}")
                self.anomaly_threshold_label.config(text=f"Anomaly Threshold: {info.get('anomaly_threshold', 'Unknown')}")
                self.defect_threshold_label.config(text=f"Defect Threshold: {info.get('defect_threshold', 'Unknown')}")
                
                # Enable/disable buttons based on system readiness
                state = 'normal' if info.get('system_ready', False) else 'disabled'
                if hasattr(self, 'detect_button'):
                    self.detect_button.config(state=state if self.selected_image_path else 'disabled')
                if hasattr(self, 'batch_test_button'):
                    self.batch_test_button.config(state=state if self.folder_path_var.get() else 'disabled')
                
            except Exception as e:
                self.log_message(f"Error refreshing system info: {e}")
        else:
            # Set default values when detector is not available
            self.device_label.config(text="Device: Not Available")
            self.models_label.config(text="Models Loaded: [X] No")
            self.anomaly_threshold_label.config(text="Anomaly Threshold: N/A")
            self.defect_threshold_label.config(text="Defect Threshold: N/A")
    
    def reload_models(self):
        """Reload models"""
        def reload_thread():
            try:
                self.log_message("Reloading models...")
                if self.detector and hasattr(self.detector, 'load_models'):
                    success = self.detector.load_models()
                    if success:
                        self.log_message("[OK] Models reloaded successfully!")
                        self.update_system_status("[OK] Ready", "green")
                    else:
                        self.log_message("[X] Model reloading failed")
                        self.update_system_status("[X] Model Load Failed", "red")
                else:
                    self.initialize_detector()
                
                self.refresh_system_info()
                
            except Exception as e:
                self.log_message(f"[X] Model reload error: {e}")
        
        threading.Thread(target=reload_thread, daemon=True).start()
    
    def select_image(self):
        """Select image for single testing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            filename = os.path.basename(file_path)
            self.selected_image_label.config(text=f"Selected: {filename}")
            
            # Display image in preview and main display
            self.display_image(file_path, self.preview_label, max_size=(200, 150))
            self.display_image(file_path, self.original_image_label, max_size=(600, 400))
            
            # Enable detection button if detector is ready
            detector_ready = self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready()
            self.detect_button.config(state='normal' if detector_ready else 'disabled')
            
            self.log_message(f"Image selected: {filename}")
    
    def capture_from_camera(self):
        """Capture image from camera"""
        try:
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Create camera window
            self.create_camera_window(cap)
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera capture failed: {e}")
            self.log_message(f"[X] Camera capture error: {e}")
    
    def create_camera_window(self, cap):
        """Create camera capture window"""
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Camera Capture")
        camera_window.geometry("800x600")
        
        # Camera display
        camera_label = tk.Label(camera_window)
        camera_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(camera_window)
        button_frame.pack(pady=10)
        
        captured_image = None
        
        def update_camera():
            ret, frame = cap.read()
            if ret:
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                camera_label.config(image=photo)
                camera_label.image = photo
                
                # Store current frame for capture
                nonlocal captured_image
                captured_image = frame.copy()
                
            camera_window.after(30, update_camera)
        
        def capture_image():
            if captured_image is not None:
                # Save captured image
                os.makedirs("captured_images", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_path = f"captured_images/capture_{timestamp}.jpg"
                cv2.imwrite(capture_path, captured_image)
                
                # Set as selected image
                self.selected_image_path = capture_path
                filename = os.path.basename(capture_path)
                self.selected_image_label.config(text=f"Captured: {filename}")
                
                # Display in main interface
                self.display_image(capture_path, self.preview_label, max_size=(200, 150))
                self.display_image(capture_path, self.original_image_label, max_size=(600, 400))
                
                # Enable detection button
                detector_ready = self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready()
                self.detect_button.config(state='normal' if detector_ready else 'disabled')
                
                self.log_message(f"Image captured: {filename}")
                camera_window.destroy()
        
        def close_camera():
            cap.release()
            camera_window.destroy()
        
        ttk.Button(button_frame, text="[CAMERA] Capture", command=capture_image).pack(side='left', padx=5)
        ttk.Button(button_frame, text="[X] Cancel", command=close_camera).pack(side='left', padx=5)
        
        # Start camera update
        update_camera()
        
        # Handle window close
        camera_window.protocol("WM_DELETE_WINDOW", close_camera)
    
    def generate_test_image(self):
        """Generate a test image"""
        # Create a simple test image
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(test_img, (150, 100), (490, 380), (180, 180, 220), -1)
        cv2.putText(test_img, "TEST PRODUCT", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        # Add some defects randomly
        import random
        if random.choice([True, False]):
            # Add scratch
            cv2.line(test_img, (200, 150), (400, 200), (50, 50, 50), 3)
            cv2.putText(test_img, "DEFECT", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Save test image
        os.makedirs("temp_test_images", exist_ok=True)
        test_path = f"temp_test_images/test_image_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(test_path, test_img)
        
        self.selected_image_path = test_path
        filename = os.path.basename(test_path)
        self.selected_image_label.config(text=f"Generated: {filename}")
        
        # Display image in preview and main display
        self.display_image(test_path, self.preview_label, max_size=(200, 150))
        self.display_image(test_path, self.original_image_label, max_size=(600, 400))
        
        # Enable detection button if detector is ready
        detector_ready = self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready()
        self.detect_button.config(state='normal' if detector_ready else 'disabled')
        
        self.log_message(f"Test image generated: {filename}")
    
    def display_image(self, image_path, label_widget, max_size=(300, 200)):
        """Display image in label widget"""
        try:
            # Load and resize image
            image = Image.open(image_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            label_widget.config(image=photo, text="")
            label_widget.image = photo  # Keep a reference
            
        except Exception as e:
            label_widget.config(text=f"Error loading image: {e}")
            self.log_message(f"Error displaying image: {e}")
    
    def run_single_detection(self):
        """Run detection on single image"""
        if not hasattr(self, 'selected_image_path') or not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        def detection_thread():
            try:
                self.single_progress.start()
                self.detect_button.config(state='disabled')
                
                filename = os.path.basename(self.selected_image_path)
                self.log_message(f"Running detection on {filename}...")
                
                start_time = time.time()
                
                # Use detector if available, otherwise use quick_detect
                if self.detector and hasattr(self.detector, 'process_image'):
                    result = self.detector.process_image(self.selected_image_path)
                else:
                    result = quick_detect(self.selected_image_path)
                
                processing_time = time.time() - start_time
                
                if result and isinstance(result, dict):
                    # Add processing time to result
                    result['processing_time'] = processing_time
                    self.current_result = result
                    self.display_single_result(result)
                    
                    # Try to display visualization if available
                    if result.get('visualization_path') and os.path.exists(result['visualization_path']):
                        self.display_image(result['visualization_path'], self.result_image_label, max_size=(600, 400))
                    
                    # Display processing steps if available
                    if result.get('steps_visualization') and os.path.exists(result['steps_visualization']):
                        self.display_image(result['steps_visualization'], self.steps_image_label, max_size=(600, 400))
                    
                    self.log_message(f"[OK] Detection completed in {processing_time:.2f}s")
                    
                    # Save results if enabled
                    if self.save_results_var.get():
                        self.save_single_result(result)
                        
                else:
                    self.log_message("[X] Detection failed or returned invalid result")
                    messagebox.showerror("Error", "Detection failed")
                
            except Exception as e:
                self.log_message(f"[X] Detection error: {e}")
                messagebox.showerror("Error", f"Detection error: {e}")
            
            finally:
                self.single_progress.stop()
                self.detect_button.config(state='normal')
        
        threading.Thread(target=detection_thread, daemon=True).start()
    
    def display_single_result(self, result):
        """Display single detection result"""
        self.result_text.delete(1.0, tk.END)
        
        image_name = os.path.basename(result.get('image_path', 'Unknown'))
        final_decision = result.get('final_decision', 'Unknown')
        processing_time = result.get('processing_time', 0)
        
        output = f"""DETECTION RESULTS
================
Image: {image_name}
Final Decision: {final_decision}
Processing Time: {processing_time:.2f}s

"""
        
        # Anomaly detection results
        if 'anomaly_detection' in result:
            anomaly = result['anomaly_detection']
            output += f"""ANOMALY DETECTION:
Score: {anomaly.get('anomaly_score', 0):.4f}
Threshold: {anomaly.get('threshold_used', 0):.2f}
Decision: {anomaly.get('decision', 'Unknown')}

"""
        
        # Defect classification results
        if final_decision == 'DEFECT' and result.get('defect_classification'):
            defect_result = result['defect_classification']
            detected_defects = defect_result.get('detected_defects', [])
            
            output += f"""DEFECT CLASSIFICATION:
Detected Defects: {', '.join(detected_defects) if detected_defects else 'None'}
Total Defect Types: {len(detected_defects)}

"""
            
            # Bounding boxes information
            bboxes = defect_result.get('bounding_boxes', {})
            if bboxes:
                output += "BOUNDING BOXES:\n"
                for defect_type, boxes in bboxes.items():
                    output += f"  {defect_type}: {len(boxes)} boxes\n"
        
        # Additional metrics if available
        if 'confidence_scores' in result:
            output += f"\nCONFIDENCE SCORES:\n"
            for metric, score in result['confidence_scores'].items():
                output += f"  {metric}: {score:.3f}\n"
        
        self.result_text.insert(tk.END, output)
    
    def save_single_result(self, result):
        """Save single detection result to file"""
        try:
            os.makedirs("detection_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"detection_results/result_{timestamp}.json"
            
            # Prepare result for saving
            save_result = result.copy()
            save_result['timestamp'] = timestamp
            save_result['image_name'] = os.path.basename(result.get('image_path', ''))
            
            with open(result_file, 'w') as f:
                json.dump(save_result, f, indent=2, default=str)
            
            self.log_message(f"Results saved to {result_file}")
            
        except Exception as e:
            self.log_message(f"Error saving results: {e}")
    
    def select_folder(self):
        """Select folder for batch testing"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        
        if folder_path:
            self.folder_path_var.set(folder_path)
            
            # Count images in folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_count = sum(1 for f in os.listdir(folder_path) 
                            if any(f.lower().endswith(ext) for ext in image_extensions))
            
            self.batch_status_label.config(text=f"Found {image_count} images in selected folder")
            
            # Enable batch test button if detector is ready
            detector_ready = self.detector and hasattr(self.detector, 'is_ready') and self.detector.is_ready()
            self.batch_test_button.config(state='normal' if detector_ready and image_count > 0 else 'disabled')
            
            self.log_message(f"Batch folder selected: {image_count} images found")
    
    def select_output_folder(self):
        """Select output folder for batch results"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_var.set(folder_path)
    
    def generate_batch_test_images(self):
        """Generate test images for batch testing"""
        output_dir = filedialog.askdirectory(title="Select Output Directory for Test Images")
        
        if not output_dir:
            return
        
        def generate_thread():
            try:
                self.log_message("Generating batch test images...")
                
                # Generate different types of test images
                test_types = [
                    ("good_product", "GOOD", lambda img: img),  # No defects
                    ("scratched", "DEFECT", lambda img: self.add_scratch(img)),
                    ("stained", "DEFECT", lambda img: self.add_stain(img)),
                    ("damaged", "DEFECT", lambda img: self.add_damage(img)),
                    ("missing_part", "DEFECT", lambda img: self.add_missing_part(img)),
                ]
                
                total_images = len(test_types) * 3
                generated = 0
                
                for i, (name, expected, modifier) in enumerate(test_types):
                    for j in range(3):  # Generate 3 of each type
                        # Create base image
                        base_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
                        cv2.rectangle(base_img, (150, 100), (490, 380), (180, 180, 220), -1)
                        cv2.putText(base_img, f"{name.upper()}", (200, 230), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                        cv2.putText(base_img, f"Sample {j+1}", (200, 270), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                        
                        # Apply modifier
                        modified_img = modifier(base_img.copy())
                        
                        # Save image
                        filename = f"{name}_{j+1}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, modified_img)
                        
                        generated += 1
                        progress = (generated / total_images) * 100
                        self.batch_status_label.config(text=f"Generated {generated}/{total_images} images ({progress:.1f}%)")
                        self.root.update_idletasks()
                
                self.log_message(f"[OK] Generated {total_images} test images in {output_dir}")
                self.folder_path_var.set(output_dir)
                self.batch_status_label.config(text=f"Generated {total_images} test images")
                
            except Exception as e:
                self.log_message(f"[X] Error generating test images: {e}")
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def add_scratch(self, img):
        """Add scratch to image"""
        cv2.line(img, (200, 150), (400, 200), (50, 50, 50), 3)
        cv2.line(img, (220, 160), (380, 190), (60, 60, 60), 2)
        return img
    
    def add_stain(self, img):
        """Add stain to image"""
        cv2.circle(img, (300, 200), 25, (80, 60, 40), -1)
        cv2.circle(img, (320, 180), 15, (90, 70, 50), -1)
        return img
    
    def add_damage(self, img):
        """Add damage to image"""
        pts = np.array([[250, 150], [280, 140], [320, 170], [300, 200], [260, 190]], np.int32)
        cv2.fillPoly(img, [pts], (20, 20, 20))
        return img
    
    def add_missing_part(self, img):
        """Add missing part to image"""
        cv2.rectangle(img, (350, 180), (420, 220), (0, 0, 0), -1)
        cv2.putText(img, "MISSING", (355, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img
    
    def run_batch_test(self):
        """Run batch testing"""
        folder_path = self.folder_path_var.get()
        if not folder_path:
            messagebox.showwarning("Warning", "Please select a folder first")
            return
        
        self.is_testing = True
        
        def batch_thread():
            try:
                self.batch_test_button.config(state='disabled')
                self.clear_batch_results()
                
                # Find all image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_files = []
                for f in os.listdir(folder_path):
                    if any(f.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(folder_path, f))
                
                if not image_files:
                    self.log_message("[X] No image files found in selected folder")
                    return
                
                total_images = len(image_files)
                self.batch_progress['maximum'] = total_images
                
                results = []
                correct_predictions = 0
                total_time = 0
                
                for i, image_path in enumerate(image_files):
                    if not self.is_testing:  # Check if testing was stopped
                        break
                        
                    try:
                        filename = os.path.basename(image_path)
                        self.batch_status_label.config(text=f"Processing {i+1}/{total_images}: {filename}")
                        
                        # Determine expected result from filename
                        expected = "GOOD" if "good" in filename.lower() else "DEFECT"
                        
                        start_time = time.time()
                        
                        # Run detection
                        if self.detector and hasattr(self.detector, 'process_image'):
                            result = self.detector.process_image(image_path)
                        else:
                            result = quick_detect(image_path)
                        
                        processing_time = time.time() - start_time
                        total_time += processing_time
                        
                        if result:
                            actual = result.get('final_decision', 'ERROR')
                            score = result.get('anomaly_detection', {}).get('anomaly_score', 0)
                            status = "[OK] CORRECT" if actual == expected else "[X] WRONG"
                            
                            if actual == expected:
                                correct_predictions += 1
                        else:
                            actual = "ERROR"
                            score = 0
                            status = "[X] ERROR"
                        
                        # Add to results tree
                        self.batch_tree.insert('', 'end', values=(
                            filename, expected, actual, f"{score:.3f}", 
                            f"{processing_time:.2f}s", status
                        ))
                        
                        results.append({
                            'filename': filename,
                            'expected': expected,
                            'actual': actual,
                            'score': score,
                            'time': processing_time,
                            'correct': actual == expected
                        })
                        
                        # Update progress
                        self.batch_progress['value'] = i + 1
                        self.root.update_idletasks()
                        
                    except Exception as e:
                        self.log_message(f"Error processing {filename}: {e}")
                        self.batch_tree.insert('', 'end', values=(
                            filename, "Unknown", "ERROR", "0", "0", "[X] ERROR"
                        ))
                
                # Calculate summary statistics
                if results:
                    accuracy = (correct_predictions / len(results)) * 100
                    avg_time = total_time / len(results)
                    
                    summary = f"""BATCH TEST SUMMARY
===================
Total Images: {len(results)}
Correct Predictions: {correct_predictions}
Accuracy: {accuracy:.1f}%
Average Processing Time: {avg_time:.2f}s
Total Processing Time: {total_time:.2f}s
"""
                    
                    self.summary_text.delete(1.0, tk.END)
                    self.summary_text.insert(tk.END, summary)
                    
                    self.log_message(f"[OK] Batch test completed: {accuracy:.1f}% accuracy")
                else:
                    self.log_message("[X] No results generated")
                
                self.batch_status_label.config(text="Batch test completed")
                
            except Exception as e:
                self.log_message(f"[X] Error during batch test: {e}")
            
            finally:
                self.batch_test_button.config(state='normal')
                self.is_testing = False
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    def stop_batch_test(self):
        """Stop batch testing"""
        self.is_testing = False
        self.log_message("Batch testing stopped by user")
        self.batch_status_label.config(text="Batch testing stopped")
    
    def clear_batch_results(self):
        """Clear batch test results"""
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self.summary_text.delete(1.0, tk.END)
        self.batch_progress['value'] = 0
    
    def run_performance_test(self):
        """Run performance testing"""
        def perf_thread():
            try:
                self.log_message("Running performance test...")
                self.perf_text.delete(1.0, tk.END)
                
                # Generate test image for performance testing
                test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
                test_path = "temp_perf_test.jpg"
                cv2.imwrite(test_path, test_img)
                
                times = []
                
                # Run multiple iterations
                for i in range(10):
                    start_time = time.time()
                    
                    if self.detector and hasattr(self.detector, 'process_image'):
                        result = self.detector.process_image(test_path)
                    else:
                        result = quick_detect(test_path)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    self.perf_text.insert(tk.END, f"Iteration {i+1}: {times[-1]:.3f}s\n")
                    self.perf_text.see(tk.END)
                    self.root.update_idletasks()
                
                # Calculate statistics
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                std_time = np.std(times)
                
                self.perf_text.insert(tk.END, f"\nPERFORMANCE SUMMARY:\n")
                self.perf_text.insert(tk.END, f"Average Time: {avg_time:.3f}s\n")
                self.perf_text.insert(tk.END, f"Min Time: {min_time:.3f}s\n")
                self.perf_text.insert(tk.END, f"Max Time: {max_time:.3f}s\n")
                self.perf_text.insert(tk.END, f"Std Dev: {std_time:.3f}s\n")
                self.perf_text.insert(tk.END, f"FPS: {1/avg_time:.1f}\n")
                
                # Plot performance chart
                self.perf_ax1.clear()
                self.perf_ax1.plot(range(1, 11), times, 'b-o')
                self.perf_ax1.set_title('Processing Time per Iteration')
                self.perf_ax1.set_xlabel('Iteration')
                self.perf_ax1.set_ylabel('Time (seconds)')
                self.perf_ax1.grid(True)
                
                # Plot histogram
                self.perf_ax2.clear()
                self.perf_ax2.hist(times, bins=5, alpha=0.7)
                self.perf_ax2.set_title('Processing Time Distribution')
                self.perf_ax2.set_xlabel('Time (seconds)')
                self.perf_ax2.set_ylabel('Frequency')
                
                self.perf_canvas.draw()
                
                # Clean up
                if os.path.exists(test_path):
                    os.remove(test_path)
                
                self.log_message(f"[OK] Performance test completed: {avg_time:.3f}s average")
                
            except Exception as e:
                self.log_message(f"[X] Performance test error: {e}")
        
        threading.Thread(target=perf_thread, daemon=True).start()
    
    def run_size_benchmark(self):
        """Run benchmark with different image sizes"""
        def benchmark_thread():
            try:
                self.log_message("Running size benchmark...")
                
                sizes = [(320, 240), (640, 480), (800, 600), (1024, 768), (1280, 1024)]
                size_times = []
                
                for width, height in sizes:
                    # Generate test image of specific size
                    test_img = np.ones((height, width, 3), dtype=np.uint8) * 200
                    test_path = f"temp_size_test_{width}x{height}.jpg"
                    cv2.imwrite(test_img, test_path)
                    
                    # Time the detection
                    start_time = time.time()
                    
                    if self.detector and hasattr(self.detector, 'process_image'):
                        result = self.detector.process_image(test_path)
                    else:
                        result = quick_detect(test_path)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    size_times.append(processing_time)
                    
                    self.perf_text.insert(tk.END, f"{width}x{height}: {processing_time:.3f}s\n")
                    self.perf_text.see(tk.END)
                    self.root.update_idletasks()
                    
                    # Clean up
                    if os.path.exists(test_path):
                        os.remove(test_path)
                
                # Plot size benchmark
                self.perf_ax1.clear()
                size_labels = [f"{w}x{h}" for w, h in sizes]
                self.perf_ax1.bar(size_labels, size_times)
                self.perf_ax1.set_title('Processing Time vs Image Size')
                self.perf_ax1.set_xlabel('Image Size')
                self.perf_ax1.set_ylabel('Time (seconds)')
                self.perf_ax1.tick_params(axis='x', rotation=45)
                
                self.perf_canvas.draw()
                
                self.log_message("[OK] Size benchmark completed")
                
            except Exception as e:
                self.log_message(f"[X] Size benchmark error: {e}")
        
        threading.Thread(target=benchmark_thread, daemon=True).start()
    
    def run_memory_test(self):
        """Run memory usage test"""
        def memory_thread():
            try:
                import psutil
                process = psutil.Process()
                
                self.log_message("Running memory usage test...")
                
                # Get initial memory
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Generate and process multiple images
                for i in range(5):
                    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    test_path = f"temp_memory_test_{i}.jpg"
                    cv2.imwrite(test_path, test_img)
                    
                    if self.detector and hasattr(self.detector, 'process_image'):
                        result = self.detector.process_image(test_path)
                    else:
                        result = quick_detect(test_path)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    self.perf_text.insert(tk.END, f"After image {i+1}: {current_memory:.1f} MB\n")
                    
                    if os.path.exists(test_path):
                        os.remove(test_path)
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = final_memory - initial_memory
                
                self.perf_text.insert(tk.END, f"\nMEMORY SUMMARY:\n")
                self.perf_text.insert(tk.END, f"Initial: {initial_memory:.1f} MB\n")
                self.perf_text.insert(tk.END, f"Final: {final_memory:.1f} MB\n")
                self.perf_text.insert(tk.END, f"Increase: {memory_increase:.1f} MB\n")
                
                self.log_message(f"[OK] Memory test completed: {memory_increase:.1f} MB increase")
                
            except ImportError:
                self.log_message("[X] psutil not available for memory testing")
            except Exception as e:
                self.log_message(f"[X] Memory test error: {e}")
        
        threading.Thread(target=memory_thread, daemon=True).start()
    
    def run_all_tests(self):
        """Run all available tests"""
        def all_tests_thread():
            try:
                self.log_message("Starting comprehensive test suite...")
                
                # Run performance test
                self.run_performance_test()
                time.sleep(2)  # Wait for completion
                
                # Run size benchmark
                self.run_size_benchmark()
                time.sleep(2)
                
                # Run memory test
                self.run_memory_test()
                
                self.log_message("[OK] All tests completed")
                
            except Exception as e:
                self.log_message(f"[X] Error running all tests: {e}")
        
        threading.Thread(target=all_tests_thread, daemon=True).start()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"test_report_{timestamp}.html"
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Defect Detection Test Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .result {{ padding: 10px; margin: 5px 0; }}
        .success {{ background-color: #d4edda; }}
        .error {{ background-color: #f8d7da; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>[DETECT] Defect Detection System Test Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>System Information</h2>
        <p><strong>Status:</strong> {self.system_status_label.cget('text')}</p>
        <p><strong>Device:</strong> {self.device_label.cget('text')}</p>
        <p><strong>Models:</strong> {self.models_label.cget('text')}</p>
    </div>
    
    <div class="section">
        <h2>Test Results Summary</h2>
        <p>Comprehensive testing completed with automated test suite.</p>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>Monitor system performance regularly</li>
            <li>Update models when new data becomes available</li>
            <li>Run batch tests before deployment</li>
        </ul>
    </div>
</body>
</html>
"""
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            self.log_message(f"[OK] Report generated: {report_file}")
            
            # Open report in default browser
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_file)}")
            
        except Exception as e:
            self.log_message(f"[X] Error generating report: {e}")
    
    def save_results(self):
        """Save all test results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"test_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save system log
            log_content = self.log_text.get(1.0, tk.END)
            with open(os.path.join(output_dir, "system_log.txt"), 'w') as f:
                f.write(log_content)
            
            # Save batch results if available
            if self.batch_tree.get_children():
                batch_results = []
                for item in self.batch_tree.get_children():
                    values = self.batch_tree.item(item)['values']
                    batch_results.append({
                        'image': values[0],
                        'expected': values[1],
                        'actual': values[2],
                        'score': values[3],
                        'time': values[4],
                        'status': values[5]
                    })
                
                with open(os.path.join(output_dir, "batch_results.json"), 'w') as f:
                    json.dump(batch_results, f, indent=2)
            
            # Save current single result if available
            if self.current_result:
                with open(os.path.join(output_dir, "latest_single_result.json"), 'w') as f:
                    json.dump(self.current_result, f, indent=2, default=str)
            
            self.log_message(f"[OK] Results saved to {output_dir}")
            messagebox.showinfo("Success", f"Results saved to {output_dir}")
            
        except Exception as e:
            self.log_message(f"[X] Error saving results: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        try:
            output_folder = self.output_folder_var.get() or "."
            
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.startfile(output_folder)
            elif system == "Darwin":  # macOS
                os.system(f"open '{output_folder}'")
            else:  # Linux
                os.system(f"xdg-open '{output_folder}'")
                
            self.log_message(f"Opened output folder: {output_folder}")
            
        except Exception as e:
            self.log_message(f"[X] Error opening output folder: {e}")
    
    def clear_all_results(self):
        """Clear all test results"""
        try:
            # Clear batch results
            self.clear_batch_results()
            
            # Clear single image results
            self.result_text.delete(1.0, tk.END)
            self.original_image_label.config(image="", text="No image loaded")
            self.result_image_label.config(image="", text="No results yet")
            self.steps_image_label.config(image="", text="Processing steps will appear here")
            self.preview_label.config(image="", text="Image Preview")
            self.selected_image_label.config(text="No image selected")
            
            # Clear performance results
            self.perf_text.delete(1.0, tk.END)
            self.perf_ax1.clear()
            self.perf_ax2.clear()
            self.perf_canvas.draw()
            
            # Clear detailed results
            self.detailed_results_text.delete(1.0, tk.END)
            
            # Clear results visualization
            for ax in [self.results_ax1, self.results_ax2, self.results_ax3, self.results_ax4]:
                ax.clear()
            self.results_canvas.draw()
            
            # Reset variables
            self.selected_image_path = None
            self.current_result = None
            self.test_results = {}
            
            self.log_message("[OK] All results cleared")
            
        except Exception as e:
            self.log_message(f"[X] Error clearing results: {e}")


def main():
    """Main function to run the GUI"""
    try:
        # Create main window
        root = tk.Tk()
        
        # Set window icon if available
        try:
            # You can add an icon file here
            # root.iconbitmap('icon.ico')
            pass
        except:
            pass
        
        # Create and run application
        app = DefectDetectionGUI(root)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Handle window close
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                # Stop any running tests
                if hasattr(app, 'is_testing'):
                    app.is_testing = False
                
                # Clean up temporary files
                temp_dirs = ["temp_test_images", "captured_images"]
                for temp_dir in temp_dirs:
                    if os.path.exists(temp_dir):
                        try:
                            import shutil
                            shutil.rmtree(temp_dir)
                        except:
                            pass
                
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the application
        print("Starting Defect Detection GUI...")
        print("GUI is ready!")
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        messagebox.showerror("Startup Error", f"Failed to start GUI: {e}")


if __name__ == "__main__":
    main()