# config.py - FIXED VERSION
"""
Configuration file for Unified Defect Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Model paths - CORRECTED (removed double "models")
ANOMALIB_MODEL_PATH = MODELS_DIR / "patchcore.pt"            # Fixed: removed extra "models/"
HRNET_MODEL_PATH = MODELS_DIR / "defect_segmentation_model.pth"  # Fixed: removed extra "models/"

# Detection thresholds
ANOMALY_THRESHOLD = 0.7
DEFECT_CONFIDENCE_THRESHOLD = 0.85

# Device configuration
DEVICE = 'cuda'  # or 'cpu'

# Enhanced defect detection classes
SPECIFIC_DEFECT_CLASSES = {
    0: "background",
    1: "damaged",
    2: "missing_component", 
    3: "open",
    4: "scratch",
    5: "stained"
}

DEFECT_COLORS = {
    0: (64, 64, 64),     # Dark gray - background
    1: (255, 0, 0),      # Red - damaged
    2: (255, 255, 0),    # Yellow - missing component
    3: (255, 0, 255),    # Magenta - open
    4: (0, 255, 255),    # Cyan - scratch
    5: (128, 0, 128),    # Purple - stained
}

# Video processing settings
VIDEO_FRAME_SKIP = 0  # Skip frames for faster processing
CAMERA_RESOLUTION = (640, 480)
ANALYSIS_INTERVAL = 2.0  # seconds

# Image preprocessing settings
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Minimum detection thresholds
MIN_DEFECT_PIXELS = 50
MIN_DEFECT_PERCENTAGE = 0.005  # 0.5%
MIN_BBOX_AREA = 100

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR / "batch", exist_ok=True)
os.makedirs(OUTPUTS_DIR / "video", exist_ok=True)
os.makedirs(OUTPUTS_DIR / "camera", exist_ok=True)

# Debug: Print actual paths for verification
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"ANOMALIB_MODEL_PATH: {ANOMALIB_MODEL_PATH}")
    print(f"HRNET_MODEL_PATH: {HRNET_MODEL_PATH}")
    print(f"Model files exist:")
    print(f"  Anomalib: {ANOMALIB_MODEL_PATH.exists()}")
    print(f"  HRNet: {HRNET_MODEL_PATH.exists()}")