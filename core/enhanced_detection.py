# core/enhanced_detection.py - FIXED VERSION
"""
Enhanced detection with precise bounding box analysis for specific defect types
FIXED: All imports and undefined variables
"""

# Required imports
import cv2
import numpy as np

# Import configuration constants
try:
    from config import (
        SPECIFIC_DEFECT_CLASSES,
        DEFECT_CONFIDENCE_THRESHOLD,
        MIN_DEFECT_PIXELS,
        MIN_DEFECT_PERCENTAGE,
        MIN_BBOX_AREA
    )
except ImportError:
    # Fallback constants if config import fails
    print("Warning: Could not import from config, using fallback constants")
    
    SPECIFIC_DEFECT_CLASSES = {
        0: "background",
        1: "damaged",
        2: "missing_component", 
        3: "open",
        4: "scratch",
        5: "stained"
    }
    
    DEFECT_CONFIDENCE_THRESHOLD = 0.85
    MIN_DEFECT_PIXELS = 50
    MIN_DEFECT_PERCENTAGE = 0.005
    MIN_BBOX_AREA = 100

def analyze_defect_predictions_enhanced(predicted_mask, confidence_scores, image_shape):
    """Enhanced defect prediction analysis with better bounding box precision"""
    h, w = predicted_mask.shape
    total_pixels = h * w
    
    analysis = {
        'detected_defects': [],
        'class_distribution': {},
        'bounding_boxes': {},
        'defect_statistics': {},
        'spatial_analysis': {}  # New: spatial distribution analysis
    }
    
    # Analyze each defect class with enhanced precision
    for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
        class_mask = (predicted_mask == class_id)
        pixel_count = np.sum(class_mask)
        percentage = (pixel_count / total_pixels) * 100
        
        analysis['class_distribution'][class_name] = {
            'pixel_count': int(pixel_count),
            'percentage': percentage,
            'class_id': class_id
        }
        
        # Enhanced processing for actual defects (not background)
        if class_id > 0 and pixel_count > 0:
            # Apply confidence threshold with adaptive filtering
            base_threshold = DEFECT_CONFIDENCE_THRESHOLD
            
            # Adaptive threshold based on defect type
            if class_name in ['missing_component', 'damaged']:
                # More strict for critical defects
                adaptive_threshold = base_threshold + 0.05
            else:
                # Standard threshold for surface defects
                adaptive_threshold = base_threshold
            
            confident_mask = class_mask & (confidence_scores > adaptive_threshold)
            confident_pixels = np.sum(confident_mask)
            
            # Enhanced detection criteria
            min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
            
            # Size-based filtering for different defect types
            if class_name == 'missing_component':
                # Missing components usually larger
                min_pixels = max(min_pixels, total_pixels * 0.01)  # At least 1%
            elif class_name in ['scratch', 'stained']:
                # Surface defects can be smaller
                min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * 0.002)  # 0.2%
            
            if confident_pixels > min_pixels or (pixel_count > total_pixels * 0.05):
                analysis['detected_defects'].append(class_name)
                
                # Extract enhanced bounding boxes with clustering
                working_mask = confident_mask if confident_pixels > min_pixels else class_mask
                bboxes = extract_enhanced_bounding_boxes(working_mask, class_name)
                analysis['bounding_boxes'][class_name] = bboxes
                
                # Enhanced statistics
                analysis['defect_statistics'][class_name] = {
                    'confident_pixels': int(confident_pixels if confident_pixels > 0 else pixel_count),
                    'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                    'avg_confidence': float(np.mean(confidence_scores[working_mask])),
                    'max_confidence': float(np.max(confidence_scores[working_mask])),
                    'num_regions': len(bboxes),
                    'largest_region_area': max([bbox['area'] for bbox in bboxes]) if bboxes else 0,
                    'total_defect_area': sum([bbox['area'] for bbox in bboxes]) if bboxes else 0
                }
                
                # Spatial analysis - where defects are located
                analysis['spatial_analysis'][class_name] = analyze_spatial_distribution(
                    working_mask, image_shape, bboxes
                )
    
    return analysis

def extract_enhanced_bounding_boxes(mask, defect_type):
    """Extract enhanced bounding boxes with clustering and filtering"""
    try:
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Use different morphological operations based on defect type
        if defect_type in ['scratch']:
            # For scratches, use opening to remove noise but preserve lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        elif defect_type in ['stained']:
            # For stains, use closing to fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        elif defect_type in ['missing_component', 'damaged']:
            # For structural defects, preserve original shape
            pass
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Dynamic minimum area based on defect type
            if defect_type == 'missing_component':
                min_area = MIN_BBOX_AREA * 2  # Larger minimum for missing components
            elif defect_type in ['scratch']:
                min_area = MIN_BBOX_AREA * 0.5  # Smaller minimum for scratches
            else:
                min_area = MIN_BBOX_AREA
            
            if area >= min_area:
                # Standard bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced bounding box with additional metrics
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Calculate orientation and shape metrics
                if len(contour) >= 5:  # Need at least 5 points for fitEllipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        orientation = ellipse[2]  # Angle
                        aspect_ratio = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) > 0 else 1
                    except:
                        orientation = 0
                        aspect_ratio = w / h if h > 0 else 1
                else:
                    orientation = 0
                    aspect_ratio = w / h if h > 0 else 1
                
                # Compactness (how circular/compact the shape is)
                perimeter = cv2.arcLength(contour, True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                bbox_info = {
                    'id': i + 1,
                    'x': int(x), 
                    'y': int(y), 
                    'width': int(w), 
                    'height': int(h),
                    'area': int(area),
                    'center_x': int(cx), 
                    'center_y': int(cy),
                    'centroid': (int(cx), int(cy)),
                    
                    # Enhanced metrics
                    'orientation': float(orientation),
                    'aspect_ratio': float(aspect_ratio),
                    'compactness': float(compactness),
                    'perimeter': float(perimeter),
                    
                    # Location analysis
                    'relative_position': {
                        'x_percent': (cx / mask.shape[1]) * 100,  # Horizontal position %
                        'y_percent': (cy / mask.shape[0]) * 100,  # Vertical position %
                        'quadrant': get_quadrant(cx, cy, mask.shape[1], mask.shape[0])
                    },
                    
                    # Shape classification
                    'shape_type': classify_defect_shape(w, h, aspect_ratio, compactness, defect_type),
                    'severity': calculate_defect_severity(area, defect_type, mask.shape[0] * mask.shape[1])
                }
                
                bounding_boxes.append(bbox_info)
        
        # Sort bounding boxes by area (largest first)
        bounding_boxes.sort(key=lambda x: x['area'], reverse=True)
        
        return bounding_boxes
        
    except Exception as e:
        print(f"Error extracting enhanced bounding boxes: {e}")
        return []

def get_quadrant(x, y, width, height):
    """Determine which quadrant of the image the defect is in"""
    mid_x, mid_y = width // 2, height // 2
    
    if x < mid_x and y < mid_y:
        return "Top-Left"
    elif x >= mid_x and y < mid_y:
        return "Top-Right"
    elif x < mid_x and y >= mid_y:
        return "Bottom-Left"
    else:
        return "Bottom-Right"

def classify_defect_shape(width, height, aspect_ratio, compactness, defect_type):
    """Classify the shape characteristics of the defect"""
    if defect_type == 'scratch':
        if aspect_ratio > 3:
            return "Linear/Elongated"
        elif aspect_ratio > 1.5:
            return "Streak-like"
        else:
            return "Spot-like"
    
    elif defect_type == 'missing_component':
        if compactness > 0.7:
            return "Circular/Round"
        elif aspect_ratio > 2:
            return "Rectangular/Elongated"
        else:
            return "Irregular"
    
    elif defect_type == 'stained':
        if compactness > 0.6:
            return "Blob-like"
        else:
            return "Irregular Stain"
    
    elif defect_type == 'damaged':
        if compactness < 0.3:
            return "Crack/Break"
        else:
            return "Impact Damage"
    
    else:
        # General classification
        if compactness > 0.7:
            return "Compact/Circular"
        elif aspect_ratio > 2:
            return "Elongated"
        else:
            return "Irregular"

def calculate_defect_severity(area, defect_type, total_area):
    """Calculate defect severity based on size and type"""
    area_percentage = (area / total_area) * 100
    
    # Base severity on area percentage
    if area_percentage < 0.1:
        severity = "Minor"
    elif area_percentage < 0.5:
        severity = "Moderate"
    elif area_percentage < 2.0:
        severity = "Significant"
    else:
        severity = "Critical"
    
    # Adjust based on defect type
    if defect_type in ['missing_component', 'damaged']:
        # These are always more serious
        if severity == "Minor":
            severity = "Moderate"
        elif severity == "Moderate":
            severity = "Significant"
    
    return severity

def analyze_spatial_distribution(mask, image_shape, bboxes):
    """Analyze spatial distribution of defects"""
    if not bboxes:
        return {}
    
    h, w = mask.shape
    
    # Calculate center of mass for all defects
    y_coords, x_coords = np.where(mask)
    if len(x_coords) > 0:
        center_of_mass_x = np.mean(x_coords)
        center_of_mass_y = np.mean(y_coords)
    else:
        center_of_mass_x = w // 2
        center_of_mass_y = h // 2
    
    # Analyze distribution patterns
    spatial_info = {
        'center_of_mass': {
            'x': float(center_of_mass_x),
            'y': float(center_of_mass_y),
            'x_percent': (center_of_mass_x / w) * 100,
            'y_percent': (center_of_mass_y / h) * 100
        },
        'distribution_pattern': analyze_distribution_pattern(bboxes, w, h),
        'edge_proximity': analyze_edge_proximity(bboxes, w, h),
        'clustering': analyze_defect_clustering(bboxes)
    }
    
    return spatial_info

def analyze_distribution_pattern(bboxes, width, height):
    """Analyze how defects are distributed across the image"""
    if len(bboxes) < 2:
        return "Single defect"
    
    # Calculate spread
    x_coords = [bbox['center_x'] for bbox in bboxes]
    y_coords = [bbox['center_y'] for bbox in bboxes]
    
    x_spread = (max(x_coords) - min(x_coords)) / width
    y_spread = (max(y_coords) - min(y_coords)) / height
    
    if x_spread < 0.3 and y_spread < 0.3:
        return "Clustered"
    elif x_spread > 0.7 or y_spread > 0.7:
        return "Scattered"
    else:
        return "Distributed"

def analyze_edge_proximity(bboxes, width, height):
    """Analyze proximity to image edges"""
    edge_distance_threshold = 0.1  # 10% from edge
    
    edge_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    for bbox in bboxes:
        x, y = bbox['center_x'], bbox['center_y']
        
        if y < height * edge_distance_threshold:
            edge_counts['top'] += 1
        if y > height * (1 - edge_distance_threshold):
            edge_counts['bottom'] += 1
        if x < width * edge_distance_threshold:
            edge_counts['left'] += 1
        if x > width * (1 - edge_distance_threshold):
            edge_counts['right'] += 1
    
    total_edge_defects = sum(edge_counts.values())
    edge_proximity_info = {
        'edge_counts': edge_counts,
        'total_near_edges': total_edge_defects,
        'edge_percentage': (total_edge_defects / len(bboxes)) * 100 if bboxes else 0
    }
    
    return edge_proximity_info

def analyze_defect_clustering(bboxes):
    """Analyze clustering of defects"""
    if len(bboxes) < 2:
        return {"cluster_count": 1, "clusters": []}
    
    # Simple distance-based clustering
    cluster_distance_threshold = 100  # pixels
    
    clusters = []
    unassigned = list(range(len(bboxes)))
    
    while unassigned:
        # Start new cluster with first unassigned defect
        cluster_seed = unassigned.pop(0)
        current_cluster = [cluster_seed]
        
        # Find nearby defects
        i = 0
        while i < len(unassigned):
            candidate = unassigned[i]
            
            # Check distance to any defect in current cluster
            min_distance = float('inf')
            for cluster_member in current_cluster:
                dx = bboxes[candidate]['center_x'] - bboxes[cluster_member]['center_x']
                dy = bboxes[candidate]['center_y'] - bboxes[cluster_member]['center_y']
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = min(min_distance, distance)
            
            if min_distance < cluster_distance_threshold:
                current_cluster.append(unassigned.pop(i))
            else:
                i += 1
        
        clusters.append(current_cluster)
    
    clustering_info = {
        'cluster_count': len(clusters),
        'clusters': [
            {
                'defect_indices': cluster,
                'size': len(cluster),
                'center': {
                    'x': np.mean([bboxes[i]['center_x'] for i in cluster]),
                    'y': np.mean([bboxes[i]['center_y'] for i in cluster])
                }
            }
            for cluster in clusters
        ]
    }
    
    return clustering_info
