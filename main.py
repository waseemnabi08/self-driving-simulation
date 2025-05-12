import cv2
import numpy as np
import time
from collections import deque
import imutils

# Configuration
CONFIG = {
    # Edge detection
    'canny_low': 40,
    'canny_high': 120,

    # Hough transform
    'hough_params': {
        'rho': 2,
        'theta': np.pi/180,
        'threshold': 15,  # Reduced for more sensitivity
        'minLineLength': 20,  # Slightly reduced
        'maxLineGap': 15  # Slightly reduced
    },

    # ROI settings
    'roi_height_ratio': 0.55,
    'roi_width_margin': 0.05,

    # Object detection
    'min_object_area': 100,  # Increased for large objects only
    'min_object_width': 40,   # Minimum width to consider
    'min_object_height': 40,  # Minimum height to consider
    'danger_zone_threshold': 0.3, 
    'object_confirmation_frames': 3,

    # General
    'scale_factor': 0.75,
    'debug_level': 2, # 0: No debug, 1: Basic debug, 2: Advanced debug
    'roc_highlight_color': (0, 255, 255)
}

# State variables
state = {
    'left_lane_history': deque(maxlen=5),
    'right_lane_history': deque(maxlen=5),
    'last_roi_vertices': None,
    'consecutive_object_frames': 0,
    'last_decision': "STOP",
    'decision_smoothing': deque(maxlen=4),
    'frame_count': 0,
    'fps': 0,
    'last_time': time.time(),
    'process_time': 0
}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(orig_width * CONFIG['scale_factor'])
    height = int(orig_height * CONFIG['scale_factor'])

    print(f"Processing video {video_path} at {width}x{height} resolution")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        if CONFIG['scale_factor'] != 1.0:
            frame = cv2.resize(frame, (width, height))

        # Process timing
        start_time = time.time()

        # Process frame
        processed = process_frame(frame)

        # Update timing
        state['process_time'] = time.time() - start_time
        state['frame_count'] += 1

        if state['frame_count'] % 10 == 0:
            update_fps()

        # Display results
        display_results(frame, processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_fps():
    current_time = time.time()
    elapsed = current_time - state['last_time']
    state['fps'] = 10 / elapsed if elapsed > 0 else 0
    state['last_time'] = current_time

def process_frame(frame):
    # Step 1: Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Step 2: Edge detection
    edges = adaptive_canny(blur)

    # Step 3: ROI processing
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = get_dynamic_roi(frame, height, width)
    cv2.fillPoly(mask, [roi_vertices], 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # Create ROC highlight overlay
    roc_overlay = np.zeros_like(frame)
    cv2.fillPoly(roc_overlay, [roi_vertices], CONFIG['roc_highlight_color'])
    frame_with_roc = cv2.addWeighted(frame, 0.7, roc_overlay, 0.3, 0)

    # Step 4: Line detection
    lines = cv2.HoughLinesP(roi_edges, **CONFIG['hough_params'])

    # Step 5: Lane detection
    line_image = np.zeros_like(frame)
    left_lane, right_lane = detect_lanes(lines, frame.shape)

    # Step 6: Object detection (enhanced for large objects)
    objects, danger_objects = detect_objects(frame, roi_edges, height, width)

    # Update object detection counter
    if danger_objects:
        state['consecutive_object_frames'] = min(state['consecutive_object_frames'] + 1,
                                                 CONFIG['object_confirmation_frames'])
    else:
        state['consecutive_object_frames'] = max(0, state['consecutive_object_frames'] - 1)

    # Step 7: Make driving decision
    decision = make_decision(left_lane, right_lane, danger_objects, frame.shape[1])
    state['last_decision'] = decision


    return {
        'line_image': line_image,
        'decision': decision,
        'left_lane': left_lane,
        'right_lane': right_lane,
        'objects': objects,
        'danger_objects': danger_objects,
        'frame_with_roc': frame_with_roc,
        'debug': {
            'gray': gray,
            'enhanced': enhanced,
            'blur': blur,
            'edges': edges,
            'roi': roi_edges,
            'lines': lines  # Pass the detected lines for debugging
        }
    }


def adaptive_canny(image):
    v = np.median(image)
    sigma = 0.33
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, low, high)

def get_dynamic_roi(frame, height, width):
    top_width = 0.5
    margin = CONFIG['roi_width_margin']

    vertices = np.array([
        (width * margin, height),
        (width * (0.5 - top_width / 2), height * CONFIG['roi_height_ratio']),
        (width * (0.5 + top_width / 2), height * CONFIG['roi_height_ratio']),
        (width * (1 - margin), height)
    ], dtype=np.int32)

    state['last_roi_vertices'] = vertices
    return vertices


def detect_lanes(lines, frame_shape):
    line_image = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    left_lane = None
    right_lane = None

    if lines is not None:
        left_lines, right_lines = separate_lines(lines, frame_shape)

        # Smooth lanes with history
        left_lane = smooth_lanes(left_lines, state['left_lane_history'])
        right_lane = smooth_lanes(right_lines, state['right_lane_history'])

        # Draw lanes if detected
        if left_lane is not None:
            draw_lane(line_image, left_lane, (0, 0, 255), 5)
            state['left_lane_history'].append(left_lane)

        if right_lane is not None:
            draw_lane(line_image, right_lane, (255, 0, 0), 5)
            state['right_lane_history'].append(right_lane)

    return left_lane, right_lane


def separate_lines(lines, frame_shape):
    left_lines = []
    right_lines = []
    height, width = frame_shape[:2]
    mid_x = width // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip nearly vertical or horizontal lines
        if abs(x2 - x1) < 15 or abs(y2 - y1) < 15:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter lines based on slope
        if abs(slope) < 0.3 or abs(slope) > 3.0:
            continue

        mid_point_x = (x1 + x2) / 2

        # Classify lines as left or right based on position relative to center
        if slope < 0 and mid_point_x < mid_x:
            left_lines.append((slope, intercept))
        elif slope > 0 and mid_point_x > mid_x:
            right_lines.append((slope, intercept))

    return left_lines, right_lines


def smooth_lanes(lines, history):
    if not lines:
        return np.mean(history, axis=0) if history else None

    # Filter out outliers based on slope consistency
    if len(lines) >= 3:
        slopes = [line[0] for line in lines]
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        filtered_lines = [line for line in lines if abs(line[0] - mean_slope) < 2 * std_slope]
        lines = filtered_lines if filtered_lines else lines


    current_avg = np.mean(lines, axis=0)

    # Blend with history for stability
    if history:
        history_avg = np.mean(history, axis=0)
        return 0.7 * current_avg + 0.3 * history_avg

    return current_avg

def draw_lane(image, lane_params, color, thickness):
    if lane_params is None:
        return

    slope, intercept = lane_params
    height, width = image.shape[:2]

    y1 = height
    y2 = int(height * CONFIG['roi_height_ratio'])

    try:
        x1 = int((y1 - intercept) / slope) if slope != 0 else 0
        x2 = int((y2 - intercept) / slope) if slope != 0 else 0

        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        y3 = int(height * 0.4)
        x3 = int((y3 - intercept) / slope) if slope != 0 else x2
        dashed_line = np.array([[x2, y2], [x3, y3]], dtype=np.int32)
        cv2.polylines(image, [dashed_line], False, color, thickness // 2, lineType=cv2.LINE_AA)
    except Exception as e:
        print(f"Error in draw_lane: {e}")
        pass


def detect_objects(frame, roi_edges, frame_height, frame_width):
    """
    Enhanced object detection function with multiple detection strategies
    
    Args:
        frame (numpy.ndarray): Original color frame
        roi_edges (numpy.ndarray): Edges within Region of Interest
        frame_height (int): Height of the frame
        frame_width (int): Width of the frame
    
    Returns:
        tuple: (all_objects, danger_objects)
    """
    # Multiple detection strategies
    objects = []
    danger_objects = []
    
    # Danger zone calculation
    danger_zone_y = int(frame_height * (1 - CONFIG['danger_zone_threshold']))
    
    # Strategy 1: Contour-based detection
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Calculate bounding box and area
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filtering criteria
        if (area < CONFIG['min_object_area'] or 
            w < CONFIG['min_object_width'] or 
            h < CONFIG['min_object_height']):
            continue
        
        # Aspect ratio and solidity checks
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        if solidity < 0.5:
            continue
        
        # Object classification and danger zone check
        objects.append((x, y, w, h))
        
        if y + h > danger_zone_y:
            center_x = x + w // 2
            frame_center_x = frame_width // 2
            distance_to_center = abs(center_x - frame_center_x)
            danger_objects.append((x, y, w, h, distance_to_center))
    
    # Strategy 2: Color-based object detection (optional enhancement)
    # Convert frame to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for different object types (adjust as needed)
    color_ranges = [
        # Bright colors might indicate objects
        ('yellow', np.array([20, 100, 100]), np.array([30, 255, 255])),
        ('red1', np.array([0, 100, 100]), np.array([10, 255, 255])),
        ('red2', np.array([160, 100, 100]), np.array([180, 255, 255])),
        ('blue', np.array([100, 100, 100]), np.array([140, 255, 255]))
    ]
    
    for color_name, lower, upper in color_ranges:
        # Create color mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours in color mask
        color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in color_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Similar filtering as contour-based detection
            if (area < CONFIG['min_object_area'] or 
                w < CONFIG['min_object_width'] or 
                h < CONFIG['min_object_height']):
                continue
            
            # Check if this object is not already detected
            if not any(abs(x - obj[0]) < w and abs(y - obj[1]) < h for obj in objects):
                objects.append((x, y, w, h))
                
                if y + h > danger_zone_y:
                    center_x = x + w // 2
                    frame_center_x = frame_width // 2
                    distance_to_center = abs(center_x - frame_center_x)
                    danger_objects.append((x, y, w, h, distance_to_center))
    
    # Sort danger objects by distance to center
    danger_objects.sort(key=lambda obj: obj[4])
    
    return objects, [obj[:4] for obj in danger_objects]

def make_decision(left_lane, right_lane, danger_objects, frame_width):
    """Decision logic for lane following and obstacle avoidance."""

    # Priority 1: Stop if any danger objects are detected in the ROI
    if danger_objects:
        return "STOP - Obstacle detected in ROI"

    # Both lanes visible → move forward
    if left_lane is not None and right_lane is not None:
        return "FORWARD"

    # Only left lane visible → steer right
    if left_lane is not None:
        return "RIGHT"

    # Only right lane visible → steer left
    if right_lane is not None:
        return "LEFT"

    # No lanes visible → stop
    return "STOP - No lanes detected"


def display_results(frame, processed):
    # Combine the original frame with the ROI highlight and lane lines
    combined = cv2.addWeighted(processed['frame_with_roc'], 0.8, processed['line_image'], 1, 0)

    # Draw danger zone line
    height, width = frame.shape[:2]
    danger_zone_y = int(height * (1 - CONFIG['danger_zone_threshold']))
    cv2.line(combined, (0, danger_zone_y), (width, danger_zone_y), (0, 0, 255), 2)

    # Draw center line
    center_x = width // 2
    cv2.line(combined, (center_x, height), (center_x, int(height*0.6)), (0, 255, 255), 1, cv2.LINE_AA)

    # Draw bounding boxes around ALL detected objects (yellow)
    for obj in processed['objects']:
        x, y, w, h = obj
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Highlight danger objects with a red bounding box
    for obj in processed['danger_objects']:
        x, y, w, h = obj
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Add decision text
    cv2.putText(combined, processed['decision'], (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add performance info
    cv2.putText(combined, f"FPS: {state['fps']:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, f"Process: {state['process_time']*1000:.1f}ms", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add object detection status
    obj_status = f"Objects: {len(processed['danger_objects'])} ({state['consecutive_object_frames']}/{CONFIG['object_confirmation_frames']})"
    cv2.putText(combined, obj_status, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if processed['danger_objects'] else (0, 255, 0), 2)

    # Add lane center indicator
    if processed['left_lane'] is not None or processed['right_lane'] is not None:
        center_x = width // 2
        lane_center = center_x

        if processed['left_lane'] is not None and processed['right_lane'] is not None:
            y = height
            try:
                left_x = int((y - processed['left_lane'][1]) / processed['left_lane'][0]) if processed['left_lane'][0] != 0 else 0
                right_x = int((y - processed['right_lane'][1]) / processed['right_lane'][0]) if processed['right_lane'][0] != 0 else width
                lane_center = (left_x + right_x) // 2
            except:
                pass

        deviation = lane_center - center_x
        cv2.circle(combined, (lane_center, height - 50), 10, (0, 255, 255), -1)
        cv2.circle(combined, (center_x, height - 50), 5, (255, 255, 255), -1)
        cv2.line(combined, (center_x, height - 50), (lane_center, height - 50), (255, 255, 255), 2)

        cv2.putText(combined, f"Deviation: {deviation}px", (center_x - 80, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Overlay detected lines on the main result
    combined = cv2.addWeighted(combined, 1, processed['line_image'], 1, 0)

    cv2.imshow('Advanced Lane Detection', combined)

    # Display debug views based on debug_level
    if CONFIG['debug_level'] > 1:
        display_debug_views(processed['debug'])


def display_debug_views(debug_imgs):
    # Convert grayscale debug images to BGR for stacking
    gray_bgr = cv2.cvtColor(debug_imgs['gray'], cv2.COLOR_GRAY2BGR)
    enhanced_bgr = cv2.cvtColor(debug_imgs['enhanced'], cv2.COLOR_GRAY2BGR)
    blur_bgr = cv2.cvtColor(debug_imgs['blur'], cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(debug_imgs['edges'], cv2.COLOR_GRAY2BGR)
    roi_bgr = cv2.cvtColor(debug_imgs['roi'], cv2.COLOR_GRAY2BGR)

    # Create an image to draw detected lines on for debugging
    lines_bgr = np.zeros_like(gray_bgr)
    if debug_imgs['lines'] is not None:
        lines = debug_imgs['lines']
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw lines in green

    # Helper function to add titles to debug images
    def add_title(img, title):
        img_copy = img.copy()
        cv2.putText(img_copy, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img_copy

    # Add titles to each debug image
    gray_bgr = add_title(gray_bgr, "1. Grayscale")
    enhanced_bgr = add_title(enhanced_bgr, "2. CLAHE Enhanced")
    blur_bgr = add_title(blur_bgr, "3. Bilateral Filter")
    edges_bgr = add_title(edges_bgr, "4. Adaptive Canny")
    roi_bgr = add_title(roi_bgr, "5. ROI Masked")
    lines_bgr = add_title(lines_bgr, "6. Detected Lines")

    # Stack debug images into a grid
    row1 = np.hstack((gray_bgr, enhanced_bgr, blur_bgr))
    row2 = np.hstack((edges_bgr, roi_bgr, lines_bgr))
    debug_grid = np.vstack((row1, row2))

    # Resize debug grid if it's too large for the screen
    # This part might need adjustment based on your screen resolution
    if debug_grid.shape[1] > 1200:
        scale = 1200 / debug_grid.shape[1]
        debug_grid = cv2.resize(debug_grid, None, fx=scale, fy=scale)

    # Display the debug grid
    cv2.imshow('Processing Steps', debug_grid)
    # cv2.waitKey(1) # Kept in the main loop's waitKey

if __name__ == "__main__":
    print("Self Driving Simulation...")
    process_video("6.mp4")  # video file path