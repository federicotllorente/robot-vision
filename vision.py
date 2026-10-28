import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

# False: COCO model (80 fixed classes, no "pencil")
# True: YOLO-World (open vocabulary, searches the names in TARGET_CLASSES)
USE_YOLO_WORLD = True

# Objects YOLO-World looks for at startup (English works best).
# The assistant can change them later through MQTT.
TARGET_CLASSES = ["pencil", "pen", "cup", "bottle", "scissors", "person"]

# Minimum confidence to accept a detection.
# YOLO-World scores are usually lower than COCO ones.
CONFIDENCE_THRESHOLD = 0.15 if USE_YOLO_WORLD else 0.5

# HSV ranges for each color. Red needs two ranges because
# its hue sits at both ends of the 0-179 scale.
COLOR_RANGES = {
    "red": [
        ([0, 100, 80], [10, 255, 255]),
        ([170, 100, 80], [179, 255, 255]),
    ],
    "orange": [([11, 100, 80], [22, 255, 255])],
    "yellow": [([23, 100, 80], [34, 255, 255])],
    "green": [([35, 100, 80], [85, 255, 255])],
    "blue": [([86, 100, 80], [130, 255, 255])],
    "purple": [([131, 100, 80], [169, 255, 255])],
}

# Minimum share of the box a color must cover to be accepted
MIN_COLOR_RATIO = 0.15

def load_model():
    if USE_YOLO_WORLD:
        model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(TARGET_CLASSES)
    else:
        model = YOLO("yolo11n.pt")

    return model

def get_targets(model):
    # Class names the model can currently detect
    return list(model.names.values())

def set_targets(model, targets):
    # Only YOLO-World can change its classes at runtime
    if not USE_YOLO_WORLD:
        return False

    # Clean the names and drop empty ones and duplicates
    cleaned = []

    for name in targets:
        name = str(name).strip().lower()

        if name and name not in cleaned:
            cleaned.append(name)

    if not cleaned:
        return False

    model.set_classes(cleaned)
    return True

def get_grid_cell(center_x, center_y, width, height):
    # Determine the horizontal position
    if center_x < width // 3:
        horizontal = "LEFT"
    elif center_x < (2 * width) // 3:
        horizontal = "CENTER"
    else:
        horizontal = "RIGHT"

    # Determine the vertical position
    if center_y < height // 3:
        vertical = "TOP"
    elif center_y < (2 * height) // 3:
        vertical = "MIDDLE"
    else:
        vertical = "BOTTOM"

    return f"{vertical}-{horizontal}"

def get_dominant_color(hsv_region):
    total_pixels = hsv_region.shape[0] * hsv_region.shape[1]

    if total_pixels == 0:
        return "unknown"

    best_color = "unknown"
    best_ratio = 0.0

    for color, ranges in COLOR_RANGES.items():
        # Combine all the ranges of this color into one mask
        mask = np.zeros(hsv_region.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            range_mask = cv2.inRange(
                hsv_region, np.array(lower), np.array(upper)
            )
            mask = cv2.bitwise_or(mask, range_mask)

        ratio = cv2.countNonZero(mask) / total_pixels

        if ratio > best_ratio:
            best_color = color
            best_ratio = ratio

    # Ignore colors that only cover a few pixels of the box
    if best_ratio < MIN_COLOR_RATIO:
        return "unknown"

    return best_color

def detect_objects(model, frame):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Run the model on the frame (one image -> one result)
    result = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    detections = []

    for box in result.boxes:
        # Box corners in pixels
        left, top, right, bottom = box.xyxy[0].int().tolist()

        class_id = int(box.cls[0])
        name = result.names[class_id]
        confidence = float(box.conf[0])

        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Check the color only inside the detected box
        color = get_dominant_color(hsv[top:bottom, left:right])

        detections.append({
            "name": name,
            "color": color,
            "confidence": confidence,
            "box": (left, top, right, bottom),
            "center": (center_x, center_y),
            "cell": get_grid_cell(center_x, center_y, width, height),
        })

    return detections

def draw_grid(frame):
    height, width = frame.shape[:2]

    x1 = width // 3
    x2 = (2 * width) // 3
    y1 = height // 3
    y2 = (2 * height) // 3

    # Draw the 3x3 grid
    cv2.line(frame, (x1, 0), (x1, height), (255, 255, 0), 2)
    cv2.line(frame, (x2, 0), (x2, height), (255, 255, 0), 2)
    cv2.line(frame, (0, y1), (width, y1), (255, 255, 0), 2)
    cv2.line(frame, (0, y2), (width, y2), (255, 255, 0), 2)

    # Label the grid columns
    cv2.putText(
        frame, "LEFT", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "CENTER", (x1 + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "RIGHT", (x2 + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )

def draw_detection(frame, detection):
    left, top, right, bottom = detection["box"]

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.circle(frame, detection["center"], 5, (255, 0, 0), -1)

    label = (
        f"{detection['color']} {detection['name']} "
        f"{detection['confidence']:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (left, max(top - 10, 45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
