import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import ui_theme as ui

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

# The 21 hand landmarks MediaPipe returns, by index
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (tip, pip) pairs for the four long fingers
FINGERS = {
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}

CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def midpoint(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

class Hand:
    """One detected hand: 21 points in pixels plus helpers."""

    def __init__(self, points, label):
        self.points = points  # list of (x, y) in pixels
        self.label = label  # "Left" or "Right"

    def size(self):
        # Wrist to middle knuckle: grows when the hand is closer to the camera.
        # Dividing distances by it makes thresholds work at any distance.
        return max(distance(self.points[WRIST], self.points[MIDDLE_MCP]), 1.0)

    def fingers_up(self):
        """Dict finger -> True if extended."""
        wrist = self.points[WRIST]
        state = {}

        # A long finger is extended when its tip is farther from the wrist
        # than its middle joint (works with the hand in any rotation)
        for name, (tip, pip) in FINGERS.items():
            state[name] = distance(self.points[tip], wrist) > distance(self.points[pip], wrist)

        # The thumb bends sideways, so compare against the pinky knuckle instead
        pinky_base = self.points[PINKY_MCP]
        state["thumb"] = (
            distance(self.points[THUMB_TIP], pinky_base)
            > distance(self.points[THUMB_IP], pinky_base) * 1.1
        )
        return state

    def count_fingers(self):
        return sum(self.fingers_up().values())

    def pinch_ratio(self):
        # Thumb tip to index tip, relative to hand size
        return distance(self.points[THUMB_TIP], self.points[INDEX_TIP]) / self.size()

    def pinch_point(self):
        return midpoint(self.points[THUMB_TIP], self.points[INDEX_TIP])

class HandTracker:
    """Wraps MediaPipe's HandLandmarker for a webcam stream."""

    def __init__(self, max_hands=2):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            # VIDEO mode reuses the previous frame to track faster and steadier
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start = time.monotonic()
        self.last_timestamp = -1

    def detect(self, frame):
        height, width = frame.shape[:2]

        # MediaPipe wants RGB; OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps in milliseconds
        timestamp = int((time.monotonic() - self.start) * 1000)
        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp

        result = self.landmarker.detect_for_video(image, timestamp)

        hands = []

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # Landmarks come normalized (0-1): convert to pixels
            points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
            hands.append(Hand(points, handedness[0].category_name))

        return hands

    def close(self):
        self.landmarker.close()

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

def draw_hand(canvas, hand, color=ui.CYAN):
    # Bones with a soft halo, like Kimmi's wave
    for start, end in CONNECTIONS:
        ui.glow_line(canvas, hand.points[start], hand.points[end], color, thickness=1)

    for i, point in enumerate(hand.points):
        if i in FINGERTIPS:
            # Fingertips: a ring, they are the points that do things
            cv2.circle(canvas, point, ui.u(4), color, ui.u(1), cv2.LINE_AA)
        cv2.circle(canvas, point, ui.u(1.5), ui.TEXT, -1, cv2.LINE_AA)
