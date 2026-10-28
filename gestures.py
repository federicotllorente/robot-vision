
"""Hand gesture playground.

Keys:  1 landmarks   2 mouse   3 windows   4 HUD   Q quit
Mouse mode: hold a fist 1 s (or press SPACE) to turn control on/off.
"""
import time

import cv2

import desktop_control as desktop
import ui_theme as ui
from aspect_window import AspectWindow
from gesture_helpers import HoldDetector, PinchDetector, Smoother, classify_pose
from hand_tracking import HandTracker, distance, draw_hand, midpoint
from hud import ChartCard, Hud, NoteCard

WINDOW_NAME = "Robot Vision - Gestures"

MODES = {
    ord("1"): "LANDMARKS",
    ord("2"): "MOUSE",
    ord("3"): "WINDOWS",
    ord("4"): "HUD",
}
# Header tab for each mode: "1 LANDMARKS", "2 MOUSE"...
TAB_OF_MODE = {name: f"{chr(key)} {name}" for key, name in MODES.items()}

# Mouse: only this central part of the image maps to the whole screen,
# so you don't need to reach the edges of the camera view
ACTIVE_AREA = (0.2, 0.2, 0.8, 0.8)  # left, top, right, bottom (0-1)

# Windows: minimum time between two resizes (moving windows every frame is heavy)
RESIZE_INTERVAL = 0.05

# First line of text under the header bar
INFO_Y = ui.u(54)

def hand_labels(hands):
    """Unique label per hand. MediaPipe sometimes calls both hands "Right"."""
    labels = []
    for hand in hands:
        label = hand.label
        while label in labels:
            label += "+"
        labels.append(label)
    return labels

def update_pinches(hands, detectors):
    """dict label -> (event, pinching, point) and one PinchDetector per hand."""
    pinches = {}
    labels = hand_labels(hands)

    for label, hand in zip(labels, hands):
        detector = detectors.setdefault(label, PinchDetector())
        event = detector.update(hand.pinch_ratio())
        pinches[label] = (event, detector.pinching, hand.pinch_point())

    # Forget hands that left the image
    for label in list(detectors):
        if label not in pinches:
            del detectors[label]

    return pinches

def draw_hold_bar(canvas, hold, label):
    progress = hold.progress()
    if progress > 0:
        ui.progress_bar(canvas, (ui.u(10), canvas.shape[0] - ui.u(48)), 120, progress, label)

# ---------------------------------------------------------------- mode 1
def run_landmarks(canvas, hands):
    if not hands:
        ui.status_line(canvas, "Show a hand to the camera", ui.AMBER)

    for i, hand in enumerate(hands):
        draw_hand(canvas, hand)
        up = hand.fingers_up()
        pose = classify_pose(hand) or "-"

        # Info tag next to the wrist
        x, y = hand.points[0]
        ui.panel(canvas, (x - ui.u(60), y + ui.u(10)), (x + ui.u(90), y + ui.u(52)), alpha=0.7)
        ui.text(canvas, f"{hand.label.upper()} · {sum(up.values())} FINGERS", (x - ui.u(52), y + ui.u(27)),
                size=8, color=ui.CYAN, font="title", weight=700)
        ui.text(canvas, f"pose: {pose}   pinch: {hand.pinch_ratio():.2f}", (x - ui.u(52), y + ui.u(44)), size=11)

        # Which fingers are up, as a list
        names = "  ".join(name[0].upper() if is_up else "·" for name, is_up in up.items())
        ui.text(canvas, f"{hand.label}: {names}", (ui.u(12), INFO_Y + ui.u(18) * i), size=12)

# ---------------------------------------------------------------- mode 2
class MouseMode:
    def __init__(self):
        self.enabled = False
        self.smoother = Smoother(alpha=0.35)
        self.pinch = PinchDetector()
        self.toggle = HoldDetector(seconds=1.0)
        self.button_down = False

    def stop(self):
        if self.button_down:
            desktop.mouse_up()
            self.button_down = False
        self.smoother.reset()
        self.pinch.reset()

    def run(self, canvas, hands):
        height, width = canvas.shape[:2]
        left, top, right, bottom = ACTIVE_AREA
        color = ui.TEAL if self.enabled else ui.ERROR
        area = ((int(left * width), int(top * height)), (int(right * width), int(bottom * height)))
        cv2.rectangle(canvas, area[0], area[1], ui.MUTED, ui.u(0.5), cv2.LINE_AA)
        ui.corner_brackets(canvas, area[0], area[1], color, length=18, thickness=3)

        if self.enabled:
            ui.status_line(canvas, "MOUSE ON · pinch = click / drag", ui.TEAL)
        else:
            ui.status_line(canvas, "MOUSE OFF · hold a fist or press SPACE", ui.ERROR)

        if not hands:
            # Hand lost: never leave the button pressed
            self.stop()
            self.toggle.update(None)
            return

        hand = hands[0]
        draw_hand(canvas, hand, ui.CYAN if self.enabled else ui.MUTED)

        event = self.pinch.update(hand.pinch_ratio())

        # A pinch folds the index, which looks like a fist: ignore poses while pinching
        pose = None if self.pinch.pinching else classify_pose(hand)
        if self.toggle.update(pose if pose == "fist" else None) == "fist":
            self.set_enabled(not self.enabled)
        draw_hold_bar(canvas, self.toggle, "toggle mouse")

        if not self.enabled:
            return

        # Pointer = middle point between thumb and index tips. When you pinch,
        # both tips move towards that point, so the cursor stays still while
        # clicking (the index tip alone would jump)
        point = self.smoother.update(hand.pinch_point())
        cv2.circle(canvas, point, ui.u(9), ui.CYAN, ui.u(1), cv2.LINE_AA)
        cv2.circle(canvas, point, ui.u(2.5), ui.CYAN, -1, cv2.LINE_AA)

        screen_w, screen_h = desktop.screen_size()
        nx = (point[0] / width - left) / (right - left)
        ny = (point[1] / height - top) / (bottom - top)
        nx, ny = min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)
        desktop.move_mouse(nx * (screen_w - 1), ny * (screen_h - 1))

        # Pinch start/end = button down/up, so pinch-and-move drags
        if event == "start":
            desktop.mouse_down()
            self.button_down = True
        elif event == "end" and self.button_down:
            desktop.mouse_up()
            self.button_down = False

        if self.button_down:
            ui.text(canvas, "CLICK", (point[0] + ui.u(12), point[1]), size=9, color=ui.CYAN, font="title", weight=700)

    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.stop()

# ---------------------------------------------------------------- mode 3
class WindowsMode:
    def __init__(self):
        self.target = None
        self.hold = HoldDetector(seconds=0.8)
        self.pinch_detectors = {}
        self.resize_start = None  # (distance, window rect)
        self.last_resize = 0.0

    def remember_target(self):
        # The window to control is the last one you used that is not this one
        hwnd = desktop.foreground_window()
        if hwnd and desktop.window_title(hwnd) != WINDOW_NAME:
            self.target = hwnd

    def run(self, canvas, hands):
        self.remember_target()
        title = desktop.window_title(self.target) if self.target else "(click a window first)"
        ui.text(canvas, "TARGET", (ui.u(12), INFO_Y), size=8, color=ui.MUTED, font="title", weight=700)
        ui.text(canvas, title[:60], (ui.u(62), INFO_Y), size=12, color=ui.CYAN)
        ui.status_line(canvas, "fist: minimize · open palm: maximize/restore · pinch both hands: resize", ui.TEXT)

        for hand in hands:
            draw_hand(canvas, hand)

        pinches = update_pinches(hands, self.pinch_detectors)
        pinching = [p for _event, is_pinching, p in pinches.values() if is_pinching]

        if self.target and len(pinching) == 2:
            self.resize(canvas, pinching)
            self.hold.update(None)
            return

        self.resize_start = None

        # Single static pose held for a moment
        pose = classify_pose(hands[0]) if len(hands) == 1 and not pinching else None
        gesture = pose if pose in ("fist", "open_palm") else None
        done = self.hold.update(gesture)
        draw_hold_bar(canvas, self.hold, (gesture or "").replace("_", " "))

        if self.target and done == "fist":
            desktop.minimize(self.target)
        elif self.target and done == "open_palm":
            desktop.toggle_maximize(self.target)

    def resize(self, canvas, points):
        a, b = points
        ui.glow_line(canvas, a, b, ui.CYAN, thickness=2)
        current = max(distance(a, b), 1.0)

        if self.resize_start is None:
            self.resize_start = (current, desktop.window_rect(self.target))
            return

        start_distance, start_rect = self.resize_start
        scale = current / start_distance
        center = midpoint(a, b)
        ui.text(canvas, f"x{scale:.2f}", (center[0] - ui.u(16), center[1] - ui.u(10)),
                size=11, color=ui.CYAN, font="title", weight=700)

        now = time.monotonic()
        if now - self.last_resize >= RESIZE_INTERVAL:
            desktop.resize_around_center(self.target, start_rect, scale)
            self.last_resize = now

# ---------------------------------------------------------------- mode 4
class HudMode:
    def __init__(self):
        # Positions and sizes in design units (a 640 px wide image)
        self.chart = ChartCard("Fingers up", 400, 320)
        self.hud = Hud([
            NoteCard("Hello!", ["Pinch a card to grab it.", "Pinch with your other", "hand to scale it."], 30, 80),
            NoteCard("Robot arm", ["ESP32 + servos", "Next: calibrate camera"], 400, 80, accent=ui.GLOW),
            self.chart,
        ])
        self.pinch_detectors = {}

    def run(self, canvas, hands):
        for hand in hands:
            draw_hand(canvas, hand)

        self.chart.push(hands[0].count_fingers() if hands else 0)

        pinches = update_pinches(hands, self.pinch_detectors)
        self.hud.update(pinches)
        self.hud.draw(canvas)

        for _event, pinching, point in pinches.values():
            cv2.circle(canvas, point, ui.u(7), ui.CYAN if pinching else ui.MUTED, ui.u(1), cv2.LINE_AA)

# ---------------------------------------------------------------- main loop
def main():
    tracker = HandTracker(max_hands=2)
    mouse_mode = MouseMode()
    windows_mode = WindowsMode()
    hud_mode = HudMode()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    window = None
    mode = "LANDMARKS"
    last_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not capture a frame.")
                break

            # Mirror the image: moving your hand right moves it right on screen.
            # MediaPipe's Left/Right labels also assume a mirrored image.
            frame = cv2.flip(frame, 1)

            # Everything (tracking and drawing) happens on the canvas, so all
            # positions are canvas pixels
            canvas = ui.prepare_canvas(frame)
            hands = tracker.detect(canvas)

            if mode == "LANDMARKS":
                run_landmarks(canvas, hands)
            elif mode == "MOUSE":
                mouse_mode.run(canvas, hands)
            elif mode == "WINDOWS":
                windows_mode.run(canvas, hands)
            elif mode == "HUD":
                hud_mode.run(canvas, hands)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
            last_time = now
            ui.header(
                canvas, "GESTURES", TAB_OF_MODE.values(),
                active=TAB_OF_MODE[mode], right_text=f"{fps:4.1f} FPS",
            )

            if window is None:
                window = AspectWindow(WINDOW_NAME, canvas.shape[1], canvas.shape[0])
            window.show(canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not window.is_open():
                break
            if key in MODES and MODES[key] != mode:
                mouse_mode.set_enabled(False)
                mode = MODES[key]
            if key == ord(" ") and mode == "MOUSE":
                mouse_mode.set_enabled(not mouse_mode.enabled)
    finally:
        # Whatever happens, never leave the mouse button pressed
        mouse_mode.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
