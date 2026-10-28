import time

class Smoother:
    """Exponential moving average for a point.

    Raw landmarks jitter a few pixels every frame. Mixing each new position
    with the previous one removes the tremble, at the cost of a little lag:
    alpha close to 1 = fast but shaky, close to 0 = smooth but slow.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, point):
        if self.value is None:
            self.value = point
        else:
            self.value = (
                self.value[0] + self.alpha * (point[0] - self.value[0]),
                self.value[1] + self.alpha * (point[1] - self.value[1]),
            )
        return (int(self.value[0]), int(self.value[1]))

    def reset(self):
        self.value = None

class PinchDetector:
    """Pinch on/off with hysteresis.

    With a single threshold, a pinch right at the limit flickers on/off and
    fires many clicks. Two thresholds fix it: it closes below `close_at`
    and only opens again above `open_at`.
    """

    def __init__(self, close_at=0.25, open_at=0.40):
        self.close_at = close_at
        self.open_at = open_at
        self.pinching = False

    def update(self, ratio):
        """Returns "start", "end" or None, and updates self.pinching."""
        if not self.pinching and ratio < self.close_at:
            self.pinching = True
            return "start"

        if self.pinching and ratio > self.open_at:
            self.pinching = False
            return "end"

        return None

    def reset(self):
        self.pinching = False

class HoldDetector:
    """Fires once when the same gesture is held for `seconds`.

    A small state machine: IDLE -> HOLDING (timer running) -> FIRED.
    It only goes back to IDLE when the gesture changes, so holding a fist
    for ten seconds minimizes one window, not twenty.
    """

    def __init__(self, seconds=0.8):
        self.seconds = seconds
        self.gesture = None
        self.since = 0.0
        self.fired = False

    def update(self, gesture):
        """Returns the gesture name the moment it completes, else None."""
        now = time.monotonic()

        if gesture != self.gesture:
            self.gesture = gesture
            self.since = now
            self.fired = False
            return None

        if gesture is not None and not self.fired and now - self.since >= self.seconds:
            self.fired = True
            return gesture

        return None

    def progress(self):
        """0-1 while holding, to draw a progress bar."""
        if self.gesture is None or self.fired:
            return 0.0
        return min((time.monotonic() - self.since) / self.seconds, 1.0)

def classify_pose(hand):
    """Simple static poses from the fingers that are up."""
    up = hand.fingers_up()
    count = sum(up.values())

    if count == 0 or (count == 1 and up["thumb"]):
        return "fist"
    if count == 5:
        return "open_palm"
    if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
        return "peace"
    if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "point"
    return None
