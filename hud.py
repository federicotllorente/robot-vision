"""Floating "Iron Man" cards drawn over the camera image.

Pinch a card to grab it and move it. While one hand holds a card, pinch with
the other hand (anywhere) and pull the hands apart or together to scale it.
"""
from collections import deque

import cv2
import numpy as np

import ui_theme as ui
from hand_tracking import distance, midpoint

MIN_CARD_SCALE = 0.5
MAX_CARD_SCALE = 3.0

class Card:
    """Position and size in design units (see ui_theme.u), converted here."""

    def __init__(self, title, x, y, width, height, accent=ui.CYAN):
        self.title = title
        self.x, self.y = ui.u(x), ui.u(y)
        self.base_width, self.base_height = ui.u(width), ui.u(height)
        self.scale = 1.0
        self.accent = accent

    @property
    def width(self):
        return int(self.base_width * self.scale)

    @property
    def height(self):
        return int(self.base_height * self.scale)

    def contains(self, point):
        return (
            self.x <= point[0] <= self.x + self.width
            and self.y <= point[1] <= self.y + self.height
        )

    def center(self):
        return (self.x + self.width // 2, self.y + self.height // 2)

    def draw(self, canvas, highlighted):
        top_left = (int(self.x), int(self.y))
        bottom_right = (int(self.x) + self.width, int(self.y) + self.height)
        ui.panel(canvas, top_left, bottom_right, accent=self.accent, alpha=0.7, highlighted=highlighted)

        # Title in Orbitron with a thin accent line under it
        title_y = int(self.y + ui.u(18) * self.scale)
        ui.text(
            canvas, self.title.upper(), (self.x + ui.u(9) * self.scale, title_y),
            size=9 * self.scale, color=self.accent, font="title", weight=700,
        )
        line_y = title_y + int(ui.u(6) * self.scale)
        cv2.line(
            canvas, (int(self.x + ui.u(9) * self.scale), line_y),
            (int(self.x + self.width - ui.u(9) * self.scale), line_y),
            self.accent, max(ui.u(0.5), 1), cv2.LINE_AA,
        )
        self.draw_content(canvas)

    def put_text(self, canvas, value, line, color=ui.TEXT):
        # Body lines below the title; everything grows with the card
        line_height = ui.u(17) * self.scale
        position = (self.x + ui.u(9) * self.scale, self.y + ui.u(28) * self.scale + line_height * line)
        ui.text(canvas, value, position, size=12 * self.scale, color=color)

    def draw_content(self, canvas):
        pass

class NoteCard(Card):
    def __init__(self, title, lines, x, y, width=200, height=95, accent=ui.CYAN):
        super().__init__(title, x, y, width, height, accent)
        self.lines = lines

    def draw_content(self, canvas):
        for i, line in enumerate(self.lines):
            self.put_text(canvas, line, i + 1)

class ChartCard(Card):
    """Live line chart of the last values pushed into it."""

    def __init__(self, title, x, y, width=210, height=110, accent=ui.TEAL, max_points=90):
        super().__init__(title, x, y, width, height, accent)
        self.values = deque(maxlen=max_points)

    def push(self, value):
        self.values.append(value)

    def draw_content(self, canvas):
        if len(self.values) < 2:
            return

        # Chart area: below the title and the value line, with a small margin
        margin = int(ui.u(9) * self.scale)
        top = self.y + int(ui.u(46) * self.scale)
        left = self.x + margin
        width = self.width - 2 * margin
        height = self.y + self.height - margin - top

        low, high = min(self.values), max(self.values)
        span = (high - low) or 1

        points = []
        for i, value in enumerate(self.values):
            px = left + width * i / (self.values.maxlen - 1)
            py = top + height * (1 - (value - low) / span)
            points.append((int(px), int(py)))

        line = [np.array(points, dtype=np.int32)]
        thickness = max(int(ui.u(1) * self.scale), 1)
        cv2.polylines(canvas, line, False, ui.GLOW, thickness * 3, cv2.LINE_AA)
        cv2.polylines(canvas, line, False, self.accent, thickness, cv2.LINE_AA)
        self.put_text(canvas, f"now: {self.values[-1]}", 1, color=self.accent)

class Hud:
    def __init__(self, cards):
        self.cards = list(cards)  # last = drawn on top
        self.grabs = {}  # hand label -> card it holds
        self.offsets = {}  # hand label -> pinch point minus card corner (None = recompute)
        self.scaling = None  # (card, start distance between hands, start scale)

    def card_at(self, point):
        # Search from the top-most card down
        for card in reversed(self.cards):
            if card.contains(point):
                return card
        return None

    def update(self, pinches):
        """pinches: dict hand label -> (event, pinching, point) for this frame.

        event is "start", "end" or None (see PinchDetector).
        """
        # 1. Releases first: hands that opened the pinch or left the image
        for label in list(self.grabs):
            if label not in pinches or not pinches[label][1]:
                self.release(label)

        # 2. New pinches
        for label, (event, _pinching, point) in pinches.items():
            if event == "start":
                self.grab(label, point, pinches)

        # 3. Two hands on one card -> scale; otherwise drag
        if self.scaling is not None:
            self.apply_scaling(pinches)
        else:
            for label, card in self.grabs.items():
                point = pinches[label][2]
                if self.offsets[label] is None:
                    self.offsets[label] = (point[0] - card.x, point[1] - card.y)
                offset = self.offsets[label]
                card.x, card.y = point[0] - offset[0], point[1] - offset[1]

    def grab(self, label, point, pinches):
        if self.grabs:
            # A second hand joins: scale the card the first hand is holding
            other = next(iter(self.grabs))
            card = self.grabs[other]
            self.grabs[label] = card
            self.offsets[label] = None
            start = max(distance(pinches[other][2], point), 1.0)
            self.scaling = (card, start, card.scale)
            return

        card = self.card_at(point)
        if card is None:
            return

        # Bring it to the front
        self.cards.remove(card)
        self.cards.append(card)
        self.grabs[label] = card
        self.offsets[label] = (point[0] - card.x, point[1] - card.y)

    def apply_scaling(self, pinches):
        card, start_distance, start_scale = self.scaling
        point_a, point_b = [pinches[label][2] for label in self.grabs]

        scale = start_scale * distance(point_a, point_b) / start_distance
        card.scale = min(max(scale, MIN_CARD_SCALE), MAX_CARD_SCALE)

        # Keep the card centered between both hands
        center = midpoint(point_a, point_b)
        card.x = center[0] - card.width // 2
        card.y = center[1] - card.height // 2

    def release(self, label):
        self.grabs.pop(label, None)
        self.offsets.pop(label, None)
        self.scaling = None

        # The hand still holding the card keeps dragging it from where it is
        for other in self.offsets:
            self.offsets[other] = None

    def draw(self, frame):
        held = set(self.grabs.values())
        for card in self.cards:
            card.draw(frame, card in held)
