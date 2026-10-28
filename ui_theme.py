"""Shared look for robot-vision, matching Kimmi's desktop window.

Keep the palette in sync with ai-assistant-test/desktop/theme.py.

Everything is drawn on a canvas CANVAS_WIDTH pixels wide (twice the
camera), so the text stays sharp when the window is enlarged. Layout sizes are
written in "design units" for a 640 px wide image and converted with u().
"""
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- palette
# Colors are BGR (OpenCV order). The RGB value from Kimmi is in the comment.
DEEP = (15, 8, 2)  # rgb(2, 8, 15)      Kimmi's background
PANEL = (40, 22, 8)  # rgb(8, 22, 40)     card / bar background
CYAN = (255, 212, 0)  # rgb(0, 212, 255)   Kimmi's wave: main accent
GLOW = (255, 180, 0)  # rgb(0, 180, 255)   Kimmi's glow: secondary accent
TEAL = (182, 196, 46)  # rgb(46, 196, 182)  Kimmi's tray icon: "ok" / active
AMBER = (0, 190, 255)  # rgb(255, 190, 0)   warnings, "holding" progress
ERROR = (96, 64, 255)  # rgb(255, 64, 96)   Kimmi's error color
TEXT = (255, 246, 232)  # rgb(232, 246, 255) main text
MUTED = (200, 165, 120)  # rgb(120, 165, 200) secondary text

# ---------------------------------------------------------------- sizes
CANVAS_WIDTH = 1280
DESIGN_WIDTH = 640
UNIT = CANVAS_WIDTH / DESIGN_WIDTH

def u(value):
    """Design units -> canvas pixels."""
    return int(round(value * UNIT))

# ---------------------------------------------------------------- fonts
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")

# Orbitron: titles and labels (sci-fi HUD look). Rajdhani: readable body text.
TITLE_FONT = cv2.FontFace(os.path.join(FONTS_DIR, "Orbitron-Variable.ttf"))
BODY_FONT = cv2.FontFace(os.path.join(FONTS_DIR, "Rajdhani-SemiBold.ttf"))

def _font(kind):
    return TITLE_FONT if kind == "title" else BODY_FONT

def text(img, value, org, size=14, color=TEXT, font="body", weight=0, shadow=True):
    """Draw text. org = bottom-left in canvas pixels; size in design units."""
    org = (int(org[0]), int(org[1]))
    if shadow:
        # Soft dark offset copy: readable over any camera background
        offset = max(u(1), 1)
        cv2.putText(img, value, (org[0] + offset, org[1] + offset), DEEP, _font(font), u(size), weight)
    cv2.putText(img, value, org, color, _font(font), u(size), weight)

def text_width(img, value, size=14, font="body", weight=0):
    rect = cv2.getTextSize(img.shape[1::-1], value, (0, 0), _font(font), u(size), weight)
    return rect[2]

# ---------------------------------------------------------------- shapes
def prepare_canvas(frame):
    """Camera frame -> canvas: scaled to CANVAS_WIDTH and tinted towards Kimmi's blue."""
    height, width = frame.shape[:2]
    canvas = cv2.resize(
        frame, (CANVAS_WIDTH, int(height * CANVAS_WIDTH / width)), interpolation=cv2.INTER_LINEAR
    )
    tint = np.full_like(canvas, DEEP)
    return cv2.addWeighted(canvas, 0.78, tint, 0.22, 0)

def blend_rect(img, top_left, bottom_right, color, alpha):
    """Fill a rectangle with transparency, blending only that region."""
    height, width = img.shape[:2]
    x1, y1 = max(int(top_left[0]), 0), max(int(top_left[1]), 0)
    x2, y2 = min(int(bottom_right[0]), width), min(int(bottom_right[1]), height)
    if x2 <= x1 or y2 <= y1:
        return
    region = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(region, 1 - alpha, np.full_like(region, color), alpha, 0)

def corner_brackets(img, top_left, bottom_right, color, length=10, thickness=2):
    """The four "[ ]" corners of a HUD frame."""
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    n = min(u(length), (x2 - x1) // 3, (y2 - y1) // 3)
    t = u(thickness / 2)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * n, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * n), color, t, cv2.LINE_AA)

def panel(img, top_left, bottom_right, accent=CYAN, alpha=0.62, highlighted=False):
    """Translucent HUD panel: dark fill, thin border and bright corners."""
    blend_rect(img, top_left, bottom_right, PANEL, alpha)
    p1 = (int(top_left[0]), int(top_left[1]))
    p2 = (int(bottom_right[0]), int(bottom_right[1]))
    border = accent if highlighted else MUTED
    cv2.rectangle(img, p1, p2, border, u(1.5 if highlighted else 0.5), cv2.LINE_AA)
    corner_brackets(img, p1, p2, accent, thickness=3 if highlighted else 2)

def glow_line(img, p1, p2, color=CYAN, thickness=1.5):
    """Line with a faint wider halo, like Kimmi's wave stroke."""
    p1, p2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, GLOW, u(thickness * 2.5), cv2.LINE_AA)
    cv2.line(img, p1, p2, color, u(thickness), cv2.LINE_AA)

def header(img, title, tabs=(), active=None, right_text=""):
    """Top bar: app title, optional tabs (the active one highlighted) and a right-side text."""
    width = img.shape[1]
    blend_rect(img, (0, 0), (width, u(34)), PANEL, 0.75)
    cv2.line(img, (0, u(34)), (width, u(34)), CYAN, u(1), cv2.LINE_AA)

    text(img, title, (u(12), u(23)), size=13, color=CYAN, font="title", weight=700)
    x = u(14) + text_width(img, title, 13, "title", 700) + u(18)

    for tab in tabs:
        is_active = tab == active
        tab_w = text_width(img, tab, 10, "title", 600) + u(14)
        if is_active:
            panel(img, (x - u(7), u(7)), (x + tab_w - u(7), u(28)), accent=CYAN, alpha=0.5, highlighted=True)
        text(img, tab, (x, u(22)), size=10, color=CYAN if is_active else MUTED, font="title", weight=600)
        x += tab_w + u(8)

    if right_text:
        right_w = text_width(img, right_text, 12)
        text(img, right_text, (width - right_w - u(12), u(23)), size=12, color=MUTED)

def status_line(img, value, color=TEXT):
    """Message at the bottom-left, on a small panel."""
    height = img.shape[0]
    width = text_width(img, value, 13)
    panel(img, (u(8), height - u(30)), (u(8) + width + u(16), height - u(8)), accent=color, alpha=0.6)
    text(img, value, (u(16), height - u(14)), size=13, color=color)

def progress_bar(img, origin, width, progress, label="", color=AMBER):
    """Thin bar in design units at origin (canvas px)."""
    x, y = int(origin[0]), int(origin[1])
    w, h = u(width), u(5)
    blend_rect(img, (x, y), (x + w, y + h), PANEL, 0.7)
    cv2.rectangle(img, (x, y), (x + int(w * progress), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), MUTED, 1)
    if label:
        text(img, label, (x + w + u(8), y + h), size=11, color=color)
