"""OpenCV window you can resize with the mouse while it keeps its proportions.

cv2.WINDOW_NORMAL lets the user resize the window and stretches the image to
fill it. On its own, dragging an edge would squash the image. While you drag,
Windows sends WM_SIZING with the rectangle the window is about to take; we
intercept it ("subclassing" the window) and fix that rectangle so its inner
area keeps the image's width/height ratio.
"""
import ctypes
from ctypes import wintypes

import cv2

user32 = ctypes.windll.user32

# Real pixels on scaled displays (otherwise Windows blurs the window)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    user32.SetProcessDPIAware()

WM_SIZING = 0x0214
GWLP_WNDPROC = -4

# Which edge or corner is being dragged (wParam of WM_SIZING)
WMSZ_LEFT, WMSZ_RIGHT, WMSZ_TOP = 1, 2, 3
WMSZ_TOPLEFT, WMSZ_TOPRIGHT, WMSZ_BOTTOM = 4, 5, 6
WMSZ_BOTTOMLEFT, WMSZ_BOTTOMRIGHT = 7, 8

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.SetWindowLongPtrW.restype = ctypes.c_void_p
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
user32.CallWindowProcW.restype = LRESULT
user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]

MIN_WIDTH = 320

class AspectWindow:
    def __init__(self, name, content_width, content_height, initial_width=960):
        self.name = name
        self.aspect = content_width / content_height

        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, initial_width, int(initial_width / self.aspect))

        self.hwnd = user32.FindWindowW(None, name)
        self._old_proc = None
        self._new_proc = None  # keep a reference: if Python frees it, Windows crashes

        if self.hwnd:
            self._new_proc = WNDPROC(self._window_proc)
            self._old_proc = user32.SetWindowLongPtrW(
                self.hwnd, GWLP_WNDPROC, ctypes.cast(self._new_proc, ctypes.c_void_p)
            )
        else:
            print(f"Window '{name}' not found: it can be resized but not proportionally.")

    def _border(self):
        # Title bar + borders = outer size minus inner (client) size
        outer, inner = wintypes.RECT(), wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(outer))
        user32.GetClientRect(self.hwnd, ctypes.byref(inner))
        return (
            (outer.right - outer.left) - (inner.right - inner.left),
            (outer.bottom - outer.top) - (inner.bottom - inner.top),
        )

    def _window_proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_SIZING:
            self._keep_aspect(wparam, wintypes.RECT.from_address(lparam))
            return 1
        return user32.CallWindowProcW(self._old_proc, hwnd, msg, wparam, lparam)

    def _keep_aspect(self, edge, rect):
        border_w, border_h = self._border()

        if edge in (WMSZ_TOP, WMSZ_BOTTOM):
            # Dragging top/bottom: the height decides, the width follows
            inner_h = max(rect.bottom - rect.top - border_h, int(MIN_WIDTH / self.aspect))
            inner_w = int(inner_h * self.aspect)
        else:
            # Sides and corners: the width decides, the height follows
            inner_w = max(rect.right - rect.left - border_w, MIN_WIDTH)
            inner_h = int(inner_w / self.aspect)

        # Move only the edges being dragged; the opposite ones stay put
        if edge in (WMSZ_LEFT, WMSZ_TOPLEFT, WMSZ_BOTTOMLEFT):
            rect.left = rect.right - inner_w - border_w
        else:
            rect.right = rect.left + inner_w + border_w

        if edge in (WMSZ_TOP, WMSZ_TOPLEFT, WMSZ_TOPRIGHT):
            rect.top = rect.bottom - inner_h - border_h
        else:
            rect.bottom = rect.top + inner_h + border_h

    def show(self, image):
        cv2.imshow(self.name, image)

    def is_open(self):
        # The user closed the window with the X button
        return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
