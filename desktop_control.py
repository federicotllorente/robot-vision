"""Move the mouse and manage windows with the Windows API (ctypes, no extra packages)."""
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

# Declare window-handle types: by default ctypes assumes 32-bit ints, which can
# corrupt handles on 64-bit Windows
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsZoomed.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.UINT,
]

# Without this, on a scaled display (125%, 150%...) Windows reports a smaller
# "virtual" screen and the cursor would never reach the real edges
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    user32.SetProcessDPIAware()

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9

SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010

# Smallest window size we allow when resizing with the hands
MIN_WINDOW_SIZE = 300

def screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def move_mouse(x, y):
    user32.SetCursorPos(int(x), int(y))

def mouse_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def mouse_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def foreground_window():
    return user32.GetForegroundWindow()

def window_title(hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value

def window_rect(hwnd):
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

def is_maximized(hwnd):
    return bool(user32.IsZoomed(hwnd))

def minimize(hwnd):
    user32.ShowWindow(hwnd, SW_MINIMIZE)

def toggle_maximize(hwnd):
    user32.ShowWindow(hwnd, SW_RESTORE if is_maximized(hwnd) else SW_MAXIMIZE)

def resize_around_center(hwnd, start_rect, scale):
    """Resize from the size it had when the gesture started, keeping its center."""
    if is_maximized(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    left, top, width, height = start_rect
    screen_w, screen_h = screen_size()
    new_w = int(min(max(width * scale, MIN_WINDOW_SIZE), screen_w))
    new_h = int(min(max(height * scale, MIN_WINDOW_SIZE), screen_h))
    new_left = left + (width - new_w) // 2
    new_top = top + (height - new_h) // 2

    user32.SetWindowPos(
        hwnd, 0, new_left, new_top, new_w, new_h, SWP_NOZORDER | SWP_NOACTIVATE
    )
