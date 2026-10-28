
import time

import cv2

import mqtt_link
import ui_theme as ui
import vision
from aspect_window import AspectWindow

WINDOW_NAME = "Robot Vision - Object Detection"

# How often the detections are sent to MQTT (seconds)
PUBLISH_INTERVAL = 0.2

# Bottom of the header + "looking for" strip; the grid starts below it
INFO_BOTTOM = ui.u(58)

model = vision.load_model()
targets = vision.get_targets(model)

mqtt_client = mqtt_link.connect()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

window = None
last_summary = None
last_publish = 0.0
last_time = time.monotonic()
fps = 0.0

while True:
    success, frame = camera.read()

    if not success:
        print("Could not capture a frame.")
        break

    height, width = frame.shape[:2]

    # Apply new targets requested by the assistant
    requested = mqtt_link.get_target_request()

    if requested is not None and vision.set_targets(model, requested):
        targets = vision.get_targets(model)
        print(f"New targets: {', '.join(targets)}")

    # Detect on the camera image: the robot needs coordinates in camera pixels
    detections = vision.detect_objects(model, frame)

    # Send the detections a few times per second, not on every frame
    now = time.time()

    if now - last_publish >= PUBLISH_INTERVAL:
        mqtt_link.publish_objects(
            mqtt_client, detections, targets, width, height
        )
        last_publish = now

    # Draw on a bigger canvas: the window scales it, so the text stays sharp
    canvas = ui.prepare_canvas(frame)
    scale = canvas.shape[1] / width

    vision.draw_grid(canvas, top_margin=INFO_BOTTOM)

    for detection in detections:
        vision.draw_detection(canvas, detection, scale, top_margin=INFO_BOTTOM)

    tick = time.monotonic()
    fps = 0.9 * fps + 0.1 / max(tick - last_time, 1e-6)
    last_time = tick

    mqtt_state = "MQTT ON" if mqtt_client.is_connected() else "MQTT OFF"
    ui.header(canvas, "ROBOT VISION", right_text=f"{mqtt_state}   {fps:4.1f} FPS")

    # Strip under the header with what YOLO-World is looking for
    ui.blend_rect(canvas, (0, ui.u(34)), (canvas.shape[1], INFO_BOTTOM), ui.PANEL, 0.6)
    ui.text(canvas, "LOOKING FOR", (ui.u(12), ui.u(50)), size=7, color=ui.MUTED, font="title", weight=700)
    ui.text(canvas, ", ".join(targets), (ui.u(84), ui.u(50)), size=11, color=ui.TEXT)

    if detections:
        ui.status_line(canvas, f"{len(detections)} object(s) detected", ui.TEAL)
    else:
        ui.status_line(canvas, "No objects detected", ui.AMBER)

    # Print only when the list of objects or their cells changes
    summary = sorted(
        f"{d['color']} {d['name']} in {d['cell']}" for d in detections
    )

    if summary != last_summary:
        if summary:
            print("Objects: " + " | ".join(summary))
        else:
            print("No objects detected")
        last_summary = summary

    # The window is created with the first canvas, to know its proportions
    if window is None:
        window = AspectWindow(WINDOW_NAME, canvas.shape[1], canvas.shape[0])

    window.show(canvas)

    # Press Q (or close the window) to quit
    if cv2.waitKey(1) & 0xFF == ord("q") or not window.is_open():
        break

camera.release()
cv2.destroyAllWindows()
mqtt_link.disconnect(mqtt_client)
