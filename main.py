
import time

import cv2

import mqtt_link
import vision

# How often the detections are sent to MQTT (seconds)
PUBLISH_INTERVAL = 0.2

model = vision.load_model()
targets = vision.get_targets(model)

mqtt_client = mqtt_link.connect()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

last_summary = None
last_publish = 0.0

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

    detections = vision.detect_objects(model, frame)

    # Send the detections a few times per second, not on every frame
    now = time.time()

    if now - last_publish >= PUBLISH_INTERVAL:
        mqtt_link.publish_objects(
            mqtt_client, detections, targets, width, height
        )
        last_publish = now

    vision.draw_grid(frame)

    for detection in detections:
        vision.draw_detection(frame, detection)

    if detections:
        status = f"{len(detections)} object(s) detected"
        status_color = (0, 255, 0)
    else:
        status = "No objects detected"
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

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

    cv2.imshow("Robot Vision - Object Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
mqtt_link.disconnect(mqtt_client)
