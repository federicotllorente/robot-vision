import json
import queue
import time

import paho.mqtt.client as mqtt

# Same broker the assistant (Kimmi) uses
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883

# Vision -> everyone: latest detections (retained)
OBJECTS_TOPIC = "robot/vision/objects"

# Assistant -> vision: list of object names to look for
TARGETS_TOPIC = "robot/vision/targets"

# Target lists received from MQTT, waiting for the main loop.
# The MQTT callbacks run in another thread, so we never touch the
# model from there: we only queue the request.
target_requests = queue.Queue()

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"MQTT connection refused: {reason_code}")
        return

    print(f"MQTT connected to {MQTT_HOST}:{MQTT_PORT}")

    # Subscribe here so it also happens again after a reconnect
    client.subscribe(TARGETS_TOPIC)

def on_message(client, userdata, message):
    try:
        targets = json.loads(message.payload)
    except json.JSONDecodeError:
        print(f"Ignoring invalid targets message: {message.payload!r}")
        return

    if not isinstance(targets, list):
        print(f"Targets must be a JSON list, got: {targets!r}")
        return

    target_requests.put(targets)

def connect():
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="robot-vision",
    )
    client.on_connect = on_connect
    client.on_message = on_message

    # connect_async + loop_start: the network runs in a background
    # thread and reconnects by itself, so the camera keeps working
    # even if the broker is down.
    client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    return client

def get_target_request():
    # Return the newest pending target list, or None
    latest = None

    while not target_requests.empty():
        latest = target_requests.get_nowait()

    return latest

def publish_objects(client, detections, targets, width, height):
    objects = []

    for detection in detections:
        objects.append({
            "name": detection["name"],
            "color": detection["color"],
            "confidence": round(detection["confidence"], 2),
            "cell": detection["cell"],
            "center": list(detection["center"]),
            "box": list(detection["box"]),
        })

    payload = {
        "ts": time.time(),
        "frame": {"width": width, "height": height},
        "targets": targets,
        "objects": objects,
    }

    # retain=True: anyone who subscribes later gets the latest state at once
    client.publish(OBJECTS_TOPIC, json.dumps(payload), qos=0, retain=True)

def disconnect(client):
    client.loop_stop()
    client.disconnect()
