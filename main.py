
import cv2
import numpy as np

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

last_cell = None

while True:
    success, frame = camera.read()

    if not success:
        print("Could not capture a frame.")
        break

    height, width = frame.shape[:2]

    # Calculate the grid boundaries
    x1 = width // 3
    x2 = (2 * width) // 3
    y1 = height // 3
    y2 = (2 * height) // 3

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red color ranges
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([179, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    # Find red objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the 3x3 grid
    cv2.line(frame, (x1, 0), (x1, height), (255, 255, 0), 2)
    cv2.line(frame, (x2, 0), (x2, height), (255, 255, 0), 2)
    cv2.line(frame, (0, y1), (width, y1), (255, 255, 0), 2)
    cv2.line(frame, (0, y2), (width, y2), (255, 255, 0), 2)

    # Label the grid columns
    cv2.putText(
        frame, "LEFT", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "CENTER", (x1 + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "RIGHT", (x2 + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
    )

    object_found = False

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Ignore small red regions
        if area > 500:
            object_found = True

            x, y, w, h = cv2.boundingRect(largest)

            # Calculate the object's center
            center_x = x + w // 2
            center_y = y + h // 2

            # Determine the horizontal position
            if center_x < x1:
                horizontal = "LEFT"
            elif center_x < x2:
                horizontal = "CENTER"
            else:
                horizontal = "RIGHT"

            # Determine the vertical position
            if center_y < y1:
                vertical = "TOP"
            elif center_y < y2:
                vertical = "MIDDLE"
            else:
                vertical = "BOTTOM"

            # Combine both positions
            cell = f"{vertical}-{horizontal}"

            # Draw the detected object
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Display the position and pixel coordinates
            label = f"{cell} | X:{center_x} Y:{center_y}"

            cv2.putText(
                frame,
                label,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # Print only when the object changes cells
            if cell != last_cell:
                print(
                    f"Object: {cell} | "
                    f"X={center_x}, Y={center_y}"
                )
                last_cell = cell

    # Reset the previous position when the object disappears
    if not object_found:
        last_cell = None

        cv2.putText(
            frame,
            "No red object detected",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    # Display the camera feed and the binary mask
    cv2.imshow("Robot Vision - 3x3 Grid", frame)
    cv2.imshow("Red Object Mask", mask)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
