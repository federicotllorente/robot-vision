
import cv2
import numpy as np

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("No se pudo abrir la webcam.")

last_zone = None

while True:
    success, frame = camera.read()

    if not success:
        print("No se pudo capturar la imagen.")
        break

    height, width = frame.shape[:2]

    # Dividir la imagen en tres zonas
    left_limit = width // 3
    right_limit = (2 * width) // 3

    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detectar el color rojo
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([179, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    # Encontrar los objetos rojos
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Dibujar los límites de las zonas
    cv2.line(
        frame, (left_limit, 0), (left_limit, height),
        (255, 255, 0), 2
    )
    cv2.line(
        frame, (right_limit, 0), (right_limit, height),
        (255, 255, 0), 2
    )

    # Etiquetar las zonas
    cv2.putText(
        frame, "IZQUIERDA", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "CENTRO", (left_limit + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "DERECHA", (right_limit + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:
            x, y, w, h = cv2.boundingRect(largest)

            center_x = x + w // 2
            center_y = y + h // 2

            # Determinar la zona según la coordenada X
            if center_x < left_limit:
                zone = "IZQUIERDA"
            elif center_x < right_limit:
                zone = "CENTRO"
            else:
                zone = "DERECHA"

            # Dibujar el objeto detectado
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            cv2.circle(
                frame, (center_x, center_y),
                5, (255, 0, 0), -1
            )

            # Mostrar posición y zona
            cv2.putText(
                frame,
                f"{zone} | X:{center_x} Y:{center_y}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            # Informar solo cuando cambia de zona
            if zone != last_zone:
                print(
                    f"Objeto en {zone}: "
                    f"X={center_x}, Y={center_y}"
                )
                last_zone = zone

    cv2.imshow("Vision del robot", frame)
    cv2.imshow("Mascara roja", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
