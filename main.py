
import cv2
import numpy as np

# 1. Conectar con la webcam USB
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "No se pudo abrir la webcam. Prueba otro índice de cámara."
    )

while True:
    # 2. Capturar una imagen de la webcam
    success, frame = camera.read()

    if not success:
        print("No se pudo capturar la imagen.")
        break

    # 3. Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 4. Definir los rangos del color rojo
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([179, 255, 255])

    # 5. Crear una máscara que identifica los píxeles rojos
    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    # 6. Encontrar las regiones rojas de la imagen
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 7. Seleccionar la región roja más grande
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Ignorar pequeñas manchas de ruido
        if area > 500:
            x, y, w, h = cv2.boundingRect(largest)

            # Centro del objeto en píxeles
            center_x = x + w // 2
            center_y = y + h // 2

            # Dibujar un rectángulo y el centro
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            cv2.circle(
                frame, (center_x, center_y),
                5, (255, 0, 0), -1
            )

            # Mostrar las coordenadas
            text = f"X: {center_x}, Y: {center_y}"
            cv2.putText(
                frame, text, (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            print(f"Objeto rojo: X={center_x}, Y={center_y}")

    # 8. Mostrar la imagen y la máscara
    cv2.imshow("Vision del robot", frame)
    cv2.imshow("Mascara roja", mask)

    # Pulsar Q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
