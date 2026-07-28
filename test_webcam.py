import cv2

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Webcam tidak berhasil dibuka")

while True:
    success, frame = camera.read()

    if not success:
        print("Gagal membaca frame")
        break

    cv2.imshow("Webcam Raspberry Pi", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
