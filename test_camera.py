import cv2

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
# camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam tidak dapat dibuka")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

success, frame = camera.read()
camera.release()

if not success:
    raise RuntimeError("Gagal mengambil gambar dari webcam")

if not cv2.imwrite("capture.jpg", frame):
    raise RuntimeError("Gagal menyimpan gambar")

print("Berhasil menyimpan capture.jpg")
print("Ukuran gambar:", frame.shape)
