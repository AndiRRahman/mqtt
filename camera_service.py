from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

import config


class CameraService:
    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
    ) -> None:
        self.camera_index = camera_index
        self.camera: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            return

        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_V4L2,
        )

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            raise RuntimeError(
                f"Webcam pada index {self.camera_index} "
                "tidak berhasil dibuka."
            )

        time.sleep(
            config.CAMERA_WARMUP_SECONDS
        )

        success, frame = self.camera.read()

        if not success or frame is None:
            self.close()

            raise RuntimeError(
                "Webcam berhasil dibuka, tetapi frame "
                "tidak dapat dibaca."
            )

        print("Webcam berhasil dibuka")
        print(
            f"Resolusi aktual: "
            f"{frame.shape[1]}x{frame.shape[0]}"
        )

    def read_frame(
        self,
    ) -> tuple[bool, np.ndarray | None, float]:
        if self.camera is None:
            raise RuntimeError(
                "Webcam belum dibuka."
            )

        started = time.perf_counter()

        success, frame = self.camera.read()

        camera_read_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if not success or frame is None:
            return False, None, camera_read_ms

        return True, frame, camera_read_ms

    def get_information(
        self,
    ) -> dict[str, Any]:
        if self.camera is None:
            return {
                "opened": False,
                "camera_index": self.camera_index,
            }

        return {
            "opened": self.camera.isOpened(),
            "camera_index": self.camera_index,
            "width": int(
                self.camera.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            ),
            "height": int(
                self.camera.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            ),
            "fps": float(
                self.camera.get(
                    cv2.CAP_PROP_FPS
                )
            ),
            "backend": self.camera.getBackendName(),
        }

    def show_preview(
        self,
        frame: np.ndarray,
        status_text: str = "",
    ) -> bool:
        if not config.SHOW_PREVIEW:
            return True

        preview_frame = frame.copy()

        if status_text:
            cv2.putText(
                preview_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            config.PREVIEW_WINDOW_NAME,
            preview_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        return key != ord("q")

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        cv2.destroyAllWindows()

        print("Webcam ditutup")

    def __enter__(self) -> "CameraService":
        self.open()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.close()
