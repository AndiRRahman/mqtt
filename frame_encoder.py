from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

import config


class FrameEncoder:
    def __init__(
        self,
        jpeg_quality: int = config.JPEG_QUALITY,
        max_payload_bytes: int = config.MAX_PAYLOAD_BYTES,
    ) -> None:
        self.jpeg_quality = jpeg_quality
        self.max_payload_bytes = max_payload_bytes

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                "JPEG quality harus berada antara 1 sampai 100."
            )

    def encode_jpeg(
        self,
        frame: np.ndarray,
    ) -> bytes:
        if frame is None or frame.size == 0:
            raise ValueError("Frame kosong.")

        success, encoded_image = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not success:
            raise RuntimeError(
                "Frame gagal dikompres menjadi JPEG."
            )

        return encoded_image.tobytes()

    def create_metadata(
        self,
        frame: np.ndarray,
        frame_id: int,
        jpeg_size_bytes: int,
        camera_read_ms: float,
    ) -> dict[str, Any]:
        return {
            "device_id": config.DEVICE_ID,
            "frame_id": frame_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "format": config.FRAME_FORMAT,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "jpeg_quality": self.jpeg_quality,
            "jpeg_size_bytes": jpeg_size_bytes,
            "camera_read_ms": round(
                camera_read_ms,
                3,
            ),
        }

    def build_payload(
        self,
        frame: np.ndarray,
        frame_id: int,
        camera_read_ms: float,
    ) -> tuple[bytes, dict[str, Any]]:
        jpeg_bytes = self.encode_jpeg(frame)

        metadata = self.create_metadata(
            frame=frame,
            frame_id=frame_id,
            jpeg_size_bytes=len(jpeg_bytes),
            camera_read_ms=camera_read_ms,
        )

        metadata_bytes = json.dumps(
            metadata,
            separators=(",", ":"),
        ).encode("utf-8")

        # Struktur payload:
        # metadata JSON + newline + data JPEG
        payload = (
            metadata_bytes
            + b"\n"
            + jpeg_bytes
        )

        if len(payload) > self.max_payload_bytes:
            raise ValueError(
                "Ukuran payload melebihi batas. "
                f"Ukuran={len(payload)} byte, "
                f"batas={self.max_payload_bytes} byte."
            )

        return payload, metadata

    @staticmethod
    def split_payload(
        payload: bytes,
    ) -> tuple[dict[str, Any], bytes]:
        try:
            metadata_bytes, jpeg_bytes = payload.split(
                b"\n",
                1,
            )
        except ValueError as error:
            raise ValueError(
                "Payload tidak memiliki pemisah metadata dan JPEG."
            ) from error

        metadata = json.loads(
            metadata_bytes.decode("utf-8")
        )

        return metadata, jpeg_bytes

    @staticmethod
    def decode_jpeg(
        jpeg_bytes: bytes,
    ) -> np.ndarray:
        encoded_array = np.frombuffer(
            jpeg_bytes,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            encoded_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise RuntimeError(
                "Data JPEG gagal diterjemahkan kembali."
            )

        return frame
