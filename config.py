from __future__ import annotations

import os
import socket


# ============================================================
# IDENTITAS PERANGKAT
# ============================================================

DEVICE_ID = os.getenv(
    "DEVICE_ID",
    socket.gethostname(),
)


# ============================================================
# KONFIGURASI MQTT
# ============================================================

# Ganti dengan alamat IP server atau broker MQTT.
MQTT_BROKER_HOST = os.getenv(
    "MQTT_BROKER_HOST",
    "localhost",
)

MQTT_BROKER_PORT = int(
    os.getenv(
        "MQTT_BROKER_PORT",
        "1883",
    )
)

MQTT_KEEPALIVE_SECONDS = int(
    os.getenv(
        "MQTT_KEEPALIVE_SECONDS",
        "60",
    )
)

# Kosongkan jika broker belum memakai autentikasi.
MQTT_USERNAME = os.getenv(
    "MQTT_USERNAME",
    "",
)

MQTT_PASSWORD = os.getenv(
    "MQTT_PASSWORD",
    "",
)

MQTT_CLIENT_ID = os.getenv(
    "MQTT_CLIENT_ID",
    "raspberry_ai"
)

MQTT_QOS_FRAME = 0
MQTT_QOS_STATUS = 1
MQTT_QOS_RESULT = 1


# ============================================================
# MQTT TOPIC
# ============================================================

FRAME_TOPIC = (
    f"sampah/camera/{DEVICE_ID}/frame"
)

STATUS_TOPIC = (
    f"sampah/camera/{DEVICE_ID}/status"
)

RESULT_TOPIC = (
    f"sampah/hasil/{DEVICE_ID}"
)

COMMAND_TOPIC = (
    "sampah/command"
)


# ============================================================
# KONFIGURASI USB WEBCAM
# ============================================================

CAMERA_INDEX = int(
    os.getenv(
        "CAMERA_INDEX",
        "0",
    )
)

FRAME_WIDTH = int(
    os.getenv(
        "FRAME_WIDTH",
        "640",
    )
)

FRAME_HEIGHT = int(
    os.getenv(
        "FRAME_HEIGHT",
        "480",
    )
)

CAMERA_WARMUP_SECONDS = float(
    os.getenv(
        "CAMERA_WARMUP_SECONDS",
        "2",
    )
)


# ============================================================
# KONFIGURASI PENGIRIMAN GAMBAR
# ============================================================

JPEG_QUALITY = int(
    os.getenv(
        "JPEG_QUALITY",
        "70",
    )
)

# Interval 1,0 berarti satu gambar dikirim setiap detik.
CAPTURE_INTERVAL_SECONDS = float(
    os.getenv(
        "CAPTURE_INTERVAL_SECONDS",
        "1.0",
    )
)

MAX_PAYLOAD_BYTES = int(
    os.getenv(
        "MAX_PAYLOAD_BYTES",
        "500000",
    )
)

FRAME_FORMAT = "jpeg"


# ============================================================
# KONFIGURASI TAMPILAN
# ============================================================

SHOW_PREVIEW = (
    os.getenv(
        "SHOW_PREVIEW",
        "true",
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

PREVIEW_WINDOW_NAME = (
    "Raspberry Pi MQTT Camera"
)


# ============================================================
# KONFIGURASI RECONNECT
# ============================================================

MQTT_RECONNECT_MIN_SECONDS = 1
MQTT_RECONNECT_MAX_SECONDS = 30

CAMERA_RETRY_DELAY_SECONDS = 0.2


# ============================================================
# VALIDASI KONFIGURASI
# ============================================================

def validate_config() -> None:
    if not MQTT_BROKER_HOST:
        raise ValueError(
            "MQTT_BROKER_HOST belum diisi."
        )

    if not 1 <= MQTT_BROKER_PORT <= 65535:
        raise ValueError(
            "MQTT_BROKER_PORT tidak valid."
        )

    if FRAME_WIDTH <= 0 or FRAME_HEIGHT <= 0:
        raise ValueError(
            "Ukuran frame harus lebih dari nol."
        )

    if not 1 <= JPEG_QUALITY <= 100:
        raise ValueError(
            "JPEG_QUALITY harus berada "
            "antara 1 sampai 100."
        )

    if CAPTURE_INTERVAL_SECONDS <= 0:
        raise ValueError(
            "CAPTURE_INTERVAL_SECONDS "
            "harus lebih dari nol."
        )

    if MAX_PAYLOAD_BYTES <= 0:
        raise ValueError(
            "MAX_PAYLOAD_BYTES harus "
            "lebih dari nol."
        )
