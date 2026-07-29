from __future__ import annotations

import time
from typing import Any

import config

from camera_service import CameraService
from mqtt_service import MQTTService
from inference_service import InferenceService


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_classification_result(
    result: dict[str, Any] | None,
) -> str:
    if not result:
        return "Belum ada hasil klasifikasi"

    label = result.get(
        "label",
        "Unknown",
    )

    confidence = safe_float(
        result.get(
            "confidence",
            0.0,
        )
    )

    return (
        f"{label} "
        f"({confidence:.1%})"
    )


def main():

    config.validate_config()

    camera = CameraService()

    mqtt_service = MQTTService()

    ai = InferenceService(
        model_path="efficientnet_b0.tflite",
        labels_path="labels.txt"
    )

    last_prediction_time = 0.0
    prediction_interval = 2.0

    latest_prediction = None

    try:
        print("=" * 60)
        print("RASPBERRY PI AI MQTT SYSTEM")
        print("=" * 60)

        print(
            f"Broker : "
            f"{config.MQTT_BROKER_HOST}:"
            f"{config.MQTT_BROKER_PORT}"
        )

        print("=" * 60)

        camera.open()

        mqtt_service.start()

        print("System berjalan")

        while True:

            success, frame, camera_read_ms = (
                camera.read_frame()
            )

            if not success or frame is None:
                print(
                    "Gagal membaca kamera"
                )
                time.sleep(
                    config.CAMERA_RETRY_DELAY_SECONDS
                )
                continue

            current_time = time.monotonic()

            publish_status = (
                "Menunggu prediksi"
            )

            if (
                current_time -
                last_prediction_time
                >= prediction_interval
            ):

                try:
                    prediction = ai.predict(
                        frame
                    )

                    latest_prediction = prediction

                    last_prediction_time = (
                        current_time
                    )

                    print(
                        "HASIL AI:",
                        prediction
                    )

                    if mqtt_service.is_connected():

                        sent = (
                            mqtt_service
                            .publish_prediction_command(
                                prediction
                            )
                        )

                        if sent:
                            publish_status = (
                                "Command ESP32 terkirim"
                            )
                        else:
                            publish_status = (
                                "Command gagal dikirim"
                            )

                except Exception as error:
                    print(
                        "Inference error:",
                        error
                    )

                    publish_status = (
                        "Inference gagal"
                    )

            ai_status = (
                format_classification_result(
                    latest_prediction
                )
            )

            preview_status = (
                f"{publish_status} | "
                f"AI: {ai_status}"
            )

            running = camera.show_preview(
                frame,
                preview_status
            )

            if not running:
                print(
                    "Preview dihentikan"
                )
                break

    except KeyboardInterrupt:
        print(
            "\nProgram dihentikan"
        )

    except Exception as error:
        print(
            "System error:",
            error
        )
        raise

    finally:
        mqtt_service.stop()
        camera.close()

        print(
            "System selesai"
        )


if __name__ == "__main__":
    main()
