from __future__ import annotations

import time
from typing import Any

import config
from camera_service import CameraService
from frame_encoder import FrameEncoder
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
        result.get("confidence", 0.0)
    )

    result_frame_id = result.get(
        "frame_id",
        "-",
    )

    return (
        f"Hasil frame {result_frame_id}: "
        f"{label} ({confidence:.1%})"
    )


def main() -> None:
    config.validate_config()

    camera = CameraService()
    encoder = FrameEncoder()
    mqtt_service = MQTTService()

    ai = InferenceService(
        model_path="model/efficientnet_b0.tflite",
        labels_path="labels.txt"
    )

    frame_id = 0
    last_publish_time = 0.0

    try:
        print("=" * 60)
        print("RASPBERRY PI CAMERA MQTT PUBLISHER")
        print("=" * 60)
        print(f"Device ID : {config.DEVICE_ID}")
        print(
            f"Broker    : "
            f"{config.MQTT_BROKER_HOST}:"
            f"{config.MQTT_BROKER_PORT}"
        )
        print(f"Topic     : {config.FRAME_TOPIC}")
        print(
            f"Interval  : "
            f"{config.CAPTURE_INTERVAL_SECONDS} detik"
        )
        print("=" * 60)

        camera.open()

        mqtt_service.start(
            connection_timeout_seconds=10.0
        )

        print("Program berjalan")
        print("Tekan q pada preview atau Ctrl+C untuk berhenti")

        while True:
            success, frame, camera_read_ms = (
                camera.read_frame()
            )

            if not success or frame is None:
                print("Gagal membaca frame webcam")

                time.sleep(
                    config.CAMERA_RETRY_DELAY_SECONDS
                )
                continue

            prediction = ai.predict(
                frame
            )


            print(
                "HASIL AI:",
                prediction
            )

            current_time = time.monotonic()

            publish_status = (
                "Menunggu jadwal pengiriman"
            )

            interval_reached = (
                current_time - last_publish_time
                >= config.CAPTURE_INTERVAL_SECONDS
            )

            if interval_reached:
                if mqtt_service.is_connected():
                    next_frame_id = frame_id + 1

                    try:
                        payload, metadata = (
                            encoder.build_payload(
                                frame=frame,
                                frame_id=next_frame_id,
                                camera_read_ms=camera_read_ms,
                            )
                        )

                        published = (
                            mqtt_service.publish_frame(
                                payload
                            )
                        )

                        if published:
                            frame_id = next_frame_id
                            last_publish_time = current_time

                            publish_status = (
                                f"Frame {frame_id} terkirim"
                            )

                            print(
                                f"Frame {frame_id} terkirim | "
                                f"{metadata['width']}x"
                                f"{metadata['height']} | "
                                f"{len(payload)} byte | "
                                f"camera "
                                f"{camera_read_ms:.2f} ms"
                            )

                        else:
                            publish_status = (
                                "Publish MQTT gagal"
                            )

                    except (
                        ValueError,
                        RuntimeError,
                    ) as error:
                        publish_status = (
                            "Frame gagal diproses"
                        )

                        print(
                            f"Kesalahan frame: {error}"
                        )

                else:
                    publish_status = (
                        "MQTT belum terhubung"
                    )

            latest_result = (
                mqtt_service.get_latest_result()
            )

            result_status = (
                format_classification_result(
                    latest_result
                )
            )

            preview_status = (
                f"{publish_status} | {result_status}"
            )

            continue_running = (
                camera.show_preview(
                    frame=frame,
                    status_text=preview_status,
                )
            )

            if not continue_running:
                print("Tombol q ditekan")
                break

    except KeyboardInterrupt:
        print("\nProgram dihentikan dengan Ctrl+C")

    except Exception as error:
        print(f"Program mengalami kesalahan: {error}")

        raise

    finally:
        mqtt_service.stop()
        camera.close()

        print("Program selesai")


if __name__ == "__main__":
    main()
