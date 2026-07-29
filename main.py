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

    except (
        TypeError,
        ValueError,
    ):
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
            0.0
        )
    )


    return (
        f"{label} "
        f"({confidence:.1%})"
    )



def main() -> None:


    config.validate_config()


    # ============================
    # SERVICE INIT
    # ============================

    camera = CameraService()


    encoder = FrameEncoder()


    mqtt_service = MQTTService()



    ai = InferenceService(
        model_path=(
            "Paradigma B No Mix/"
            "efficientnet_b0.tflite"
        ),

        labels_path=(
            "labels.txt"
        )
    )



    frame_id = 0


    last_publish_time = 0.0


    last_prediction_time = 0.0


    prediction_interval = 2.0
    # AI predict setiap 2 detik



    latest_prediction = None



    try:


        print("=" * 60)

        print(
            "RASPBERRY PI AI MQTT SYSTEM"
        )

        print("=" * 60)


        print(
            f"Device ID : "
            f"{config.DEVICE_ID}"
        )


        print(
            f"Broker    : "
            f"{config.MQTT_BROKER_HOST}:"
            f"{config.MQTT_BROKER_PORT}"
        )


        print(
            f"Topic     : "
            f"{config.FRAME_TOPIC}"
        )


        print("=" * 60)



        # ============================
        # START CAMERA
        # ============================


        camera.open()



        # ============================
        # START MQTT
        # ============================


        mqtt_service.start(
            # connection_timeout_seconds=10.0
        )



        print(
            "System berjalan"
        )



        while True:



            success, frame, camera_read_ms = (
                camera.read_frame()
            )



            if not success or frame is None:


                print(
                    "Gagal membaca kamera"
                )


                time.sleep(
                    config
                    .CAMERA_RETRY_DELAY_SECONDS
                )


                continue




            current_time = time.monotonic()



            # ==================================================
            # AI PREDICTION
            # ==================================================


            if (
                current_time -
                last_prediction_time
                >= prediction_interval
            ):


                try:


                    prediction = (
                        ai.predict(
                            frame
                        )
                    )


                    latest_prediction = (
                        prediction
                    )


                    last_prediction_time = (
                        current_time
                    )



                    print(
                        "HASIL AI:",
                        prediction
                    )



                    # Kirim hasil AI ke ESP32

                    if mqtt_service.is_connected():


                        mqtt_service.publish_prediction_command(
                            prediction
                        )



                except Exception as error:


                    print(
                        "Inference error:",
                        error
                    )




            # ==================================================
            # SEND FRAME KE MQTT (OPSIONAL)
            # ==================================================


            publish_status = (
                "Tidak mengirim frame"
            )



            interval_reached = (

                current_time -
                last_publish_time

                >=

                config.CAPTURE_INTERVAL_SECONDS

            )



            if interval_reached:


                if mqtt_service.is_connected():


                    next_frame_id = (
                        frame_id + 1
                    )



                    try:


                        payload, metadata = (
                            encoder.build_payload(
                                frame=frame,
                                frame_id=next_frame_id,
                                camera_read_ms=(
                                    camera_read_ms
                                ),
                            )
                        )



                        published = (
                            mqtt_service.publish_frame(
                                payload
                            )
                        )



                        if published:


                            frame_id = (
                                next_frame_id
                            )


                            last_publish_time = (
                                current_time
                            )


                            publish_status = (
                                f"Frame "
                                f"{frame_id} terkirim"
                            )


                            print(

                                f"Frame {frame_id} | "

                                f"{metadata['width']}x"
                                f"{metadata['height']} | "

                                f"{len(payload)} byte"

                            )


                        else:


                            publish_status = (
                                "Frame gagal publish"
                            )



                    except Exception as error:


                        print(
                            "Frame error:",
                            error
                        )



                else:


                    publish_status = (
                        "MQTT belum connect"
                    )




            # ==================================================
            # DISPLAY STATUS
            # ==================================================


            ai_status = (
                format_classification_result(
                    latest_prediction
                )
            )



            preview_status = (

                f"{publish_status} | "
                f"AI: {ai_status}"

            )



            running = (
                camera.show_preview(
                    frame=frame,
                    status_text=preview_status,
                )
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
