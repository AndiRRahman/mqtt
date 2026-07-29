from __future__ import annotations

import os
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
        "Unknown"
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


def load_models():

    model_folder = "model"

    models = []

    if not os.path.exists(model_folder):
        raise FileNotFoundError(
            "Folder model tidak ditemukan"
        )


    for file in os.listdir(model_folder):

        if file.endswith(".tflite"):

            model_path = os.path.join(
                model_folder,
                file
            )

            print(
                "Loading model:",
                model_path
            )

            model = InferenceService(
                model_path=model_path,
                labels_path="labels.txt"
            )

            models.append(
                model
            )


    if len(models) == 0:
        raise RuntimeError(
            "Tidak ada model TFLite ditemukan"
        )


    print(
        f"{len(models)} model berhasil dimuat"
    )

    return models



def predict_with_all_models(
    models,
    frame
):

    results = []


    for index, model in enumerate(models):

        try:

            prediction = model.predict(
                frame
            )

            prediction["model_id"] = index

            results.append(
                prediction
            )


            print(
                f"Model {index}:",
                prediction
            )


        except Exception as error:

            print(
                f"Model {index} gagal:",
                error
            )


    if len(results) == 0:

        return None


    best_result = max(
        results,
        key=lambda x:
        x.get(
            "confidence",
            0
        )
    )


    return best_result



def main():

    config.validate_config()


    camera = CameraService()

    mqtt_service = MQTTService()


    models = load_models()


    last_prediction_time = 0.0

    prediction_interval = 2.0


    latest_prediction = None



    try:

        print("=" * 60)
        print(
            "RASPBERRY PI AI MQTT SYSTEM"
        )
        print("=" * 60)


        print(
            f"Broker : "
            f"{config.MQTT_BROKER_HOST}:"
            f"{config.MQTT_BROKER_PORT}"
        )


        print("=" * 60)


        camera.open()


        mqtt_service.start()


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

                    prediction = (
                        predict_with_all_models(
                            models,
                            frame
                        )
                    )


                    if prediction:


                        latest_prediction = (
                            prediction
                        )


                        last_prediction_time = (
                            current_time
                        )


                        print(
                            "HASIL AKHIR:",
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
                                    "Command gagal"
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
