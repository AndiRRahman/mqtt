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
        "Unknown"
    )


    confidence = safe_float(
        result.get(
            "confidence",
            0.0
        )
    )


    model_name = result.get(
        "model",
        "-"
    )


    return (
        f"{label} "
        f"({confidence:.1%}) "
        f"[{model_name}]"
    )



def load_all_models():

    model_directory = "model"

    models = []


    if not os.path.exists(
        model_directory
    ):

        raise FileNotFoundError(
            "Folder model tidak ditemukan"
        )



    for filename in sorted(
        os.listdir(model_directory)
    ):


        if filename.endswith(
            ".tflite"
        ):


            model_path = os.path.join(
                model_directory,
                filename
            )


            print(
                "Memuat model:",
                filename
            )


            model = InferenceService(
                model_path=model_path,
                labels_path="labels.txt"
            )


            models.append(
                {
                    "name": filename,
                    "model": model
                }
            )



    if len(models) == 0:

        raise RuntimeError(
            "Tidak ada file .tflite"
        )


    print(
        f"Total model aktif: {len(models)}"
    )


    return models



def predict_all_models(
    models,
    frame
):

    results = []


    for item in models:


        name = item["name"]

        model = item["model"]


        try:

            result = model.predict(
                frame
            )


            result["model"] = name


            results.append(
                result
            )


            print(
                name,
                ":",
                result
            )



        except Exception as error:

            print(
                f"{name} gagal:",
                error
            )



    if not results:

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



    models = load_all_models()



    prediction_interval = 2.0

    last_prediction_time = 0.0


    latest_prediction = None



    try:


        print("=" * 60)

        print(
            "RASPBERRY PI MULTI MODEL AI MQTT SYSTEM"
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
                    "Frame kamera gagal"
                )


                time.sleep(
                    0.2
                )

                continue




            current_time = time.monotonic()



            status = (
                "Menunggu prediksi"
            )



            if (
                current_time -
                last_prediction_time
                >= prediction_interval
            ):



                try:



                    prediction = (
                        predict_all_models(
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
                            "HASIL TERBAIK:",
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

                                status = (
                                    "Command ESP32 terkirim"
                                )

                            else:

                                status = (
                                    "Command gagal"
                                )



                except Exception as error:


                    print(
                        "Inference error:",
                        error
                    )


                    status = (
                        "Inference error"
                    )



            ai_status = (
                format_classification_result(
                    latest_prediction
                )
            )



            preview_status = (
                f"{status} | "
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
