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
        return "Belum ada hasil"


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


    model = result.get(
        "model",
        "-"
    )


    return (
        f"{label} "
        f"({confidence:.1%}) "
        f"{model}"
    )



# ==================================================
# LOAD MODEL
# ==================================================

def load_models():


    model_dir = "Paradigma B No Mix"


    model_paths = {}



    if not os.path.exists(model_dir):

        raise FileNotFoundError(
            "Folder model tidak ditemukan"
        )



    for filename in sorted(
        os.listdir(model_dir)
    ):


        if filename.endswith(
            ".tflite"
        ):


            name = filename.replace(
                ".tflite",
                ""
            )


            path = os.path.join(
                model_dir,
                filename
            )


            model_paths[name] = path



            print(
                "Model ditemukan:",
                filename
            )



    if len(model_paths) == 0:

        raise RuntimeError(
            "Tidak ada model TFLite"
        )



    print(
        f"{len(model_paths)} model aktif"
    )



    models = InferenceService(

        model_paths=model_paths,

        labels_path="labels.txt"

    )


    return models



# ==================================================
# PREDICT
# ==================================================

def predict_all_models(
    models,
    frame
):

    try:


        prediction = models.predict_all(
            frame
        )


        print(
            "HASIL MODEL:",
            prediction
        )


        return prediction



    except Exception as error:


        print(
            "Inference error:",
            error
        )


        return None



# ==================================================
# MAIN
# ==================================================

def main():


    config.validate_config()



    camera = CameraService()


    mqtt_service = MQTTService()



    models = load_models()



    prediction_interval = 2.0


    last_prediction_time = 0.0



    latest_prediction = None



    try:


        print("=" * 60)

        print(
            "RASPBERRY PI AI SORTING SYSTEM"
        )

        print("=" * 60)



        print(
            f"MQTT : "
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

                continue



            current_time = time.monotonic()



            status = (
                "Scanning"
            )



            if (
                current_time -
                last_prediction_time
                >= prediction_interval
            ):



                last_prediction_time = current_time



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


                    print(
                        "HASIL AKHIR:",
                        prediction
                    )


                    status = (
                        "Sampah terdeteksi"
                    )



                    if mqtt_service.is_connected():


                        mqtt_service.publish_prediction_command(
                            prediction
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
