from __future__ import annotations

import os
import time
from typing import Any

import config

from camera_service import CameraService
from mqtt_service import MQTTService
from inference_service import InferenceService
from presence_detector import PresenceDetector



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


    return (
        f"{label} "
        f"({confidence:.1%})"
    )



# ==================================================
# LOAD ALL MODEL FOR SOFT VOTING
# ==================================================

def load_ai_model():

    model_dir = "Paradigma B No Mix"


    if not os.path.exists(model_dir):

        raise FileNotFoundError(
            "Folder model tidak ditemukan"
        )



    model_paths = {}



    for filename in sorted(
        os.listdir(model_dir)
    ):


        if filename.endswith(
            ".tflite"
        ):


            model_name = filename.replace(
                ".tflite",
                ""
            )


            model_paths[
                model_name
            ] = os.path.join(
                model_dir,
                filename
            )



    if len(model_paths) == 0:

        raise RuntimeError(
            "Model TFLite tidak ditemukan"
        )



    print(
        "Model aktif:"
    )


    for model in model_paths:

        print(
            "-",
            model
        )



    ai = InferenceService(

        model_paths=model_paths,

        labels_path="labels.txt"

    )


    return ai





# ==================================================
# MAIN
# ==================================================

def main():


    config.validate_config()



    camera = CameraService()



    mqtt_service = MQTTService()



    presence_detector = PresenceDetector(

        threshold=20000

    )



    ai = load_ai_model()



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



            status = "Scanning"



            # ======================================
            # PREDICTION INTERVAL
            # ======================================


            if (

                current_time -
                last_prediction_time

                >=

                prediction_interval

            ):



                last_prediction_time = current_time



                # ==================================
                # CHECK OBJECT EXISTENCE
                # ==================================


                # object_detected = (

                #     presence_detector.detect(
                #         frame
                #     )

                # )
                object_detected = True


                if not object_detected:


                    latest_prediction = {


                        "label":
                        "NO_OBJECT",


                        "confidence":
                        1.0,


                        "class_id":
                        -1


                    }



                    print(
                        "Tidak ada sampah"
                    )



                    status = (
                        "Tidak ada sampah"
                    )



                else:



                    print(
                        "Objek terdeteksi"
                    )



                    try:


                        prediction = ai.predict(
                            frame
                        )



                        latest_prediction = (
                            prediction
                        )



                        print(
                            "\nHASIL SOFT VOTING:"
                        )


                        print(
                            prediction
                        )



                        status = (
                            "Sampah terdeteksi"
                        )



                    except Exception as error:


                        print(
                            "Inference error:",
                            error
                        )



                        continue





                # ==================================
                # SEND TO ESP32
                # ==================================


                if (

                    mqtt_service.is_connected()

                    and

                    latest_prediction

                ):


                    mqtt_service.publish_prediction_command(

                        latest_prediction

                    )





            # ======================================
            # DISPLAY
            # ======================================


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
