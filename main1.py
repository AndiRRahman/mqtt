from __future__ import annotations

import time
from datetime import datetime

import cv2

from performance_logger import PerformanceLogger
from realtime_ensemble import EnsembleClassifier


CAMERA_INDEX = 0
WINDOW_NAME = "Real-Time Multi-CNN Ensemble"

# Kosongkan saat penggunaan biasa.
# Isi ketika melakukan pengujian terkontrol.
GROUND_TRUTH = ""
TEST_CONDITION = "real_time_webcam"


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.52,
) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    print("Memuat empat model TFLite...")

    classifier = EnsembleClassifier(
        num_threads=2,
    )

    logger = PerformanceLogger(
        filename="raspberry_pi_realtime.csv",
    )

    print("Membuka USB webcam...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "USB webcam tidak berhasil dibuka."
        )

    # Memberi waktu webcam melakukan inisialisasi.
    time.sleep(2)

    print("Kamera aktif.")
    print("Tekan q untuk keluar.")
    print(f"Log disimpan di: {logger.get_log_path()}")

    run_number = 0

    try:
        while True:
            camera_started = time.perf_counter()

            success, frame = camera.read()

            camera_read_ms = (
                time.perf_counter() - camera_started
            ) * 1000.0

            if not success or frame is None:
                print("Gagal membaca frame webcam.")
                time.sleep(0.2)
                continue

            run_number += 1

            run_id = (
                f"RT_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{run_number:06d}"
            )

            prediction = classifier.predict(
                frame
            )

            logged_row = logger.log(
                prediction_result=prediction,
                camera_read_ms=camera_read_ms,
                run_id=run_id,
                ground_truth=GROUND_TRUTH,
                test_condition=TEST_CONDITION,
            )

            y_position = 25

            for model_result in prediction["models"]:
                model_text = (
                    f"{model_result['model']}: "
                    f"{model_result['label']} "
                    f"{model_result['confidence']:.1%} "
                    f"({model_result['inference_ms']:.0f} ms)"
                )

                draw_text(
                    frame,
                    model_text,
                    (10, y_position),
                    (0, 255, 0),
                )

                y_position += 25

            y_position += 8

            hard_result = prediction["hard_voting"]

            draw_text(
                frame,
                f"Hard Voting: {hard_result['label']}",
                (10, y_position),
                (255, 255, 0),
            )

            y_position += 25

            soft_result = prediction["soft_voting"]

            draw_text(
                frame,
                (
                    f"Soft Voting: "
                    f"{soft_result['label']} "
                    f"{soft_result['confidence']:.1%}"
                ),
                (10, y_position),
                (0, 255, 255),
            )

            y_position += 25

            weighted_result = prediction[
                "weighted_soft_voting"
            ]

            draw_text(
                frame,
                (
                    f"Weighted Voting: "
                    f"{weighted_result['label']} "
                    f"{weighted_result['confidence']:.1%}"
                ),
                (10, y_position),
                (255, 0, 255),
            )

            y_position += 25

            draw_text(
                frame,
                (
                    f"Agreement: "
                    f"{prediction['agreement_count']}/"
                    f"{prediction['number_of_models']}"
                ),
                (10, y_position),
                (255, 255, 255),
            )

            frame_height = frame.shape[0]

            draw_text(
                frame,
                (
                    f"Camera: {camera_read_ms:.1f} ms | "
                    f"AI: {prediction['total_latency_ms']:.1f} ms"
                ),
                (10, frame_height - 65),
                (255, 255, 255),
            )

            draw_text(
                frame,
                (
                    f"FPS AI: {prediction['fps']:.2f} | "
                    f"CPU: {logged_row['cpu_percent']:.1f}%"
                ),
                (10, frame_height - 40),
                (255, 255, 255),
            )

            temperature = logged_row[
                "cpu_temperature_c"
            ]

            temperature_text = (
                f"{temperature:.1f} C"
                if temperature is not None
                else "N/A"
            )

            draw_text(
                frame,
                (
                    f"RAM proses: "
                    f"{logged_row['process_ram_mb']:.1f} MB | "
                    f"Suhu: {temperature_text}"
                ),
                (10, frame_height - 15),
                (255, 255, 255),
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nProgram dihentikan dari terminal.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("Webcam ditutup.")
        print(f"Log tersimpan di: {logger.get_log_path()}")


if __name__ == "__main__":
    main()
