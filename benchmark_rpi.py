from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from realtime_ensemble import (
    EnsembleClassifier,
    LABELS,
    MODEL_FILES,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE_DIR / "benchmark_images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "benchmark_results"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

GROUND_TRUTH_ALIASES = {
    "cardboard": "CardBoard",
    "card_board": "CardBoard",
    "card board": "CardBoard",
    "kardus": "CardBoard",

    "food waste": "Food Waste",
    "food_waste": "Food Waste",
    "foodwaste": "Food Waste",
    "organic": "Food Waste",
    "organik": "Food Waste",
    "sisa makanan": "Food Waste",

    "metal": "Metal",
    "logam": "Metal",
    "kaleng": "Metal",

    "paper": "Paper",
    "kertas": "Paper",

    "plastic": "Plastic",
    "plastik": "Plastic",
}


def normalize_ground_truth(
    folder_name: str,
) -> str:
    normalized = (
        folder_name
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized in GROUND_TRUTH_ALIASES:
        return GROUND_TRUTH_ALIASES[normalized]

    for label in LABELS:
        if normalized == label.lower():
            return label

    return ""


def get_cpu_temperature() -> float | None:
    thermal_path = Path(
        "/sys/class/thermal/thermal_zone0/temp"
    )

    try:
        raw_value = thermal_path.read_text(
            encoding="utf-8"
        ).strip()

        return round(
            float(raw_value) / 1000.0,
            2,
        )

    except (FileNotFoundError, ValueError):
        return None


def get_system_metrics() -> dict[str, float | None]:
    process = psutil.Process()

    process_memory_mb = (
        process.memory_info().rss
        / 1024
        / 1024
    )

    return {
        "cpu_percent": round(
            psutil.cpu_percent(interval=None),
            2,
        ),
        "ram_percent": round(
            psutil.virtual_memory().percent,
            2,
        ),
        "process_ram_mb": round(
            process_memory_mb,
            2,
        ),
        "cpu_temperature_c": (
            get_cpu_temperature()
        ),
    }


def collect_images(
    image_dir: Path,
) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: "
            f"{image_dir}"
        )

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"Tidak ada gambar uji di: "
            f"{image_dir}"
        )

    return image_paths


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float32,
            ),
            percentile,
        )
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_confusion_matrix(
    rows: list[dict[str, Any]],
    prediction_column: str,
) -> list[list[int]]:
    matrix = np.zeros(
        (
            len(LABELS),
            len(LABELS),
        ),
        dtype=np.int32,
    )

    label_to_index = {
        label: index
        for index, label in enumerate(
            LABELS
        )
    }

    for row in rows:
        ground_truth = row["ground_truth"]
        prediction = row[prediction_column]

        if (
            ground_truth not in label_to_index
            or prediction not in label_to_index
        ):
            continue

        actual_index = label_to_index[
            ground_truth
        ]

        predicted_index = label_to_index[
            prediction
        ]

        matrix[
            actual_index,
            predicted_index,
        ] += 1

    return matrix.tolist()


def calculate_accuracy(
    rows: list[dict[str, Any]],
    prediction_column: str,
) -> float | None:
    valid_rows = [
        row
        for row in rows
        if row["ground_truth"] in LABELS
    ]

    if not valid_rows:
        return None

    correct = sum(
        row[prediction_column]
        == row["ground_truth"]
        for row in valid_rows
    )

    return correct / len(valid_rows)


def summarize_configuration(
    name: str,
    rows: list[dict[str, Any]],
    prediction_column: str,
    confidence_column: str | None,
    latency_column: str,
) -> dict[str, Any]:
    latency_values = [
        float(row[latency_column])
        for row in rows
    ]

    confidence_values = []

    if confidence_column is not None:
        confidence_values = [
            float(row[confidence_column])
            for row in rows
        ]

    average_latency = statistics.mean(
        latency_values
    )

    accuracy = calculate_accuracy(
        rows,
        prediction_column,
    )

    return {
        "configuration": name,
        "number_of_runs": len(rows),

        "accuracy": (
            round(accuracy, 6)
            if accuracy is not None
            else ""
        ),

        "average_confidence": (
            round(
                statistics.mean(
                    confidence_values
                ),
                6,
            )
            if confidence_values
            else ""
        ),

        "average_latency_ms": round(
            average_latency,
            3,
        ),

        "median_latency_ms": round(
            statistics.median(
                latency_values
            ),
            3,
        ),

        "minimum_latency_ms": round(
            min(latency_values),
            3,
        ),

        "maximum_latency_ms": round(
            max(latency_values),
            3,
        ),

        "p95_latency_ms": round(
            calculate_percentile(
                latency_values,
                95,
            ),
            3,
        ),

        "average_fps": round(
            (
                1000.0 / average_latency
                if average_latency > 0
                else 0.0
            ),
            3,
        ),
    }


def model_size_mb(
    model_dir: Path,
    filename: str,
) -> float:
    path = model_dir / filename

    if not path.exists():
        return 0.0

    return round(
        path.stat().st_size
        / 1024
        / 1024,
        3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark empat CNN dan ensemble "
            "pada Raspberry Pi."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=(
            "Folder gambar benchmark. "
            "Ground truth dibaca dari nama folder."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Jumlah warm-up sebelum pengujian.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Jumlah pengulangan untuk "
            "setiap gambar."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help=(
            "Jumlah thread LiteRT "
            "untuk setiap model."
        ),
    )

    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError(
            "Warm-up tidak boleh negatif."
        )

    if args.repeats < 1:
        raise ValueError(
            "Repeats minimal 1."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        args.output_dir
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("BENCHMARK RASPBERRY PI")
    print("=" * 65)
    print("Folder gambar :", args.images)
    print("Output        :", output_dir)
    print("Warm-up       :", args.warmup)
    print("Repeats       :", args.repeats)
    print("Thread/model  :", args.threads)
    print()

    image_paths = collect_images(
        args.images
    )

    print(
        f"Jumlah gambar ditemukan: "
        f"{len(image_paths)}"
    )

    print()
    print("Memuat empat model TFLite...")

    classifier = EnsembleClassifier(
        num_threads=args.threads
    )

    print()
    print("Semua model berhasil dimuat.")

    first_image = cv2.imread(
        str(image_paths[0])
    )

    if first_image is None:
        raise RuntimeError(
            f"Gagal membaca gambar pertama: "
            f"{image_paths[0]}"
        )

    print()
    print(
        f"Menjalankan warm-up "
        f"{args.warmup} kali..."
    )

    for warmup_index in range(
        args.warmup
    ):
        classifier.predict(
            first_image
        )

        print(
            f"Warm-up "
            f"{warmup_index + 1}/"
            f"{args.warmup}",
            end="\r",
        )

    print()
    print("Warm-up selesai.")

    # Menstabilkan pembacaan awal CPU.
    psutil.cpu_percent(interval=None)

    run_rows: list[dict[str, Any]] = []

    total_runs = (
        len(image_paths)
        * args.repeats
    )

    current_run = 0

    print()
    print("Memulai benchmark...")

    for image_path in image_paths:
        ground_truth = normalize_ground_truth(
            image_path.parent.name
        )

        for repeat_index in range(
            1,
            args.repeats + 1,
        ):
            current_run += 1

            image_read_started = (
                time.perf_counter()
            )

            frame = cv2.imread(
                str(image_path)
            )

            image_read_ms = (
                time.perf_counter()
                - image_read_started
            ) * 1000.0

            if frame is None:
                print(
                    f"Gagal membaca: "
                    f"{image_path}"
                )
                continue

            system_before = (
                get_system_metrics()
            )

            end_to_end_started = (
                time.perf_counter()
            )

            prediction = classifier.predict(
                frame
            )

            end_to_end_ms = (
                time.perf_counter()
                - end_to_end_started
            ) * 1000.0

            system_after = (
                get_system_metrics()
            )

            model_results = {
                result["model"]: result
                for result in prediction[
                    "models"
                ]
            }

            efficientnet = model_results[
                "EfficientNetB0"
            ]

            resnet50 = model_results[
                "ResNet50"
            ]

            densenet121 = model_results[
                "DenseNet121"
            ]

            shufflenet = model_results[
                "ShuffleNetV2"
            ]

            hard_voting = prediction[
                "hard_voting"
            ]

            soft_voting = prediction[
                "soft_voting"
            ]

            weighted_voting = prediction[
                "weighted_soft_voting"
            ]

            run_id = (
                f"RUN_{current_run:05d}"
            )

            row = {
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                ),
                "run_id": run_id,
                "repeat": repeat_index,
                "image_path": str(
                    image_path
                ),
                "image_name": (
                    image_path.name
                ),
                "ground_truth": ground_truth,

                "image_read_ms": round(
                    image_read_ms,
                    3,
                ),

                "efficientnet_label": (
                    efficientnet["label"]
                ),
                "efficientnet_confidence": (
                    efficientnet["confidence"]
                ),
                "efficientnet_preprocessing_ms": (
                    efficientnet[
                        "preprocessing_ms"
                    ]
                ),
                "efficientnet_inference_ms": (
                    efficientnet[
                        "inference_ms"
                    ]
                ),
                "efficientnet_total_ms": round(
                    efficientnet[
                        "preprocessing_ms"
                    ]
                    + efficientnet[
                        "inference_ms"
                    ],
                    3,
                ),
                "efficientnet_correct": (
                    int(
                        efficientnet["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "resnet50_label": (
                    resnet50["label"]
                ),
                "resnet50_confidence": (
                    resnet50["confidence"]
                ),
                "resnet50_preprocessing_ms": (
                    resnet50[
                        "preprocessing_ms"
                    ]
                ),
                "resnet50_inference_ms": (
                    resnet50[
                        "inference_ms"
                    ]
                ),
                "resnet50_total_ms": round(
                    resnet50[
                        "preprocessing_ms"
                    ]
                    + resnet50[
                        "inference_ms"
                    ],
                    3,
                ),
                "resnet50_correct": (
                    int(
                        resnet50["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "densenet121_label": (
                    densenet121["label"]
                ),
                "densenet121_confidence": (
                    densenet121[
                        "confidence"
                    ]
                ),
                "densenet121_preprocessing_ms": (
                    densenet121[
                        "preprocessing_ms"
                    ]
                ),
                "densenet121_inference_ms": (
                    densenet121[
                        "inference_ms"
                    ]
                ),
                "densenet121_total_ms": round(
                    densenet121[
                        "preprocessing_ms"
                    ]
                    + densenet121[
                        "inference_ms"
                    ],
                    3,
                ),
                "densenet121_correct": (
                    int(
                        densenet121["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "shufflenet_label": (
                    shufflenet["label"]
                ),
                "shufflenet_confidence": (
                    shufflenet[
                        "confidence"
                    ]
                ),
                "shufflenet_preprocessing_ms": (
                    shufflenet[
                        "preprocessing_ms"
                    ]
                ),
                "shufflenet_inference_ms": (
                    shufflenet[
                        "inference_ms"
                    ]
                ),
                "shufflenet_total_ms": round(
                    shufflenet[
                        "preprocessing_ms"
                    ]
                    + shufflenet[
                        "inference_ms"
                    ],
                    3,
                ),
                "shufflenet_correct": (
                    int(
                        shufflenet["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "hard_voting_label": (
                    hard_voting["label"]
                ),
                "hard_voting_correct": (
                    int(
                        hard_voting["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "soft_voting_label": (
                    soft_voting["label"]
                ),
                "soft_voting_confidence": (
                    soft_voting[
                        "confidence"
                    ]
                ),
                "soft_voting_correct": (
                    int(
                        soft_voting["label"]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "weighted_voting_label": (
                    weighted_voting["label"]
                ),
                "weighted_voting_confidence": (
                    weighted_voting[
                        "confidence"
                    ]
                ),
                "weighted_voting_correct": (
                    int(
                        weighted_voting[
                            "label"
                        ]
                        == ground_truth
                    )
                    if ground_truth
                    else ""
                ),

                "agreement_count": (
                    prediction[
                        "agreement_count"
                    ]
                ),
                "number_of_models": (
                    prediction[
                        "number_of_models"
                    ]
                ),

                "ensemble_ms": (
                    prediction[
                        "ensemble_ms"
                    ]
                ),
                "ai_total_latency_ms": (
                    prediction[
                        "total_latency_ms"
                    ]
                ),
                "ai_fps": (
                    prediction["fps"]
                ),

                "end_to_end_ms": round(
                    end_to_end_ms
                    + image_read_ms,
                    3,
                ),
                "end_to_end_fps": round(
                    (
                        1000.0
                        / (
                            end_to_end_ms
                            + image_read_ms
                        )
                        if (
                            end_to_end_ms
                            + image_read_ms
                        ) > 0
                        else 0.0
                    ),
                    3,
                ),

                "cpu_percent_before": (
                    system_before[
                        "cpu_percent"
                    ]
                ),
                "cpu_percent_after": (
                    system_after[
                        "cpu_percent"
                    ]
                ),

                "ram_percent_before": (
                    system_before[
                        "ram_percent"
                    ]
                ),
                "ram_percent_after": (
                    system_after[
                        "ram_percent"
                    ]
                ),

                "process_ram_mb_before": (
                    system_before[
                        "process_ram_mb"
                    ]
                ),
                "process_ram_mb_after": (
                    system_after[
                        "process_ram_mb"
                    ]
                ),

                "cpu_temperature_before_c": (
                    system_before[
                        "cpu_temperature_c"
                    ]
                ),
                "cpu_temperature_after_c": (
                    system_after[
                        "cpu_temperature_c"
                    ]
                ),
            }

            run_rows.append(row)

            print(
                f"[{current_run}/{total_runs}] "
                f"{image_path.name} | "
                f"GT={ground_truth or 'NA'} | "
                f"Weighted="
                f"{weighted_voting['label']} | "
                f"{prediction['total_latency_ms']}"
                f" ms"
            )

    if not run_rows:
        raise RuntimeError(
            "Tidak ada hasil benchmark "
            "yang berhasil dicatat."
        )

    runs_path = (
        output_dir
        / "benchmark_runs.csv"
    )

    write_csv(
        runs_path,
        run_rows,
    )

    summary_rows = [
        summarize_configuration(
            name="EfficientNetB0",
            rows=run_rows,
            prediction_column=(
                "efficientnet_label"
            ),
            confidence_column=(
                "efficientnet_confidence"
            ),
            latency_column=(
                "efficientnet_total_ms"
            ),
        ),

        summarize_configuration(
            name="ResNet50",
            rows=run_rows,
            prediction_column=(
                "resnet50_label"
            ),
            confidence_column=(
                "resnet50_confidence"
            ),
            latency_column=(
                "resnet50_total_ms"
            ),
        ),

        summarize_configuration(
            name="DenseNet121",
            rows=run_rows,
            prediction_column=(
                "densenet121_label"
            ),
            confidence_column=(
                "densenet121_confidence"
            ),
            latency_column=(
                "densenet121_total_ms"
            ),
        ),

        summarize_configuration(
            name="ShuffleNetV2",
            rows=run_rows,
            prediction_column=(
                "shufflenet_label"
            ),
            confidence_column=(
                "shufflenet_confidence"
            ),
            latency_column=(
                "shufflenet_total_ms"
            ),
        ),

        summarize_configuration(
            name="Hard Voting",
            rows=run_rows,
            prediction_column=(
                "hard_voting_label"
            ),
            confidence_column=None,
            latency_column=(
                "ai_total_latency_ms"
            ),
        ),

        summarize_configuration(
            name="Soft Voting",
            rows=run_rows,
            prediction_column=(
                "soft_voting_label"
            ),
            confidence_column=(
                "soft_voting_confidence"
            ),
            latency_column=(
                "ai_total_latency_ms"
            ),
        ),

        summarize_configuration(
            name="Weighted Soft Voting",
            rows=run_rows,
            prediction_column=(
                "weighted_voting_label"
            ),
            confidence_column=(
                "weighted_voting_confidence"
            ),
            latency_column=(
                "ai_total_latency_ms"
            ),
        ),
    ]

    summary_path = (
        output_dir
        / "benchmark_summary.csv"
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    confusion_matrices = {
        "labels": LABELS,

        "EfficientNetB0": (
            build_confusion_matrix(
                run_rows,
                "efficientnet_label",
            )
        ),

        "ResNet50": (
            build_confusion_matrix(
                run_rows,
                "resnet50_label",
            )
        ),

        "DenseNet121": (
            build_confusion_matrix(
                run_rows,
                "densenet121_label",
            )
        ),

        "ShuffleNetV2": (
            build_confusion_matrix(
                run_rows,
                "shufflenet_label",
            )
        ),

        "Hard Voting": (
            build_confusion_matrix(
                run_rows,
                "hard_voting_label",
            )
        ),

        "Soft Voting": (
            build_confusion_matrix(
                run_rows,
                "soft_voting_label",
            )
        ),

        "Weighted Soft Voting": (
            build_confusion_matrix(
                run_rows,
                "weighted_voting_label",
            )
        ),
    }

    confusion_path = (
        output_dir
        / "confusion_matrices.json"
    )

    confusion_path.write_text(
        json.dumps(
            confusion_matrices,
            indent=2,
        ),
        encoding="utf-8",
    )

    model_information = {
        "raspberry_pi": {
            "cpu_count_logical": (
                psutil.cpu_count(
                    logical=True
                )
            ),
            "cpu_count_physical": (
                psutil.cpu_count(
                    logical=False
                )
            ),
            "total_ram_mb": round(
                psutil.virtual_memory().total
                / 1024
                / 1024,
                2,
            ),
        },

        "benchmark": {
            "number_of_images": (
                len(image_paths)
            ),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "threads_per_model": (
                args.threads
            ),
            "total_successful_runs": (
                len(run_rows)
            ),
        },

        "model_sizes_mb": {
            model_name: model_size_mb(
                classifier.model_dir,
                filename,
            )
            for model_name, filename
            in MODEL_FILES.items()
        },

        "model_runtime_information": (
            classifier
            .get_model_information()
        ),
    }

    information_path = (
        output_dir
        / "benchmark_information.json"
    )

    information_path.write_text(
        json.dumps(
            model_information,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 65)
    print("BENCHMARK SELESAI")
    print("=" * 65)
    print("Data setiap run :")
    print(runs_path)
    print()
    print("Ringkasan       :")
    print(summary_path)
    print()
    print("Confusion matrix:")
    print(confusion_path)
    print()
    print("Informasi sistem:")
    print(information_path)


if __name__ == "__main__":
    main()
