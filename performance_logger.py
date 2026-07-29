from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class PerformanceLogger:
    def __init__(
        self,
        filename: str = "raspberry_pi_performance.csv",
    ) -> None:
        self.log_path = LOG_DIR / filename
        self.process = psutil.Process(os.getpid())

        self.fieldnames = [
            "timestamp",
            "run_id",
            "ground_truth",
            "test_condition",

            "camera_read_ms",

            "efficientnet_label",
            "efficientnet_confidence",
            "efficientnet_preprocessing_ms",
            "efficientnet_inference_ms",

            "resnet50_label",
            "resnet50_confidence",
            "resnet50_preprocessing_ms",
            "resnet50_inference_ms",

            "densenet121_label",
            "densenet121_confidence",
            "densenet121_preprocessing_ms",
            "densenet121_inference_ms",

            "shufflenet_label",
            "shufflenet_confidence",
            "shufflenet_preprocessing_ms",
            "shufflenet_inference_ms",

            "hard_voting_label",

            "soft_voting_label",
            "soft_voting_confidence",

            "weighted_voting_label",
            "weighted_voting_confidence",

            "agreement_count",
            "number_of_models",

            "ensemble_ms",
            "total_latency_ms",
            "fps",

            "cpu_percent",
            "ram_percent",
            "process_ram_mb",
            "cpu_temperature_c",
        ]

        self._create_file_if_needed()

    def _create_file_if_needed(self) -> None:
        if self.log_path.exists():
            return

        with self.log_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )
            writer.writeheader()

    @staticmethod
    def get_cpu_temperature() -> float | None:
        thermal_path = Path(
            "/sys/class/thermal/thermal_zone0/temp"
        )

        try:
            raw_temperature = thermal_path.read_text(
                encoding="utf-8"
            ).strip()

            return round(
                float(raw_temperature) / 1000.0,
                2,
            )

        except (FileNotFoundError, ValueError):
            return None

    def get_system_metrics(self) -> dict[str, float | None]:
        process_memory = self.process.memory_info().rss

        process_ram_mb = (
            process_memory
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
                process_ram_mb,
                2,
            ),
            "cpu_temperature_c": (
                self.get_cpu_temperature()
            ),
        }

    @staticmethod
    def find_model_result(
        results: list[dict[str, Any]],
        model_name: str,
    ) -> dict[str, Any]:
        for result in results:
            if result["model"] == model_name:
                return result

        raise KeyError(
            f"Hasil model tidak ditemukan: {model_name}"
        )

    def log(
        self,
        prediction_result: dict[str, Any],
        camera_read_ms: float,
        run_id: str = "",
        ground_truth: str = "",
        test_condition: str = "",
    ) -> dict[str, Any]:

        model_results = prediction_result["models"]

        efficientnet = self.find_model_result(
            model_results,
            "EfficientNetB0",
        )

        resnet50 = self.find_model_result(
            model_results,
            "ResNet50",
        )

        densenet121 = self.find_model_result(
            model_results,
            "DenseNet121",
        )

        shufflenet = self.find_model_result(
            model_results,
            "ShuffleNetV2",
        )

        system_metrics = self.get_system_metrics()

        row = {
            "timestamp": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "run_id": run_id,
            "ground_truth": ground_truth,
            "test_condition": test_condition,

            "camera_read_ms": round(
                camera_read_ms,
                2,
            ),

            "efficientnet_label": (
                efficientnet["label"]
            ),
            "efficientnet_confidence": (
                efficientnet["confidence"]
            ),
            "efficientnet_preprocessing_ms": (
                efficientnet["preprocessing_ms"]
            ),
            "efficientnet_inference_ms": (
                efficientnet["inference_ms"]
            ),

            "resnet50_label": (
                resnet50["label"]
            ),
            "resnet50_confidence": (
                resnet50["confidence"]
            ),
            "resnet50_preprocessing_ms": (
                resnet50["preprocessing_ms"]
            ),
            "resnet50_inference_ms": (
                resnet50["inference_ms"]
            ),

            "densenet121_label": (
                densenet121["label"]
            ),
            "densenet121_confidence": (
                densenet121["confidence"]
            ),
            "densenet121_preprocessing_ms": (
                densenet121["preprocessing_ms"]
            ),
            "densenet121_inference_ms": (
                densenet121["inference_ms"]
            ),

            "shufflenet_label": (
                shufflenet["label"]
            ),
            "shufflenet_confidence": (
                shufflenet["confidence"]
            ),
            "shufflenet_preprocessing_ms": (
                shufflenet["preprocessing_ms"]
            ),
            "shufflenet_inference_ms": (
                shufflenet["inference_ms"]
            ),

            "hard_voting_label": (
                prediction_result[
                    "hard_voting"
                ]["label"]
            ),

            "soft_voting_label": (
                prediction_result[
                    "soft_voting"
                ]["label"]
            ),
            "soft_voting_confidence": (
                prediction_result[
                    "soft_voting"
                ]["confidence"]
            ),

            "weighted_voting_label": (
                prediction_result[
                    "weighted_soft_voting"
                ]["label"]
            ),
            "weighted_voting_confidence": (
                prediction_result[
                    "weighted_soft_voting"
                ]["confidence"]
            ),

            "agreement_count": (
                prediction_result[
                    "agreement_count"
                ]
            ),
            "number_of_models": (
                prediction_result[
                    "number_of_models"
                ]
            ),

            "ensemble_ms": (
                prediction_result[
                    "ensemble_ms"
                ]
            ),
            "total_latency_ms": (
                prediction_result[
                    "total_latency_ms"
                ]
            ),
            "fps": prediction_result["fps"],

            **system_metrics,
        }

        with self.log_path.open(
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )
            writer.writerow(row)

        return row

    def get_log_path(self) -> Path:
        return self.log_path
