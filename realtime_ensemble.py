from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Paradigma B No Mix"

LABELS = [
    "CardBoard",
    "Food Waste",
    "Metal",
    "Paper",
    "Plastic",
]

MODEL_FILES = {
    "EfficientNetB0": "efficientnet_b0.tflite",
    "ResNet50": "resnet50.tflite",
    "DenseNet121": "densenet121.tflite",
    "ShuffleNetV2": "shufflenet_v2_x1_0.tflite",
}

# Bobot awal berdasarkan performa validasi model.
RAW_WEIGHTS = {
    "EfficientNetB0": 0.9277,
    "ResNet50": 0.9546,
    "DenseNet121": 0.9518,
    "ShuffleNetV2": 0.8709,
}

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class EnsembleClassifier:
    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        num_threads: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.models: dict[str, dict[str, Any]] = {}

        self.weights = self._normalize_weights(
            RAW_WEIGHTS
        )

        self._load_models()

    @staticmethod
    def _normalize_weights(
        raw_weights: dict[str, float],
    ) -> dict[str, float]:
        total = sum(raw_weights.values())

        if total <= 0:
            raise ValueError(
                "Jumlah bobot ensemble harus lebih dari nol."
            )

        return {
            name: value / total
            for name, value in raw_weights.items()
        }

    def _load_models(self) -> None:
        for model_name, filename in MODEL_FILES.items():
            model_path = self.model_dir / filename

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model tidak ditemukan: {model_path}"
                )

            interpreter = Interpreter(
                model_path=str(model_path),
                num_threads=self.num_threads,
            )

            interpreter.allocate_tensors()

            input_detail = (
                interpreter.get_input_details()[0]
            )
            output_detail = (
                interpreter.get_output_details()[0]
            )

            self.models[model_name] = {
                "interpreter": interpreter,
                "input_detail": input_detail,
                "output_detail": output_detail,
            }

            print(
                f"{model_name} berhasil dimuat"
            )

    @staticmethod
    def _softmax(
        logits: np.ndarray,
    ) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits = logits - np.max(logits)

        exponentials = np.exp(logits)
        denominator = np.sum(exponentials)

        if denominator == 0:
            raise FloatingPointError(
                "Softmax menghasilkan pembagi nol."
            )

        return exponentials / denominator

    @staticmethod
    def _quantize_input(
        image: np.ndarray,
        input_detail: dict[str, Any],
    ) -> np.ndarray:
        input_dtype = input_detail["dtype"]

        if np.issubdtype(
            input_dtype,
            np.floating,
        ):
            return image.astype(input_dtype)

        scale, zero_point = input_detail.get(
            "quantization",
            (0.0, 0),
        )

        if scale == 0:
            raise ValueError(
                "Skala kuantisasi input bernilai nol."
            )

        quantized = np.round(
            image / scale + zero_point
        )

        limits = np.iinfo(input_dtype)

        quantized = np.clip(
            quantized,
            limits.min,
            limits.max,
        )

        return quantized.astype(input_dtype)

    @staticmethod
    def _dequantize_output(
        output: np.ndarray,
        output_detail: dict[str, Any],
    ) -> np.ndarray:
        output_dtype = output_detail["dtype"]

        if np.issubdtype(
            output_dtype,
            np.floating,
        ):
            return output.astype(np.float32)

        scale, zero_point = output_detail.get(
            "quantization",
            (0.0, 0),
        )

        if scale == 0:
            raise ValueError(
                "Skala kuantisasi output bernilai nol."
            )

        return (
            output.astype(np.float32)
            - zero_point
        ) * scale

    def preprocess(
        self,
        frame: np.ndarray,
        input_detail: dict[str, Any],
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError(
                "Frame kamera kosong."
            )

        started = time.perf_counter()

        image = cv2.resize(
            frame,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        image = (
            image - IMAGE_MEAN
        ) / IMAGE_STD

        input_shape = tuple(
            int(value)
            for value in input_detail["shape"]
        )

        if input_shape == (1, 3, 224, 224):
            image = np.transpose(
                image,
                (2, 0, 1),
            )

        elif input_shape == (1, 224, 224, 3):
            pass

        else:
            raise ValueError(
                f"Shape input model tidak dikenali: "
                f"{input_shape}"
            )

        image = np.expand_dims(
            image,
            axis=0,
        )

        image = self._quantize_input(
            image,
            input_detail,
        )

        preprocessing_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return image, preprocessing_ms

    def _predict_one_model(
        self,
        model_name: str,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        model_data = self.models[model_name]

        interpreter = model_data["interpreter"]
        input_detail = model_data["input_detail"]
        output_detail = model_data["output_detail"]

        input_tensor, preprocessing_ms = (
            self.preprocess(
                frame,
                input_detail,
            )
        )

        started = time.perf_counter()

        interpreter.set_tensor(
            input_detail["index"],
            input_tensor,
        )

        interpreter.invoke()

        inference_ms = (
            time.perf_counter() - started
        ) * 1000.0

        raw_output = interpreter.get_tensor(
            output_detail["index"]
        )

        logits = self._dequantize_output(
            raw_output,
            output_detail,
        ).reshape(-1)

        if logits.size != len(LABELS):
            raise RuntimeError(
                f"Output {model_name} berjumlah "
                f"{logits.size}, seharusnya "
                f"{len(LABELS)}."
            )

        probabilities = self._softmax(
            logits
        )

        class_index = int(
            np.argmax(probabilities)
        )

        return {
            "model": model_name,
            "class_index": class_index,
            "label": LABELS[class_index],
            "confidence": float(
                probabilities[class_index]
            ),
            "probabilities": probabilities,
            "preprocessing_ms": round(
                preprocessing_ms,
                2,
            ),
            "inference_ms": round(
                inference_ms,
                2,
            ),
        }

    @staticmethod
    def _hard_voting(
        model_results: list[dict[str, Any]],
        soft_scores: np.ndarray,
    ) -> int:
        predicted_indices = [
            result["class_index"]
            for result in model_results
        ]

        vote_counts = np.bincount(
            predicted_indices,
            minlength=len(LABELS),
        )

        maximum_vote = int(
            np.max(vote_counts)
        )

        candidates = np.where(
            vote_counts == maximum_vote
        )[0]

        if len(candidates) == 1:
            return int(candidates[0])

        # Soft voting digunakan sebagai tie-breaker.
        candidate_scores = soft_scores[
            candidates
        ]

        return int(
            candidates[
                np.argmax(candidate_scores)
            ]
        )

    def predict(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()

        model_results = []

        for model_name in MODEL_FILES:
            result = self._predict_one_model(
                model_name,
                frame,
            )

            model_results.append(result)

        probability_matrix = np.stack(
            [
                result["probabilities"]
                for result in model_results
            ],
            axis=0,
        )

        ensemble_started = time.perf_counter()

        # Equal soft voting.
        soft_scores = np.mean(
            probability_matrix,
            axis=0,
        )

        soft_index = int(
            np.argmax(soft_scores)
        )

        # Weighted soft voting.
        model_weights = np.array(
            [
                self.weights[result["model"]]
                for result in model_results
            ],
            dtype=np.float32,
        )

        weighted_scores = np.average(
            probability_matrix,
            axis=0,
            weights=model_weights,
        )

        weighted_index = int(
            np.argmax(weighted_scores)
        )

        # Hard voting.
        hard_index = self._hard_voting(
            model_results,
            soft_scores,
        )

        ensemble_ms = (
            time.perf_counter()
            - ensemble_started
        ) * 1000.0

        total_ms = (
            time.perf_counter()
            - total_started
        ) * 1000.0

        total_fps = (
            1000.0 / total_ms
            if total_ms > 0
            else 0.0
        )

        agreement_count = sum(
            result["class_index"]
            == weighted_index
            for result in model_results
        )

        cleaned_model_results = []

        for result in model_results:
            cleaned_model_results.append(
                {
                    "model": result["model"],
                    "class_index": (
                        result["class_index"]
                    ),
                    "label": result["label"],
                    "confidence": round(
                        result["confidence"],
                        6,
                    ),
                    "preprocessing_ms": (
                        result["preprocessing_ms"]
                    ),
                    "inference_ms": (
                        result["inference_ms"]
                    ),
                    "ensemble_weight": round(
                        self.weights[
                            result["model"]
                        ],
                        6,
                    ),
                    "probabilities": [
                        round(float(value), 6)
                        for value in result[
                            "probabilities"
                        ]
                    ],
                }
            )

        return {
            "models": cleaned_model_results,

            "hard_voting": {
                "class_index": hard_index,
                "label": LABELS[hard_index],
            },

            "soft_voting": {
                "class_index": soft_index,
                "label": LABELS[soft_index],
                "confidence": round(
                    float(
                        soft_scores[soft_index]
                    ),
                    6,
                ),
                "probabilities": [
                    round(float(value), 6)
                    for value in soft_scores
                ],
            },

            "weighted_soft_voting": {
                "class_index": weighted_index,
                "label": LABELS[
                    weighted_index
                ],
                "confidence": round(
                    float(
                        weighted_scores[
                            weighted_index
                        ]
                    ),
                    6,
                ),
                "probabilities": [
                    round(float(value), 6)
                    for value in weighted_scores
                ],
            },

            "agreement_count": agreement_count,
            "number_of_models": len(
                model_results
            ),

            "ensemble_ms": round(
                ensemble_ms,
                2,
            ),

            "total_latency_ms": round(
                total_ms,
                2,
            ),

            "fps": round(
                total_fps,
                3,
            ),
        }

    def get_model_information(
        self,
    ) -> dict[str, Any]:
        information = {}

        for model_name, model_data in (
            self.models.items()
        ):
            input_detail = (
                model_data["input_detail"]
            )
            output_detail = (
                model_data["output_detail"]
            )

            information[model_name] = {
                "input_shape": [
                    int(value)
                    for value in input_detail[
                        "shape"
                    ]
                ],
                "input_dtype": str(
                    input_detail["dtype"]
                ),
                "output_shape": [
                    int(value)
                    for value in output_detail[
                        "shape"
                    ]
                ],
                "output_dtype": str(
                    output_detail["dtype"]
                ),
                "ensemble_weight": (
                    self.weights[model_name]
                ),
            }

        return information
