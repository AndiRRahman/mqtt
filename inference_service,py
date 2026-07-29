from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter


class InferenceService:

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        input_size: int = 224,
    ) -> None:

        self.model_path = Path(model_path)

        self.labels_path = Path(labels_path)

        self.input_size = input_size


        self.labels = self._load_labels()


        self.interpreter = Interpreter(
            model_path=str(
                self.model_path
            )
        )


        self.interpreter.allocate_tensors()


        self.input_details = (
            self.interpreter
            .get_input_details()
        )


        self.output_details = (
            self.interpreter
            .get_output_details()
        )


        print(
            "Model berhasil dimuat"
        )


        print(
            "Input:",
            self.input_details
        )


        print(
            "Output:",
            self.output_details
        )



    # ==========================================
    # LOAD LABEL
    # ==========================================

    def _load_labels(
        self,
    ) -> list[str]:

        with open(
            self.labels_path,
            "r",
            encoding="utf-8",
        ) as file:

            labels = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]


        return labels



    # ==========================================
    # PREPROCESS IMAGE
    # ==========================================

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:


        image = cv2.resize(
            frame,
            (
                self.input_size,
                self.input_size,
            )
        )


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )


        image = image.astype(
            np.float32
        )


        image = image / 255.0


        image = np.expand_dims(
            image,
            axis=0,
        )


        return image



    # ==========================================
    # PREDICT
    # ==========================================

    def predict(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:


        input_data = self._preprocess(
            frame
        )


        input_index = (
            self.input_details[0]
            ["index"]
        )


        self.interpreter.set_tensor(
            input_index,
            input_data,
        )


        self.interpreter.invoke()



        output_index = (
            self.output_details[0]
            ["index"]
        )


        output = (
            self.interpreter
            .get_tensor(output_index)
        )



        probabilities = (
            output[0]
        )



        class_index = int(
            np.argmax(
                probabilities
            )
        )


        confidence = float(
            probabilities[class_index]
        )



        label = "Unknown"


        if class_index < len(
            self.labels
        ):

            label = (
                self.labels[class_index]
            )



        return {

            "label": label,

            "confidence": confidence,

            "class_id": class_index,

        }
