from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path

from ai_edge_litert.interpreter import Interpreter



class InferenceService:


    TARGET_SIZE = (
        224,
        224
    )


    IMAGE_NET_MEAN = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )


    IMAGE_NET_STD = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )



    def __init__(
        self,
        model_paths: dict[str, str],
        labels_path: str
    ):


        self.models = {}


        self.labels = self.load_labels(
            labels_path
        )


        self.load_models(
            model_paths
        )



    # ======================================
    # LOAD LABEL
    # ======================================

    def load_labels(
        self,
        path
    ):


        with open(
            path,
            "r"
        ) as file:

            labels = [
                line.strip()
                for line in file.readlines()
            ]


        return labels




    # ======================================
    # LOAD TFLITE MODELS
    # ======================================

    def load_models(
        self,
        model_paths
    ):


        for name, path in model_paths.items():


            print(
                f"Loading {name}"
            )


            interpreter = Interpreter(
                model_path=str(
                    Path(path)
                )
            )


            interpreter.allocate_tensors()


            input_details = (
                interpreter
                .get_input_details()
            )


            output_details = (
                interpreter
                .get_output_details()
            )



            self.models[name] = {

                "interpreter":
                    interpreter,


                "input":
                    input_details,


                "output":
                    output_details

            }


            print(
                f"{name} loaded"
            )



    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    def preprocess(
        self,
        frame
    ):


        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = cv2.resize(
            image,
            self.TARGET_SIZE
        )


        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )


        image = (
            image -
            self.IMAGE_NET_MEAN
        ) / self.IMAGE_NET_STD



        image = np.expand_dims(
            image,
            axis=0
        )


        return image.astype(
            np.float32
        )



    # ======================================
    # SOFTMAX
    # ======================================

    def softmax(
        self,
        x
    ):


        exp = np.exp(
            x -
            np.max(x)
        )


        return (
            exp /
            np.sum(exp)
        )



    # ======================================
    # SINGLE MODEL PREDICTION
    # ======================================

    def predict_model(
        self,
        model_name,
        image
    ):


        model = self.models[
            model_name
        ]


        interpreter = model[
            "interpreter"
        ]


        input_index = model[
            "input"
        ][0]["index"]


        output_index = model[
            "output"
        ][0]["index"]



        interpreter.set_tensor(
            input_index,
            image
        )


        interpreter.invoke()



        output = interpreter.get_tensor(
            output_index
        )[0]



        probabilities = self.softmax(
            output
        )


        return probabilities



    # ======================================
    # ENSEMBLE SOFT VOTING
    # ======================================

    def predict(
        self,
        frame
    ):


        image = self.preprocess(
            frame
        )



        all_predictions = []



        for name in self.models:


            prediction = (
                self.predict_model(
                    name,
                    image
                )
            )


            all_predictions.append(
                prediction
            )



            print(
                name,
                "done"
            )



        # rata-rata probabilitas

        ensemble_probability = (
            np.mean(
                all_predictions,
                axis=0
            )
        )



        class_id = int(
            np.argmax(
                ensemble_probability
            )
        )



        confidence = float(
            ensemble_probability[
                class_id
            ]
        )



        label = self.labels[
            class_id
        ]



        probability_dict = {}



        for i, name in enumerate(
            self.labels
        ):

            probability_dict[name] = (
                float(
                    ensemble_probability[i]
                )
            )



        result = {

            "label":
                label,


            "class_id":
                class_id,


            "confidence":
                confidence,


            "probabilities":
                probability_dict

        }



        return result
