from __future__ import annotations

import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter


class InferenceService:


    def __init__(
        self,
        model_paths: dict[str, str],
        labels_path: str,
        input_size=(224,224)
    ):


        self.input_size = input_size


        self.labels = self.load_labels(
            labels_path
        )


        self.models = {}


        for name, path in model_paths.items():

            print(
                f"Loading model: {name}"
            )


            interpreter = Interpreter(
                model_path=path
            )


            interpreter.allocate_tensors()


            self.models[name] = interpreter



        print(
            "Semua model berhasil dimuat"
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
    # PREPROCESS IMAGE
    # ======================================

    def preprocess(
        self,
        frame
    ):
    
    
        image = cv2.resize(
            frame,
            self.input_size
        )
    
    
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    
    
        image = image.astype(
            np.float32
        )
    
    
        image = image / 255.0
    
    
    
        # HWC -> CHW
        image = np.transpose(
            image,
            (2,0,1)
        )
    
    
        # tambah batch
        image = np.expand_dims(
            image,
            axis=0
        )
    
    
        return image

    # ======================================
    # SOFTMAX
    # ======================================

    def softmax(
        self,
        output
    ):


        exp = np.exp(
            output -
            np.max(output)
        )


        return exp / np.sum(
            exp
        )




    # ======================================
    # PREDICT SATU MODEL
    # ======================================

    def predict_single(
        self,
        interpreter,
        image
    ):


        input_details = (
            interpreter
            .get_input_details()
        )


        output_details = (
            interpreter
            .get_output_details()
        )


        input_index = (
            input_details[0]["index"]
        )


        output_index = (
            output_details[0]["index"]
        )



        interpreter.set_tensor(
            input_index,
            image
        )


        interpreter.invoke()



        output = (
            interpreter
            .get_tensor(
                output_index
            )
        )



        output = np.squeeze(
            output
        )



        probabilities = self.softmax(
            output
        )



        result = {}



        for i, label in enumerate(
            self.labels
        ):

            result[label] = float(
                probabilities[i]
            )



        return result




#    ======================================
  #  PREDICT SEMUA MODEL
 #   ======================================

    def predict_all(
        self,
        frame
    ):


        image = self.preprocess(
            frame
        )


        results = {}



        for name, interpreter in self.models.items():


            prediction = (
                self.predict_single(
                    interpreter,
                    image
                )
            )


            results[name] = prediction



        return results
