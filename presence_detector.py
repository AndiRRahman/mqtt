import cv2
import numpy as np


class PresenceDetector:


    def __init__(
        self,
        threshold=5000
    ):

        self.threshold = threshold



    def detect(
        self,
        frame
    ):


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        _, binary = cv2.threshold(
            gray,
            80,
            255,
            cv2.THRESH_BINARY_INV
        )


        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )



        for contour in contours:


            area = cv2.contourArea(
                contour
            )


            if area > self.threshold:

                return True



        return False
