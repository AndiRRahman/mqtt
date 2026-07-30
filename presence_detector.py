from __future__ import annotations

import cv2
import numpy as np


class PresenceDetector:

    def __init__(
        self,
        threshold=20000,
        min_area=5000
    ):

        self.background = None

        self.threshold = threshold

        self.min_area = min_area



    def detect(
        self,
        frame
    ) -> bool:


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray = cv2.GaussianBlur(
            gray,
            (21,21),
            0
        )


        if self.background is None:

            self.background = gray

            return False



        diff = cv2.absdiff(
            self.background,
            gray
        )


        _, thresh = cv2.threshold(
            diff,
            30,
            255,
            cv2.THRESH_BINARY
        )


        thresh = cv2.dilate(
            thresh,
            None,
            iterations=2
        )


        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        for contour in contours:

            area = cv2.contourArea(
                contour
            )


            if area > self.min_area:

                return True



        return False
