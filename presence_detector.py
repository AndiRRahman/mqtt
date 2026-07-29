from __future__ import annotations

import cv2
import numpy as np


class PresenceDetector:

    def __init__(
        self,
        threshold: int = 5000
    ):

        self.background = None
        self.threshold = threshold



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



        difference = cv2.absdiff(
            self.background,
            gray
        )


        _, mask = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )


        pixels = cv2.countNonZero(
            mask
        )


        return pixels > self.threshold
