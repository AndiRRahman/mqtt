import cv2
import numpy as np


class PresenceDetector:


    def __init__(
        self,
        threshold=5000
    ):

        self.previous_frame = None

        self.threshold = threshold



    def detect(
        self,
        frame
    ):


        # ======================================
        # GRAYSCALE
        # ======================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )



        # ======================================
        # FRAME PERTAMA
        # ======================================

        if self.previous_frame is None:


            self.previous_frame = gray


            return False



        # ======================================
        # HITUNG PERUBAHAN FRAME
        # ======================================

        difference = cv2.absdiff(
            self.previous_frame,
            gray
        )



        _, thresh = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )



        # jumlah area berubah

        changed_pixels = np.sum(
            thresh
        )



        # ======================================
        # DETEKSI OBJEK
        # ======================================

        object_detected = (

            changed_pixels >

            self.threshold

        )



        # ======================================
        # UPDATE BACKGROUND
        # HANYA JIKA TIDAK ADA OBJEK
        # ======================================

        if not object_detected:


            self.previous_frame = gray



        return object_detected
