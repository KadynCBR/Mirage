import cv2
from typing import Any
import numpy as np
from cv2.typing import MatLike


class CameraInterface:
    def __init__(self, input_file_name: str = None):
        self.IsImage = False
        self.input_file_name = input_file_name
        self.cap = None
        if input_file_name.split(".")[-1] in ["jpg", "jpeg", "png"]:
            self.IsImage = True
        else:
            self.cap = cv2.VideoCapture(self.input_file_name)

    def get_frame(self, frame_number: int = 0) -> MatLike:
        if self.IsImage:
            return cv2.imread(self.input_file_name)
        else:
            self.set_frame(frame_number)
            return self.get_next_frame()

    def set_frame(self, frame_number: int = 0) -> None:
        if frame_number > self.get_total_frames():
            print(
                f"ERROR[CameraInterface]: Attempting to set frame {frame_number}, when max frames for capture is {self.get_total_frames}."
            )
            exit(1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_next_frame(self) -> MatLike:
        if self.IsImage:
            return cv2.imread(self.input_file_name)
        else:
            success, image = self.cap.read()
            if success:
                return image

    def get_total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
