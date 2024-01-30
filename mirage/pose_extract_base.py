import cv2
from cv2.typing import MatLike
from abc import ABC, abstractmethod
from typing import Any


class MLAbstractInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, image: MatLike) -> MatLike:
        pass

    @abstractmethod
    def load_model(self) -> Any:
        pass

    @abstractmethod
    def predict(self, image: MatLike) -> MatLike:
        pass
