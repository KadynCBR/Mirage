import tensorflow as tf
import tensorflow_hub as hub
from typing import Union
from mirage.mirage_helpers import *
from mirage.pose_extract_base import MLAbstractInterface

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class MovenetInterface(MLAbstractInterface):
    def __init__(self):
        self.load_model()

    def preprocess(self, image: MatLike) -> MatLike:
        image = tf.expand_dims(image, axis=0)  # add 'batch' axis
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
        return image

    def load_model(self) -> None:
        self.model = hub.load(
            "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4"
        )
        self.movenet = self.model.signatures["serving_default"]

    def predict(self, image: MatLike, crop_region: dict[str, int] | None = None) -> MatLike:
        if crop_region is not None:
            print(crop_region)
            image = crop_image(
                image,
                int(crop_region["y_min"]),
                int(crop_region["height"]),
                int(crop_region["x_min"]),
                int(crop_region["width"]),
                0,
            )
        image = self.preprocess(image)
        outputs = self.movenet(image)
        keypoints = outputs["output_0"][0][0].numpy()
        return keypoints
