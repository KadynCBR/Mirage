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
        im = image.copy()
        if crop_region is not None:
            im = crop_image(
                im,
                int(crop_region["y_min"]),
                int(crop_region["height"]),
                int(crop_region["x_min"]),
                int(crop_region["width"]),
                0,
            )
        im = self.preprocess(im)
        outputs = self.movenet(im)
        keypoints = outputs["output_0"][0][0].numpy()
        keypoints = self.keypoint_to_original_image_space(keypoints, image, crop_region)
        return keypoints

    def keypoint_to_original_image_space(
        self, keypoints: np.ndarray, image: MatLike, crop_region: dict[str, int] | None = None
    ) -> np.ndarray:
        if crop_region is None:
            return keypoints
        print(keypoints)
        print("-" * 10)
        print(image.shape)
        # scale_factor_width = image.shape[1] / float(crop_region["width"])
        # scale_factor_height = image.shape[0] / float(crop_region["height"])
        # y_add = crop_region["y_min"] * 2 / float(image.shape[0])
        # x_add = crop_region["x_min"] / float(image.shape[1])
        scale_factor_height = float(crop_region["height"]) / float(image.shape[0])
        scale_factor_width = float(crop_region["width"]) / float(image.shape[1])
        print(scale_factor_height)
        print(scale_factor_width)
        # print(y_mult)
        # print(x_multi)
        for i in range(len(keypoints)):
            keypoints[i][0] = (crop_region["y_min"] / float(image.shape[0])) + keypoints[i][0] * (scale_factor_height)
            keypoints[i][1] = (crop_region["x_min"] / float(image.shape[1])) + keypoints[i][1] * (scale_factor_width)
        print(keypoints)
        return keypoints
