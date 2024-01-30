import cv2
from cv2.typing import MatLike
from argparse import ArgumentParser
import numpy as np
from dataclasses import dataclass


@dataclass
class KeypointInformation:
    name: str
    color: tuple[int, int, int]
    display: bool


KeypointMappings = {
    0: KeypointInformation("nose", (0, 0, 255), True),
    1: KeypointInformation("lefteye", (0, 0, 255), False),
    2: KeypointInformation("righteye", (0, 0, 255), False),
    3: KeypointInformation("leftear", (0, 0, 255), False),
    4: KeypointInformation("rightear", (0, 0, 255), False),
    5: KeypointInformation("leftshoulder", (0, 255, 255), True),
    6: KeypointInformation("rightshoulder", (0, 255, 255), True),
    7: KeypointInformation("leftelbow", (0, 255, 255), True),
    8: KeypointInformation("rightelbow", (0, 255, 255), True),
    9: KeypointInformation("leftwrist", (100, 100, 255), True),
    10: KeypointInformation("rightwrist", (100, 100, 255), True),
    11: KeypointInformation("lefthip", (100, 0, 100), True),
    12: KeypointInformation("righthip", (100, 0, 100), True),
    13: KeypointInformation("leftknee", (100, 0, 100), True),
    14: KeypointInformation("rightknee", (100, 0, 100), True),
    15: KeypointInformation("leftankle", (100, 0, 100), True),
    16: KeypointInformation("rightankle", (100, 0, 100), True),
}

KeypointEdges = {
    (0, 1): (50, 100, 200),
    (0, 2): (10, 200, 100),
    (1, 3): (50, 100, 200),
    (2, 4): (10, 200, 100),
    (0, 5): (50, 100, 200),
    (0, 6): (10, 200, 100),
    (5, 7): (50, 100, 200),
    (7, 9): (50, 100, 200),
    (6, 8): (10, 200, 100),
    (8, 10): (10, 200, 100),
    (5, 6): (100, 100, 0),
    (5, 11): (50, 100, 200),
    (6, 12): (10, 200, 100),
    (11, 12): (100, 100, 0),
    (11, 13): (50, 100, 200),
    (13, 15): (50, 100, 200),
    (12, 14): (10, 200, 100),
    (14, 16): (10, 200, 100),
}


def crop_image(image: MatLike, y: int, height: int, x: int, width: int, margin: int = 25) -> MatLike:
    return image[y + margin : y + height - margin, x + margin : x + width - margin]


def k_coord(image: MatLike, keypoint: tuple[int, int, int], adjust_val: int = -280) -> tuple[int, int]:
    ratio = image.shape[1] / image.shape[0]
    y_val: float = keypoint[0]
    x_val: float = keypoint[1]
    return (int(x_val * image.shape[1]), int(y_val * image.shape[0] * ratio - 280))


def keypoint_to_image(image: MatLike, keypoints: MatLike, min_confidence: float = 0.2) -> MatLike:
    drawn_image: MatLike = image.copy()
    for i, kp in enumerate(keypoints):
        kpmap = KeypointMappings[i]
        if kpmap.display:
            y_val: float = kp[0]
            x_val: float = kp[1]
            confidence: float = kp[2]
            if confidence > min_confidence:
                drawn_image = cv2.circle(
                    drawn_image,
                    k_coord(drawn_image, kp),
                    radius=5,
                    color=kpmap.color,
                    thickness=2,
                )
    for edge_k, edge_v in KeypointEdges.items():
        drawn_image = cv2.line(
            drawn_image,
            k_coord(drawn_image, keypoints[edge_k[0]]),
            k_coord(drawn_image, keypoints[edge_k[1]]),
            edge_v,
            3,
        )
    return drawn_image


def split_image_stack(stacked_image: MatLike, is_vertical_stack: bool = True) -> tuple[MatLike, MatLike]:
    resolution: list[int, int] = list(stacked_image.shape[:2])
    if is_vertical_stack:
        resolution[0] = int(resolution[0] / 2)
        return (
            crop_image(stacked_image, 0, resolution[0], 0, resolution[1], 0),
            crop_image(stacked_image, resolution[0], resolution[0], 0, resolution[1], 0),
        )
    else:
        resolution[1] = int(resolution[1] / 2)
        return (
            crop_image(stacked_image, 0, resolution[0], 0, resolution[1], 0),
            crop_image(stacked_image, 0, resolution[0], resolution[1], resolution[1], 0),
        )


def stack_image(image_a: MatLike, image_b: MatLike, is_vertical_stack: bool = True) -> MatLike:
    if is_vertical_stack:
        return np.concatenate((image_a, image_b), axis=0)
    else:
        return np.concatenate((image_a, image_b), axis=1)
