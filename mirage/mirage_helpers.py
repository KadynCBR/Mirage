import cv2
from cv2.typing import MatLike
from argparse import ArgumentParser
import numpy as np
from mirage.skeleton import KeypointMappings, SkeletonDetection


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


def k_coord(image: MatLike, keypoint: tuple[int, int, int]) -> tuple[int, int]:
    ratio = image.shape[1] / image.shape[0]
    # assuming landscape.. and padded to be square. TODO: make more robust or informed.
    padding_val = (image.shape[0] - image.shape[1]) / 2
    y_val: float = keypoint[0]
    x_val: float = keypoint[1]
    return (int(x_val * image.shape[1]), int(y_val * image.shape[0] * ratio + padding_val))


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
        if KeypointMappings[edge_k[0]].display and KeypointMappings[edge_k[1]].display:
            drawn_image = cv2.line(
                drawn_image,
                k_coord(drawn_image, keypoints[edge_k[0]]),
                k_coord(drawn_image, keypoints[edge_k[1]]),
                edge_v,
                3,
            )
    return drawn_image


def skeleton_to_image(image: MatLike, skele: SkeletonDetection, min_confidence: float = 0.2, display_confidence=False):
    drawn_image: MatLike = image.copy()
    for i, joint in skele.joints.items():
        kpmap = joint
        if kpmap.display:
            y_val: float = joint.estimate[2]
            x_val: float = joint.estimate[0]
            confidence: float = joint.confidence
            drawn_image = cv2.circle(
                drawn_image,
                k_coord(drawn_image, (y_val, x_val, 1)),
                radius=5,
                color=kpmap.color if confidence > min_confidence else (0, 0, 255),
                thickness=2,
            )
    for edge_k, edge_v in KeypointEdges.items():
        if skele.joints[edge_k[0]].display and skele.joints[edge_k[1]].display:
            drawn_image = cv2.line(
                drawn_image,
                k_coord(drawn_image, (skele.joints[edge_k[0]].estimate[2], skele.joints[edge_k[0]].estimate[0])),
                k_coord(drawn_image, (skele.joints[edge_k[1]].estimate[2], skele.joints[edge_k[1]].estimate[0])),
                edge_v,
                3,
            )
    if display_confidence:
        drawn_image = display_log_info(drawn_image, skele)
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


def default_crop_region(image_height: int, image_width: int) -> dict[str, float]:
    """Default cropping region
    taken from movenet tutorials
    """
    return None
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {"y_min": y_min, "x_min": x_min, "height": box_height, "width": box_width}


def torso_visible(skele: SkeletonDetection, min_confidence: float = 0.2):
    """is torso visible, and are the points required for visibility confident enough?
    taken from movenet tutorials
    """
    return (skele.joints[11].confidence > min_confidence or skele.joints[12].confidence > min_confidence) and (
        skele.joints[5].confidence > min_confidence or skele.joints[6].confidence > min_confidence
    )


def determine_torso_and_body_range(
    skele: SkeletonDetection,
    center_y: float,
    center_x: float,
    image_width: int,
    image_height: int,
    min_confidence: float = 0.2,
) -> list[float, float, float, float]:
    torso_joints = [5, 6, 11, 12]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - skele.joints[joint].current_xy[1])
        dist_x = abs(center_x - skele.joints[joint].current_xy[0])
        max_torso_yrange = max(dist_y, max_torso_yrange)
        max_torso_xrange = max(dist_x, max_torso_xrange)

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for num, joint in skele.joints.items():
        if joint.confidence < min_confidence:
            continue
        dist_y = abs(center_y - joint.current_xy[1])
        dist_x = abs(center_x - joint.current_xy[0])
        max_body_yrange = max(dist_y, max_body_yrange)
        max_body_xrange = max(dist_x, max_body_xrange)
    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def determine_crop_region(skele: SkeletonDetection, image_height, image_width) -> dict[str, float]:
    """determine the crop region to run inference, uses the skeleton detection to get
    a square region that encloses the full body of the target person.  when not confident in
    torso projections, falls back on full image padded to square. Modified from movenet tutorial
    """
    if not torso_visible(skele):
        return default_crop_region(image_height, image_width)
    center_y = (skele.joints[11].current_xy[1] + skele.joints[12].current_xy[1]) / 2
    center_x = (skele.joints[11].current_xy[0] + skele.joints[12].current_xy[0]) / 2
    (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
        skele, center_y, center_x, image_width, image_height
    )
    # from ratio to resolution space
    max_body_yrange *= image_height
    max_body_xrange *= image_width
    max_torso_yrange *= image_height
    max_torso_xrange *= image_width
    center_x *= image_width
    center_y *= image_height

    ranges = [max_torso_xrange * 2, max_torso_yrange * 2, max_body_xrange * 2, max_body_yrange * 2]
    crop_length_half = np.amax(ranges)
    tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin([crop_length_half, np.amax(tmp)])
    crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
    crop_length = crop_length_half * 2
    out_of_bound = (
        crop_corner[0] < 0
        or crop_corner[1] < 0
        or ((crop_corner[0] + crop_length) - crop_corner[0]) > image_height
        or ((crop_corner[1] + crop_length) - crop_corner[1]) > image_width
        or crop_length_half > max(image_width, image_height) / 2
    )

    if out_of_bound:
        return default_crop_region(image_height, image_width)
    else:
        return {
            "y_min": crop_corner[0],
            "x_min": crop_corner[1],
            "height": (crop_corner[0] + crop_length) - crop_corner[0],
            "width": (crop_corner[1] + crop_length) - crop_corner[1],
        }


def display_log_info(im: MatLike, skele: SkeletonDetection):
    img = im.copy()
    starting_y = 150
    starting_x = im.shape[1] - 350
    height_per_text = 30
    img = overlay_rect(img, starting_x - 30, 20, im.shape[1] - starting_x + 20, im.shape[0] - 40)
    for j in skele.joints.values():
        out = f"{j.name}: {j.confidence*100:.0f}%"
        img = cv2.putText(img, f"{out:<20}", (starting_x, starting_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        starting_y += height_per_text
    return img


def overlay_rect(img: MatLike, x: int, y: int, w: int, h: int):
    # First we crop the sub-rect from the image
    sub_img = img[y : y + h, x : x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.25, white_rect, 0.75, 1.0)
    # Putting the image back to its position
    img[y : y + h, x : x + w] = res
    return img
