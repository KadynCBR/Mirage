#!/bin/env python
import cv2
from typing import Any
import subprocess
import os
from cv2.typing import MatLike
from argparse import ArgumentParser
import numpy as np
from mirage.mirage_helpers import *
from mirage.pose_extract_base import MLAbstractInterface
from mirage.movenet import MovenetInterface
from mirage.rgb_interface import CameraInterface
from mirage.skeleton import SkeletonDetection

s_a = SkeletonDetection()
s_b = SkeletonDetection()


def process_and_viz_split(i_cam: CameraInterface, ml_interface: MLAbstractInterface) -> MatLike:
    try:
        img = i_cam.get_next_frame()
        img_a, img_b = split_image_stack(img)
        a_crop = determine_crop_region(s_a, img_a.shape[0], img_a.shape[1])
        b_crop = determine_crop_region(s_b, img_b.shape[0], img_b.shape[1])
        img_a_kp = ml_interface.predict(img_a, a_crop)
        img_b_kp = ml_interface.predict(img_b, b_crop)
        s_a.update_predictions(img_a_kp, (1.0 / i_cam.get_frame_rate_per_second()))
        s_b.update_predictions(img_b_kp, (1.0 / i_cam.get_frame_rate_per_second()))
        img_a_viz = skeleton_to_image(img_a, s_a)
        img_b_viz = skeleton_to_image(img_b, s_b)
    except:
        import ipdb

        ipdb.set_trace()
    return stack_image(img_a_viz, img_b_viz)


def process_and_viz(i_cam: CameraInterface, ml_interface: MLAbstractInterface) -> MatLike:
    img = i_cam.get_next_frame()
    img_kp = ml_interface.predict(img)
    s_a.update_predictions(img_kp, (1.0 / i_cam.get_frame_rate_per_second()))
    img_viz = skeleton_to_image(img, s_a)
    return img_viz


def fix_mp4_encoding(unfixed_fn: str, fixed_fn: str):
    subprocess.run(["ffmpeg", "-y", "-i", unfixed_fn, fixed_fn])
    os.remove(unfixed_fn)


def main(
    input_file: str,
    output_fn: str,
    frame_number_start: int,
    frame_number_end: int,
    splitimage: bool = False,
    writeframe: bool = False,
) -> None:
    ml_interface = MovenetInterface()
    i_cam = CameraInterface(input_file)
    if not i_cam.IsImage:
        frames: list[MatLike] = []
        i_cam.set_frame(frame_number_start)
        frame_number_end = frame_number_end if frame_number_end != 0 else i_cam.get_total_frames()
        for frame_number in range(frame_number_end - frame_number_start):
            if splitimage:
                frames.append(process_and_viz_split(i_cam, ml_interface))
            else:
                frames.append(process_and_viz(i_cam, ml_interface))
            if writeframe:
                cv2.imwrite(f"OUTPUT/Processed_{frame_number}.jpg", frames[-1])
        unfixed_fn: str = f"{output_fn[:-5]}_UNFIXED.mp4"
        out = cv2.VideoWriter(
            unfixed_fn,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            20.0,
            (frames[-1].shape[1], frames[-1].shape[0]),
        )
        for frame in frames:
            out.write(frame)
        out.release()
        fix_mp4_encoding(unfixed_fn, output_fn)
    else:
        if splitimage:
            cv2.imwrite(f"{input_file[:-5]}_POSE.jpg", process_and_viz_split(i_cam, ml_interface))
        else:
            cv2.imwrite(f"{input_file[:-5]}_POSE.jpg", process_and_viz(i_cam, ml_interface))


if __name__ == "__main__":
    parser = ArgumentParser("Mirage Motion Capture")
    parser.add_argument("input_file", type=str)
    parser.add_argument("-s", "--frame_start", type=int, required=False, default=0)
    parser.add_argument("-e", "--frame_end", type=int, required=False, default=-1)
    parser.add_argument("--no-split", dest="split", action="store_false", required=False)
    parser.add_argument("-o", "--output_fn", type=str, required=False, default="OUTPUT/Processed.mp4")
    parser.set_defaults(split=True)
    args = parser.parse_args()
    main(args.input_file, args.output_fn, args.frame_start, args.frame_end, args.split, False)
