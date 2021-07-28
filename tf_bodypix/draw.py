import logging
from typing import List, Iterable, Tuple, Optional

import cv2
import numpy as np

from tf_bodypix.utils.image import ImageArray
from tf_bodypix.bodypix_js_utils.types import Pose, Keypoint
from tf_bodypix.bodypix_js_utils.keypoints import CONNECTED_PART_NAMES


LOGGER = logging.getLogger(__name__)


T_Color = Tuple[int, int, int]


def get_filtered_keypoints_by_score(
    keypoints: Iterable[Keypoint],
    min_score: float
) -> List[Keypoint]:
    return [
        keypoint
        for keypoint in keypoints
        if keypoint.score >= min_score
    ]


def get_adjacent_keypoints(
    keypoints: Iterable[Keypoint]
) -> List[Tuple[Keypoint, Keypoint]]:
    keypoint_by_name = {
        keypoint.part: keypoint
        for keypoint in keypoints
    }
    return [
        (keypoint_by_name[part_name_1], keypoint_by_name[part_name_2])
        for part_name_1, part_name_2 in CONNECTED_PART_NAMES
        if keypoint_by_name.get(part_name_1) and keypoint_by_name.get(part_name_2)
    ]


def get_cv_keypoints(keypoints: Iterable[Keypoint]) -> List[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(
            x=keypoint.position.x,
            y=keypoint.position.y,
            size=3
        )
        for keypoint in keypoints
    ]


def draw_skeleton(
    image: ImageArray,
    keypoints: Iterable[Keypoint],
    color: T_Color,
    thickness: int = 3
):
    adjacent_keypoints = get_adjacent_keypoints(keypoints)
    for keypoint_1, keypoint_2 in adjacent_keypoints:
        cv2.line(
            image,
            (round(keypoint_1.position.x), round(keypoint_1.position.y)),
            (round(keypoint_2.position.x), round(keypoint_2.position.y)),
            color=color,
            thickness=thickness
        )
    return image


def draw_keypoints(
    image: ImageArray,
    keypoints: Iterable[Keypoint],
    color: T_Color
):
    image = cv2.drawKeypoints(
        image,
        get_cv_keypoints(keypoints),
        outImage=image,
        color=color
    )
    return image


def draw_pose(
        image: ImageArray,
        pose: Pose,
        min_score: float = 0.1,
        keypoints_color: Optional[T_Color] = None,
        skeleton_color: Optional[T_Color] = None):
    keypoints_to_draw = get_filtered_keypoints_by_score(
        pose.keypoints.values(),
        min_score=min_score
    )
    LOGGER.debug('keypoints_to_draw: %s', keypoints_to_draw)
    if keypoints_color:
        image = draw_keypoints(image, keypoints_to_draw, color=keypoints_color)
    if skeleton_color:
        image = draw_skeleton(image, keypoints_to_draw, color=skeleton_color)
    return image


def draw_poses(image: ImageArray, poses: List[Pose], **kwargs):
    if not poses:
        return image
    output_image = image.astype(np.uint8)
    for pose in poses:
        output_image = draw_pose(output_image, pose, **kwargs)
    return output_image
