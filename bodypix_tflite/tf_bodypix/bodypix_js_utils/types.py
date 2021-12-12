# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/types.ts


from typing import Dict, NamedTuple

import numpy as np


class Part(NamedTuple):
    heatmap_x: int
    heatmap_y: int
    keypoint_id: int


class Vector2D(NamedTuple):
    y: float
    x: float


TensorBuffer3D = np.ndarray
T_ArrayLike_3D = TensorBuffer3D


class PartWithScore(NamedTuple):
    score: float
    part: Part


class Keypoint(NamedTuple):
    score: float
    position: Vector2D
    part: str


class Pose(NamedTuple):
    keypoints: Dict[int, Keypoint]
    score: float
