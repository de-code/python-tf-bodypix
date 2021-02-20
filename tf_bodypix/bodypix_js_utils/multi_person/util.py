# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/util.ts

import logging

from ..types import Part, TensorBuffer3D, Vector2D
from ..keypoints import NUM_KEYPOINTS


LOGGER = logging.getLogger(__name__)


def getOffsetPoint(
    y: int, x: int, keypoint_id: int, offsets: TensorBuffer3D
) -> Vector2D:
    return Vector2D(
        y=offsets[y, x, keypoint_id],
        x=offsets[y, x, keypoint_id + NUM_KEYPOINTS]
    )


def getImageCoords(
    part: Part, outputStride: int, offsets: TensorBuffer3D
) -> Vector2D:
    LOGGER.debug('part: %s', part)
    offset_point = getOffsetPoint(
        part.heatmap_y, part.heatmap_x, part.keypoint_id, offsets
    )
    LOGGER.debug('offset_point: %s', offset_point)
    LOGGER.debug('offsets.shape: %s', offsets.shape)
    return Vector2D(
        x=part.heatmap_x * outputStride + offset_point.x,
        y=part.heatmap_y * outputStride + offset_point.y
    )


def clamp(a: int, min_value: int, max_value: int) -> int:
    return min(max_value, max(min_value, a))


def squaredDistance(
    y1: int, x1: int, y2: int, x2: int
) -> int:
    dy = y2 - y1
    dx = x2 - x1
    return dy * dy + dx * dx


def squared_distance_vector(a: Vector2D, b: Vector2D) -> int:
    return squaredDistance(a.y, a.x, b.y, b.x)


def addVectors(a: Vector2D, b: Vector2D) -> Vector2D:
    return Vector2D(x=a.x + b.x, y=a.y + b.y)
