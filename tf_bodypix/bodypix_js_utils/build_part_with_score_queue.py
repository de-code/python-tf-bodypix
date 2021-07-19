# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/build_part_with_score_queue.ts

import logging
from collections import deque
from typing import Deque

from tf_bodypix.bodypix_js_utils.types import PartWithScore, Part, T_ArrayLike_3D


LOGGER = logging.getLogger(__name__)


def score_is_maximum_in_local_window(
    keypoint_id: int,
    score: float,
    heatmap_y: int,
    heatmap_x: int,
    local_maximum_radius: float,
    scores: T_ArrayLike_3D
) -> bool:
    height, width = scores.shape[:2]
    y_start = int(max(heatmap_y - local_maximum_radius, 0))
    y_end = int(min(heatmap_y + local_maximum_radius + 1, height))
    for y_current in range(y_start, y_end):
        x_start = int(max(heatmap_x - local_maximum_radius, 0))
        x_end = int(min(heatmap_x + local_maximum_radius + 1, width))
        for x_current in range(x_start, x_end):
            if scores[y_current, x_current, keypoint_id] > score:
                return False
    return True


def build_part_with_score_queue(
    score_threshold: float,
    local_maximum_radius: float,
    scores: T_ArrayLike_3D
) -> Deque[PartWithScore]:
    height, width, num_keypoints = scores.shape[:3]
    part_with_scores = []

    LOGGER.debug('num_keypoints=%s', num_keypoints)

    for heatmap_y in range(height):
        for heatmap_x in range(width):
            for keypoint_id in range(num_keypoints):
                score = scores[heatmap_y, heatmap_x, keypoint_id]

                # Only consider parts with score greater or equal to threshold as
                # root candidates.
                if score < score_threshold:
                    continue

                # Only consider keypoints whose score is maximum in a local window.
                if not score_is_maximum_in_local_window(
                    keypoint_id, score, heatmap_y, heatmap_x, local_maximum_radius,
                    scores
                ):
                    continue

                part_with_scores.append(PartWithScore(
                    score=score,
                    part=Part(heatmap_y=heatmap_y, heatmap_x=heatmap_x, keypoint_id=keypoint_id)
                ))

    return deque(
        sorted(
            part_with_scores,
            key=lambda part_with_score: part_with_score.score,
            reverse=True
        )
    )
