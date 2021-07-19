# based on;
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.5/body-pix/src/multi_person/decode_multiple_poses.ts

import logging

from typing import Dict, List

from tf_bodypix.bodypix_js_utils.types import (
    Pose, TensorBuffer3D, Vector2D,
    Keypoint
)
from tf_bodypix.bodypix_js_utils.build_part_with_score_queue import (
    build_part_with_score_queue
)

from .util import getImageCoords, squared_distance_vector
from .decode_pose import decodePose


LOGGER = logging.getLogger(__name__)


kLocalMaximumRadius = 1


def withinNmsRadiusOfCorrespondingPoint(
    poses: List[Pose],
    squaredNmsRadius: float,
    vector: Vector2D,
    keypointId: int
) -> bool:
    return any(
        squared_distance_vector(
            vector, pose.keypoints[keypointId].position
        ) <= squaredNmsRadius
        for pose in poses
    )


def getInstanceScore(
    existingPoses: List[Pose],
    squaredNmsRadius: float,
    instanceKeypoints: Dict[int, Keypoint]
) -> float:
    LOGGER.debug('instanceKeypoints: %s', instanceKeypoints)
    notOverlappedKeypointScores = sum([
        keypoint.score
        for keypointId, keypoint in instanceKeypoints.items()
        if not withinNmsRadiusOfCorrespondingPoint(
            existingPoses, squaredNmsRadius,
            keypoint.position, keypointId
        )
    ])

    return notOverlappedKeypointScores / len(instanceKeypoints)


def decodeMultiplePoses(
    scoresBuffer: TensorBuffer3D, offsetsBuffer: TensorBuffer3D,
    displacementsFwdBuffer: TensorBuffer3D,
    displacementsBwdBuffer: TensorBuffer3D, outputStride: int,
    maxPoseDetections: int, scoreThreshold: float = 0.5, nmsRadius: float = 20
) -> List[Pose]:
    poses: List[Pose] = []

    queue = build_part_with_score_queue(
        scoreThreshold, kLocalMaximumRadius, scoresBuffer
    )
    # LOGGER.debug('queue: %s', queue)

    squaredNmsRadius = nmsRadius * nmsRadius

    # Generate at most maxDetections object instances per image in
    # decreasing root part score order.
    while len(poses) < maxPoseDetections and queue:
        # The top element in the queue is the next root candidate.
        root = queue.popleft()

        # Part-based non-maximum suppression: We reject a root candidate if it
        # is within a disk of `nmsRadius` pixels from the corresponding part of
        # a previously detected instance.
        rootImageCoords = getImageCoords(
            root.part, outputStride, offsetsBuffer
        )
        if withinNmsRadiusOfCorrespondingPoint(
            poses, squaredNmsRadius, rootImageCoords, root.part.keypoint_id
        ):
            continue

        # Start a new detection instance at the position of the root.
        keypoints = decodePose(
            root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer,
            displacementsBwdBuffer
        )

        # LOGGER.debug('keypoints: %s', keypoints)

        score = getInstanceScore(poses, squaredNmsRadius, keypoints)

        poses.append(Pose(keypoints=keypoints, score=score))

    return poses
