import logging

from typing import Dict

from ..types import PartWithScore, TensorBuffer3D, Keypoint, Vector2D
from ..keypoints import PART_NAMES, PART_IDS, POSE_CHAIN
from .util import getImageCoords, clamp, addVectors, getOffsetPoint


LOGGER = logging.getLogger(__name__)


parentChildrenTuples = [
    (PART_IDS[parentJoinName], PART_IDS[childJoinName])
    for parentJoinName, childJoinName in POSE_CHAIN
]

parentToChildEdges = [
    childJointId
    for _, childJointId in parentChildrenTuples
]

childToParentEdges = [
    parentJointId
    for parentJointId, _ in parentChildrenTuples
]


def getDisplacement(
    edgeId: int, point: Vector2D, displacements: TensorBuffer3D
) -> Vector2D:
    numEdges = displacements.shape[2] // 2
    # LOGGER.debug('point=%s, edgeId=%s, numEdges=%s', point, edgeId, numEdges)
    return Vector2D(
        y=displacements[point.y, point.x, edgeId],
        x=displacements[point.y, point.x, numEdges + edgeId]
    )


def getStridedIndexNearPoint(
    point: Vector2D, outputStride: int, height: int,
    width: int
) -> Vector2D:
    # LOGGER.debug('point: %s', point)
    return Vector2D(
        y=clamp(round(point.y / outputStride), 0, height - 1),
        x=clamp(round(point.x / outputStride), 0, width - 1)
    )


def traverseToTargetKeypoint(  # pylint: disable=too-many-locals
    edgeId: int,
    sourceKeypoint: Keypoint,
    targetKeypointId: int,
    scoresBuffer: TensorBuffer3D,
    offsets: TensorBuffer3D, outputStride: int,
    displacements: TensorBuffer3D,
    offsetRefineStep: int = 2
) -> Keypoint:
    height, width = scoresBuffer.shape[:2]

    # Nearest neighbor interpolation for the source->target displacements.
    sourceKeypointIndices = getStridedIndexNearPoint(
        sourceKeypoint.position, outputStride, height, width
    )

    displacement = getDisplacement(
        edgeId, sourceKeypointIndices, displacements
    )

    displacedPoint = addVectors(sourceKeypoint.position, displacement)
    targetKeypoint = displacedPoint
    for _ in range(offsetRefineStep):
        targetKeypointIndices = getStridedIndexNearPoint(
            targetKeypoint, outputStride, height, width
        )

        offsetPoint = getOffsetPoint(
            targetKeypointIndices.y, targetKeypointIndices.x, targetKeypointId,
            offsets
        )

        targetKeypoint = addVectors(
            Vector2D(
                x=targetKeypointIndices.x * outputStride,
                y=targetKeypointIndices.y * outputStride
            ),
            Vector2D(
                x=offsetPoint.x, y=offsetPoint.y
            )
        )

    targetKeyPointIndices = getStridedIndexNearPoint(
        targetKeypoint, outputStride, height, width
    )
    score = scoresBuffer[
        targetKeyPointIndices.y, targetKeyPointIndices.x, targetKeypointId
    ]

    return Keypoint(
        position=targetKeypoint,
        part=PART_NAMES[targetKeypointId],
        score=score
    )


def decodePose(
    root: PartWithScore, scores: TensorBuffer3D, offsets: TensorBuffer3D,
    outputStride: int, displacementsFwd: TensorBuffer3D,
    displacementsBwd: TensorBuffer3D
) -> Dict[int, Keypoint]:
    # numParts = scores.shape[2]
    numEdges = len(parentToChildEdges)

    instanceKeypoints: Dict[int, Keypoint] = {}
    # Start a new detection instance at the position of the root.
    # const {part: rootPart, score: rootScore} = root;
    rootPoint = getImageCoords(root.part, outputStride, offsets)

    instanceKeypoints[root.part.keypoint_id] = Keypoint(
        score=root.score,
        part=PART_NAMES[root.part.keypoint_id],
        position=rootPoint
    )

    # Decode the part positions upwards in the tree, following the backward
    # displacements.
    for edge in reversed(range(numEdges)):
        sourceKeypointId = parentToChildEdges[edge]
        targetKeypointId = childToParentEdges[edge]
        if (
            instanceKeypoints.get(sourceKeypointId)
            and not instanceKeypoints.get(targetKeypointId)
        ):
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, outputStride, displacementsBwd
            )

    # Decode the part positions downwards in the tree, following the forward
    # displacements.
    for edge in range(numEdges):
        sourceKeypointId = childToParentEdges[edge]
        targetKeypointId = parentToChildEdges[edge]
        if (
            instanceKeypoints.get(sourceKeypointId)
            and not instanceKeypoints.get(targetKeypointId)
        ):
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores,
                offsets, outputStride, displacementsFwd
            )

    return instanceKeypoints
