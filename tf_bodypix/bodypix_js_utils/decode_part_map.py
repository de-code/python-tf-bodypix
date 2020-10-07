# based on:
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/decode_part_map.ts

import tensorflow as tf

import numpy as np


def to_mask_tensor(
    segment_scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    return tf.cast(
        tf.greater(segment_scores, threshold),
        tf.int32
    )
