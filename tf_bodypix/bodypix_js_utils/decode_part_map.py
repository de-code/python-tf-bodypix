# based on:
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/decode_part_map.ts

try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np


DEFAULT_DTYPE = (
    tf.int32 if tf is not None else np.int32
)


def to_mask_tensor(
    segment_scores: np.ndarray,
    threshold: float,
    dtype: type = DEFAULT_DTYPE
) -> np.ndarray:
    if tf is None:
        return (segment_scores > threshold).astype(dtype)
    return tf.cast(
        tf.greater(segment_scores, threshold),
        dtype
    )
