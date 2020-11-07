import logging
from unittest.mock import MagicMock

import numpy as np

from tf_bodypix.model import BodyPixModelWrapper


LOGGER = logging.getLogger(__name__)


ANY_INT_FACTOR_1 = 5


class TestBodyPixModelWrapper:
    def test_should_be_able_to_padded_and_resized_image_matching_output_stride_plus_one(self):
        predict_fn = MagicMock(name='predict_fn')
        output_stride = 16
        internal_resolution = 0.5
        model = BodyPixModelWrapper(
            predict_fn=predict_fn,
            output_stride=output_stride,
            internal_resolution=internal_resolution
        )
        resolution_matching_output_stride_plus_1 = int(
            (output_stride * ANY_INT_FACTOR_1 + 1) / internal_resolution
        )
        LOGGER.debug(
            'resolution_matching_output_stride_plus_1: %s',
            resolution_matching_output_stride_plus_1
        )
        image = np.ones(
            shape=(
                resolution_matching_output_stride_plus_1,
                resolution_matching_output_stride_plus_1,
                3
            )
        )
        model.predict_single(image)
