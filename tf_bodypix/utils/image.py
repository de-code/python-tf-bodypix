import logging
from collections import namedtuple
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from cv2 import cv2
except ImportError:
    cv2 = None


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


ImageArray = np.ndarray


def require_opencv():
    if cv2 is None:
        raise ImportError('OpenCV is required')


def box_blur_image(image: np.ndarray, blur_size: int) -> np.ndarray:
    if not blur_size:
        return image
    require_opencv()
    if len(image.shape) == 4:
        image = image[0]
    result = cv2.blur(np.asarray(image), (blur_size, blur_size))
    if len(result.shape) == 2:
        result = np.expand_dims(result, axis=-1)
    result = result.astype(np.float32)
    return result


def get_image_size(image: np.ndarray):
    height, width, *_ = image.shape
    return ImageSize(height=height, width=width)


def _resize_image_to_using_tf(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    if not resize_method:
        resize_method = tf.image.ResizeMethod.BILINEAR
    return tf.image.resize(
        image_array,
        (image_size.height, image_size.width),
        method=resize_method
    )


def _resize_image_to_using_pillow(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    assert not resize_method or resize_method == 'bilinear'
    import PIL.Image  # pylint: disable=import-outside-toplevel
    if len(image_array.shape) == 4:
        assert image_array.shape[0] == 1
        image_array = image_array[0]
    image_array = image_array.astype(np.int8)
    pil_image = PIL.Image.fromarray(image_array)
    resized_pil_image = pil_image.resize(
        size=[image_size.width, image_size.height],
        resample=PIL.Image.BILINEAR
    )
    resized_image_array = np.asarray(resized_pil_image, dtype=np.float32)
    return resized_image_array


def resize_image_to(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    if get_image_size(image_array) == image_size:
        LOGGER.debug('image has already desired size: %s', image_size)
        return image_array

    if tf is not None:
        return _resize_image_to_using_tf(image_array, image_size, resize_method)
    return _resize_image_to_using_pillow(image_array, image_size, resize_method)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
    return image[..., ::-1]


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(image)


def _load_image_using_tf(
    local_image_path: str,
    image_size: ImageSize = None
) -> np.ndarray:
    image = tf.keras.preprocessing.image.load_img(
        local_image_path
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    if image_size is not None:
        image_array = resize_image_to(image_array, image_size)
    return image_array


def _load_image_using_pillow(
    local_image_path: str,
    image_size: ImageSize = None
) -> np.ndarray:
    import PIL.Image  # pylint: disable=import-outside-toplevel
    with PIL.Image.open(local_image_path) as image:
        image_array = np.asarray(image)
        if image_size is not None:
            image_array = resize_image_to(image_array, image_size)
        return image_array


def load_image(
    local_image_path: str,
    image_size: ImageSize = None
) -> np.ndarray:
    if tf is not None:
        return _load_image_using_tf(local_image_path, image_size=image_size)
    return _load_image_using_pillow(local_image_path, image_size=image_size)
