import logging
import os
from collections import namedtuple
from typing import Optional, Sequence

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import PIL.Image
except ImportError:
    PIL = None


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


ImageArray = np.ndarray


class ResizeMethod:
    BILINEAR = 'bilinear'


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
    LOGGER.debug('resizing image: %r -> %r', image_array.shape, image_size)
    return tf.image.resize(
        image_array,
        (image_size.height, image_size.width),
        method=resize_method
    )


def _get_pil_image(image_array: np.ndarray) -> 'PIL.Image':
    if image_array.shape[-1] == 1:
        pil_mode = 'L'
        image_array = np.reshape(image_array, image_array.shape[:2])
    else:
        pil_mode = 'RGB'
    image_array = image_array.astype(np.uint8)
    pil_image = PIL.Image.fromarray(image_array, mode=pil_mode)
    return pil_image


# copied from:
#   https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
def _numpy_bilinear_resize_2d(  # pylint: disable=too-many-locals
    image: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:
    """
    `image` is a 2-D numpy array
    `height` and `width` are the desired spatial dimension of the new 2-D array.
    """
    img_height, img_width = image.shape

    image = image.ravel()

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    y, x = np.divmod(np.arange(height * width), width)

    x_l = np.floor(x_ratio * x).astype('int32')
    y_l = np.floor(y_ratio * y).astype('int32')

    x_h = np.ceil(x_ratio * x).astype('int32')
    y_h = np.ceil(y_ratio * y).astype('int32')

    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = image[y_l * img_width + x_l]
    b = image[y_l * img_width + x_h]
    c = image[y_h * img_width + x_l]
    d = image[y_h * img_width + x_h]

    resized = (
        a * (1 - x_weight) * (1 - y_weight) +
        b * x_weight * (1 - y_weight) +
        c * y_weight * (1 - x_weight) +
        d * x_weight * y_weight
    )

    return resized.reshape(height, width)


def _numpy_bilinear_resize_3d(image: np.ndarray, height: int, width: int) -> np.ndarray:
    _, _, dimensions = image.shape
    return np.stack(
        [
            _numpy_bilinear_resize_2d(
                image[:, :, dimension], height, width
            )
            for dimension in range(dimensions)
        ],
        axis=-1
    )


def _resize_image_to_using_numpy(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    assert not resize_method or resize_method == 'bilinear'
    if len(image_array.shape) == 4:
        assert image_array.shape[0] == 1
        image_array = image_array[0]
    LOGGER.debug(
        'resizing image: %r (%r) -> %r', image_array.shape, image_array.dtype, image_size
    )
    resize_image_array = (
        _numpy_bilinear_resize_3d(
            np.asarray(image_array), image_size.height, image_size.width
        ).astype(image_array.dtype)
    )
    LOGGER.debug(
        'resize_image_array image: %r (%r)', image_array.shape, resize_image_array.dtype
    )
    return resize_image_array


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
    return _resize_image_to_using_numpy(image_array, image_size, resize_method)


def crop_and_resize_batch(  # pylint: disable=too-many-locals
    image_array_batch: np.ndarray,
    boxes: Sequence[Sequence[float]],
    box_indices: Sequence[int],
    crop_size: Sequence[int],
    method='bilinear',
) -> np.ndarray:
    if tf is not None:
        return tf.image.crop_and_resize(
            image_array_batch,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=crop_size,
            method=method
        )
    assert list(box_indices) == [0]
    assert len(boxes) == 1
    assert len(crop_size) == 2
    box = np.array(boxes[0])
    assert np.min(box) >= 0
    assert np.max(box) <= 1
    y1, x1, y2, x2 = list(box)
    assert y1 <= y2
    assert x1 <= x2
    assert len(image_array_batch) == 1
    image_size = get_image_size(image_array_batch[0])
    image_y1 = int(y1 * (image_size.height - 1))
    image_y2 = int(y2 * (image_size.height - 1))
    image_x1 = int(x1 * (image_size.width - 1))
    image_x2 = int(x2 * (image_size.width - 1))
    LOGGER.debug('image y1, x1, y2, x2: %r', (image_y1, image_x1, image_y2, image_x2))
    cropped_image_array = image_array_batch[0][
        image_y1:(1 + image_y2), image_x1: (1 + image_x2), :
    ]
    LOGGER.debug('cropped_image_array: %r', cropped_image_array.shape)
    resized_cropped_image_array = resize_image_to(
        cropped_image_array, ImageSize(height=crop_size[0], width=crop_size[1])
    )
    return np.expand_dims(resized_cropped_image_array, 0)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
    return image[..., ::-1]


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(image)


def _load_image_using_tf(
    local_image_path: str,
    image_size: Optional[ImageSize] = None
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
    image_size: Optional[ImageSize] = None
) -> np.ndarray:
    with PIL.Image.open(local_image_path) as image:
        image_array = np.asarray(image)
        if image_size is not None:
            image_array = resize_image_to(image_array, image_size)
        return image_array


def load_image(
    local_image_path: str,
    image_size: Optional[ImageSize] = None
) -> np.ndarray:
    if tf is not None:
        return _load_image_using_tf(local_image_path, image_size=image_size)
    return _load_image_using_pillow(local_image_path, image_size=image_size)


def save_image_using_tf(image_array: np.ndarray, path: str):
    tf.keras.preprocessing.image.save_img(path, image_array)


def save_image_using_pillow(image_array: np.ndarray, path: str):
    pil_image = _get_pil_image(image_array)
    pil_image.save(path)


def write_image_to(image_array: np.ndarray, path: str):
    LOGGER.info('writing image to: %r', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tf is not None:
        save_image_using_tf(image_array, path)
    else:
        save_image_using_pillow(image_array, path)
