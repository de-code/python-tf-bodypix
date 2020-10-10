# TensorFlow BodyPix (TF BodyPix)

[![PyPi version](https://pypip.in/v/tf-bodypix/badge.png)](https://pypi.org/project/tf-bodypix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of [body-pix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix).

## Install

Install with all dependencies:

```bash
pip install tf-bodypix[all]
```

Install with minimal or no dependencies:

```bash
pip install tf-bodypix
```

Extras are provided to make it easier to provide or exclude dependencies
when using this project as a library:

| extra name | description
| ---------- | -----------
| tf         | TensorFlow (required). But you may use your own build.
| tfjs       | TensorFlow JS Model support
| image      | Image loading via Pillow, required by the CLI.
| all        | All of the libraries

## CLI

### Creating a simple body mask

```bash
TF_CPP_MIN_LOG_LEVEL=3 \
python -m tf_bodypix \
    image-to-mask \
    --image /path/to/input-image.jpg \
    --output-mask /path/to/output-mask.jpg \
    --threshold=0.75
```

### Colorize the body mask depending on the body part

```bash
TF_CPP_MIN_LOG_LEVEL=3 \
python -m tf_bodypix \
    image-to-mask \
    --image /path/to/input-image.jpg \
    --output-mask /path/to/output-colored-mask.jpg \
    --threshold=0.75 \
    --colored
```

### Additionally select the body parts

```bash
TF_CPP_MIN_LOG_LEVEL=3 \
python -m tf_bodypix \
    image-to-mask \
    --image /path/to/input-image.jpg \
    --output-mask /path/to/output-colored-mask.jpg \
    --threshold=0.75 \
    --parts left_face right_face \
    --colored
```

## API

```python
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

image = tf.keras.preprocessing.image.load_img(
    '/path/to/input-image.jpg'
)
image_array = tf.keras.preprocessing.image.img_to_array(image)
result = bodypix_model.predict_single(image_array)
mask = result.get_mask(threshold=0.75)
tf.keras.preprocessing.image.save_img(
    '/path/to/output-mask.jpg',
    mask
)

colored_mask = result.get_colored_mask(mask)
tf.keras.preprocessing.image.save_img(
    '/path/to/output-colored-mask.jpg',
    colored_mask
)
```

## Acknowledgements

* [Original TensorFlow JS Implementation of BodyPix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix)
* [Linux-Fake-Background-Webcam](https://github.com/fangfufu/Linux-Fake-Background-Webcam), an implementation of the [blog post](https://elder.dev/posts/open-source-virtual-background/) describing using the TensorFlow JS implementation with Python via a Socket API.
* [tfjs-to-tf](https://github.com/patlevin/tfjs-to-tf) for providing an easy way to convert TensorFlow JS models
* [virtual_webcam_background](https://github.com/allo-/virtual_webcam_background) for a great pure Python implementation
