# TensorFlow BodyPix (TF BodyPix)

[![PyPi version](https://pypip.in/v/tf-bodypix/badge.png)](https://pypi.org/project/tf-bodypix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of [body-pix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix).

Goals of this project is:

* Python library, making it easy to integrate the BodyPix model
* CLI with limited functionality, mostly for demonstration purpose

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
| webcam     | Webcam support via OpenCV and pyfakewebcam
| all        | All of the libraries

## Python API

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

colored_mask = result.get_colored_part_mask(mask)
tf.keras.preprocessing.image.save_img(
    '/path/to/output-colored-mask.jpg',
    colored_mask
)
```

## CLI

### CLI Help

```bash
python -m tf_bodypix --help
```

or

```bash
python -m tf_bodypix <sub command> --help
```

### List Available Models

```bash
python -m tf_bodypix list-models
```

The result will be a list of all of the `bodypix` TensorFlow JS models available in the [tfjs-models bucket](https://storage.googleapis.com/tfjs-models/).

Those URLs can be passed as the `--model-path` arguments below, or to the `download_model` method of the Python API.

The CLI will download and cache the model from the provided path. If no `--model-path` is provided, it will use a default model (mobilenet).

### Example commands

#### Creating a simple body mask

```bash
python -m tf_bodypix \
    draw-mask \
    --source /path/to/input-image.jpg \
    --output /path/to/output-mask.jpg \
    --threshold=0.75
```

#### Colorize the body mask depending on the body part

```bash
python -m tf_bodypix \
    draw-mask \
    --source /path/to/input-image.jpg \
    --output /path/to/output-colored-mask.jpg \
    --threshold=0.75 \
    --colored
```

#### Additionally select the body parts

```bash
python -m tf_bodypix \
    draw-mask \
    --source /path/to/input-image.jpg \
    --output /path/to/output-colored-mask.jpg \
    --threshold=0.75 \
    --parts left_face right_face \
    --colored
```

#### Capture Webcam and adding mask overlay, showing the result in an image

```bash
python -m tf_bodypix \
    draw-mask \
    --source webcam:0 \
    --show-output \
    --threshold=0.75 \
    --add-overlay-alpha=0.5 \
    --colored
```

#### Capture Webcam and adding mask overlay, writing to v4l2loopback device

(replace `/dev/videoN` with the actual virtual video device)

```bash
python -m tf_bodypix \
    draw-mask \
    --source webcam:0 \
    --output /dev/videoN \
    --threshold=0.75 \
    --add-overlay-alpha=0.5 \
    --colored
```

#### Capture Webcam and blur background, writing to v4l2loopback device

(replace `/dev/videoN` with the actual virtual video device)

```bash
python -m tf_bodypix \
    blur-background \
    --source webcam:0 \
    --background-blur 20 \
    --output /dev/videoN \
    --threshold=0.75
```

#### Capture Webcam and replace background, writing to v4l2loopback device

(replace `/dev/videoN` with the actual virtual video device)

```bash
python -m tf_bodypix \
    replace-background \
    --source webcam:0 \
    --background /path/to/background-image.jpg \
    --output /dev/videoN \
    --threshold=0.75
```

## Acknowledgements

* [Original TensorFlow JS Implementation of BodyPix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix)
* [Linux-Fake-Background-Webcam](https://github.com/fangfufu/Linux-Fake-Background-Webcam), an implementation of the [blog post](https://elder.dev/posts/open-source-virtual-background/) describing using the TensorFlow JS implementation with Python via a Socket API.
* [tfjs-to-tf](https://github.com/patlevin/tfjs-to-tf) for providing an easy way to convert TensorFlow JS models
* [virtual_webcam_background](https://github.com/allo-/virtual_webcam_background) for a great pure Python implementation
