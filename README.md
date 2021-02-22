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
| tf         | [TensorFlow](https://pypi.org/project/tensorflow/) (required). But you may use your own build.
| tfjs       | TensorFlow JS Model support, using [tfjs-graph-converter](https://pypi.org/project/tfjs-graph-converter/)
| image      | Image loading via [Pillow](https://pypi.org/project/Pillow/), required by the CLI.
| video      | Video support via [OpenCV](https://pypi.org/project/opencv-python/)
| webcam     | Webcam support via [OpenCV](https://pypi.org/project/opencv-python/) and [pyfakewebcam](https://pypi.org/project/pyfakewebcam/)
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

### Inputs and Outputs

Most commands will work with inputs (source) and outputs.

The source path can be specified via the `--source` parameter.

The following inputs are supported:

| type | description |
| -----| ----------- |
| image | Static image (e.g. `.png`) |
| video | Video (e.g. `.mp4`) |
| webcam | Linux Webcam (`/dev/videoN` or `webcam:0`) |

If the source path points to an external file (e.g. `https://`), then it will be downloaded and locally cached.

The output path can be specified via `--output`, unless `--show-output` is used.

The following outpus are supported:

| type | description |
| -----| ----------- |
| image_writer | Write to a static image (e.g. `.png`) |
| v4l2 | Linux Virtual Webcam (`/dev/videoN`) |
| window | Display a window (by using `--show-output`) |

### Example commands

#### Creating a simple body mask

```bash
python -m tf_bodypix \
    draw-mask \
    --source \
    "https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1" \
    --show-output \
    --threshold=0.75
```

Image Source: [Serious black businesswoman sitting at desk in office](https://www.pexels.com/photo/serious-black-businesswoman-sitting-at-desk-in-office-5669603/)

#### Add the mask over the original image using `--mask-alpha`

```bash
python -m tf_bodypix \
    draw-mask \
    --source \
    "https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-alpha=0.5
```

Image Source: [Serious black businesswoman sitting at desk in office](https://www.pexels.com/photo/serious-black-businesswoman-sitting-at-desk-in-office-5669603/)

#### Colorize the body mask depending on the body part

```bash
python -m tf_bodypix \
    draw-mask \
    --source \
    "https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-alpha=0.5 \
    --colored
```

Image Source: [Serious black businesswoman sitting at desk in office](https://www.pexels.com/photo/serious-black-businesswoman-sitting-at-desk-in-office-5669603/)

#### Additionally select the body parts

```bash
python -m tf_bodypix \
    draw-mask \
    --source \
    "https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-alpha=0.5 \
    --parts left_face right_face \
    --colored
```

Image Source: [Serious black businesswoman sitting at desk in office](https://www.pexels.com/photo/serious-black-businesswoman-sitting-at-desk-in-office-5669603/)

#### Add mask overlay to a video

```bash
python -m tf_bodypix \
    draw-mask \
    --source \
    "https://www.dropbox.com/s/s7jga3f0dreavlb/video-of-a-man-laughing-and-happy-1608393-360p.mp4?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-alpha=0.5 \
    --colored
```

Video Source: [Video Of A Man Laughing And Happy](https://www.pexels.com/video/video-of-a-man-laughing-and-happy-1608393/)

#### Blur background of a video

```bash
python -m tf_bodypix \
    blur-background \
    --source \
    "https://www.dropbox.com/s/s7jga3f0dreavlb/video-of-a-man-laughing-and-happy-1608393-360p.mp4?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-blur=5 \
    --background-blur=20
```

Video Source: [Video Of A Man Laughing And Happy](https://www.pexels.com/video/video-of-a-man-laughing-and-happy-1608393/)

#### Replace the background of a video

```bash
python -m tf_bodypix \
    replace-background \
    --source \
    "https://www.dropbox.com/s/s7jga3f0dreavlb/video-of-a-man-laughing-and-happy-1608393-360p.mp4?dl=1" \
    --background \
    "https://www.dropbox.com/s/b22ss59j6pp83zy/brown-landscape-under-grey-sky-3244513.jpg?dl=1" \
    --show-output \
    --threshold=0.75 \
    --mask-blur=5
```

Video Source: [Video Of A Man Laughing And Happy](https://www.pexels.com/video/video-of-a-man-laughing-and-happy-1608393/)

Background: [Brown Landscape Under Grey Sky](https://www.pexels.com/photo/brown-landscape-under-grey-sky-3244513/)

#### Capture Webcam and adding mask overlay

```bash
python -m tf_bodypix \
    draw-mask \
    --source webcam:0 \
    --show-output \
    --threshold=0.75 \
    --mask-alpha=0.5 \
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
    --mask-alpha=0.5 \
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
    --background \
    "https://www.dropbox.com/s/b22ss59j6pp83zy/brown-landscape-under-grey-sky-3244513.jpg?dl=1" \
    --threshold=0.75 \
    --output /dev/videoN
```

Background: [Brown Landscape Under Grey Sky](https://www.pexels.com/photo/brown-landscape-under-grey-sky-3244513/)

## TensorFlow Lite support (experimental)

The model path may also point to a TensorFlow Lite model (`.tflite` extension). Whether that actually improves performance may depend on the platform and available hardware.

You could convert one of the available TensorFlow JS models to TensorFlow Lite using the following command:

```bash
python -m tf_bodypix \
    convert-to-tflite \
    --model-path \
    "https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/075/model-stride16.json" \
    --optimize \
    --quantization-type=float16 \
    --output-model-file "./mobilenet-float16-stride16.tflite"
```

The above command is provided for convenience.
You may use alternative methods depending on your preference and requirements.

Relevant links:

* [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/)
* [TF Lite post_training_quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
* [TF GitHub #40183](https://github.com/tensorflow/tensorflow/issues/40183).

## Docker Usage

You could also use the Docker image if you prefer.
The entrypoint will by default delegate to the CLI, except for `python` or `bash` commands.

```bash
docker pull de4code/tf-bodypix

docker run --rm \
    --device /dev/video0 \
    --device /dev/video2 \
    de4code/tf-bodypix \
    blur-background \
    --source /dev/video0 \
    --output /dev/video2 \
    --background-blur 20 \
    --threshold=0.75
```

## Example Media

Here are a few example media files you could try.

Images:

* [Serious black businesswoman sitting at desk in office](https://www.dropbox.com/s/7tsaqgdp149d8aj/serious-black-businesswoman-sitting-at-desk-in-office-5669603.jpg?dl=1) ([Source](https://www.pexels.com/photo/serious-black-businesswoman-sitting-at-desk-in-office-5669603/))
* [Woman Wearing Gray Notch Lapel Suit Jacket](https://www.dropbox.com/s/ygfudebvbm1pksk/woman-wearing-gray-notch-lapel-suit-jacket-2381069-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/woman-wearing-gray-notch-lapel-suit-jacket-2381069/))
* [Smiling Woman Standing In Front Of A Colorful Flag](https://www.dropbox.com/s/ddyj89vkz7cmzmg/smiling-woman-standing-in-front-of-a-colorful-flag-5255422-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/smiling-woman-standing-in-front-of-a-colorful-flag-5255422/))
* [Man and Woman Smiling Inside Building](https://www.dropbox.com/s/5z7v5wtwx3dmrdu/man-and-woman-smiling-inside-building-1367269-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/man-and-woman-smiling-inside-building-1367269/))
* [Two Woman in Black Sits on Chair Near Table](https://www.dropbox.com/s/dq9e2dv86qd9ror/two-woman-in-black-sits-on-chair-near-table-1181605-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/two-woman-in-black-sits-on-chair-near-table-1181605/))
* [Female barista in beanie and apron resting chin on had](https://www.dropbox.com/s/88qb3yldsb4l2id/female-barista-in-beanie-and-apron-resting-chin-on-had-4350057-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/female-barista-in-beanie-and-apron-resting-chin-on-had-4350057/))
* [Smiling Woman Holding White Android Smartphone While Sitting Front of Table](https://www.dropbox.com/s/43awel6e1mxja5v/smiling-woman-holding-white-android-smartphone-while-sitting-front-of-table-1462631-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/smiling-woman-holding-white-android-smartphone-while-sitting-front-of-table-1462631/))
* [Woman Having Coffee and Rice Bowl](https://www.dropbox.com/s/zndltp65n93poy2/woman-having-coffee-and-rice-bowl-4058316-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/woman-having-coffee-and-rice-bowl-4058316/))
* [Woman Smiling While Holding a Coffee Cup](https://www.dropbox.com/s/0txws4j79o9hewr/woman-smiling-while-holding-a-coffee-cup-6787913-small.jpg?dl=1) ([Source](https://www.pexels.com/photo/woman-smiling-while-holding-a-coffee-cup-6787913/))

Videos:

* [Video Of A Man Laughing And Happy](https://www.dropbox.com/s/s7jga3f0dreavlb/video-of-a-man-laughing-and-happy-1608393-360p.mp4?dl=1) ([Source](https://www.pexels.com/video/video-of-a-man-laughing-and-happy-1608393/))

Background:

* [Brown Landscape Under Grey Sky](https://www.dropbox.com/s/b22ss59j6pp83zy/brown-landscape-under-grey-sky-3244513.jpg?dl=1) ([Source](https://www.pexels.com/photo/brown-landscape-under-grey-sky-3244513/))

## Experimental Downstream Projects

* [Layered Vision](https://github.com/de-code/layered-vision) is an experimental project using the `tf-bodypix` Python API.

## Acknowledgements

* [Original TensorFlow JS Implementation of BodyPix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix)
* [Linux-Fake-Background-Webcam](https://github.com/fangfufu/Linux-Fake-Background-Webcam), an implementation of the [blog post](https://elder.dev/posts/open-source-virtual-background/) describing using the TensorFlow JS implementation with Python via a Socket API.
* [tfjs-to-tf](https://github.com/patlevin/tfjs-to-tf) for providing an easy way to convert TensorFlow JS models
* [virtual_webcam_background](https://github.com/allo-/virtual_webcam_background) for a great pure Python implementation
