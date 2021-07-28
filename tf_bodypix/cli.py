import argparse
import logging
import os
import re
from abc import ABC, abstractmethod
from contextlib import ExitStack
from itertools import cycle
from pathlib import Path
from time import time, sleep
from typing import Dict, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# pylint: disable=wrong-import-position
# flake8: noqa: E402

import tensorflow as tf
import numpy as np

from tf_bodypix.utils.timer import LoggingTimer
from tf_bodypix.utils.image import (
    ImageSize,
    resize_image_to,
    get_image_size,
    box_blur_image
)
from tf_bodypix.utils.s3 import iter_s3_file_urls
from tf_bodypix.download import download_model
from tf_bodypix.tflite import get_tflite_converter_for_model_path
from tf_bodypix.model import (
    load_model,
    VALID_MODEL_ARCHITECTURE_NAMES,
    PART_CHANNELS,
    DEFAULT_RESIZE_METHOD,
    BodyPixModelWrapper,
    BodyPixResultWrapper
)
from tf_bodypix.source import get_image_source, get_threaded_image_source, T_ImageSource
from tf_bodypix.sink import (
    T_OutputSink,
    get_image_output_sink_for_path,
    get_show_image_output_sink
)
try:
    from tf_bodypix.draw import draw_poses
except ImportError as exc:
    _draw_import_exc = exc
    def draw_poses(*_, **__):  # type: ignore
        raise _draw_import_exc


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = (
    r'https://storage.googleapis.com/tfjs-models/savedmodel/'
    r'bodypix/mobilenet/float/050/model-stride16.json'
)


class SubCommand(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace):
        pass


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="The path or URL to the bodypix model."
    )
    parser.add_argument(
        "--model-architecture",
        choices=VALID_MODEL_ARCHITECTURE_NAMES,
        help=(
            "The model architecture."
            " It will be guessed from the model path if not specified."
        )
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        help=(
            "The output stride to use."
            " It will be guessed from the model path if not specified."
        )
    )
    parser.add_argument(
        "--internal-resolution",
        type=float,
        default=0.5,
        help=(
            "The internal resolution factor to resize the input image to"
            " before passing it the model."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="The mask threshold."
    )
    parser.add_argument(
        "--mask-blur",
        type=int,
        default=0,
        help="The blur radius for the mask."
    )
    parser.add_argument(
        "--mask-mean-count",
        type=int,
        default=0,
        help="The number of masks to average to smooth the results."
    )
    parser.add_argument(
        "--mask-cache-time",
        type=float,
        default=0,
        help=(
            "For how long, in seconds, the mask model result should be cached."
            " e.g. if the model is very slow, you could let it calculate every second only."
            " of course that would be visible when moving quickly"
        )
    )


def _fourcc_type(text: str) -> str:
    if not text:
        return text
    if len(text) != 4:
        raise TypeError(
            'fourcc code must have exactly four characters, e.g. MJPG; but was: %r' % text
        )
    return text


def add_source_arguments(parser: argparse.ArgumentParser):
    source_group = parser.add_argument_group('source')
    source_group.add_argument(
        "--source",
        required=True,
        help="The path or URL to the source image or webcam source."
    )
    image_size_help = (
        "If width and height are specified, the source will be resized."
        "In the case of the webcam, it will be asked to produce that resolution if possible"
    )
    source_group.add_argument(
        "--source-width",
        type=int,
        help=image_size_help
    )
    source_group.add_argument(
        "--source-height",
        type=int,
        help=image_size_help
    )
    source_group.add_argument(
        "--source-fourcc",
        type=_fourcc_type,
        default="MJPG",
        help="The fourcc code to select the source to, e.g. MJPG"
    )
    source_group.add_argument(
        "--source-fps",
        type=int,
        default=None,
        help=(
            "Limit the source frame rate to desired FPS."
            " If provided, it will attempt to set the frame rate on the source device if supported."
            " Otherwise it will slow down the frame rate."
            " Use '0' for a fast as possible fps."
        )
    )
    source_group.add_argument(
        "--source-threaded",
        action='store_true',
        help="if set, will read from the source in a thread (experimental)."
    )


def add_output_arguments(parser: argparse.ArgumentParser):
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--show-output",
        action="store_true",
        help="Shows the output in a window."
    )
    output_group.add_argument(
        "--output",
        help="The path to the output file."
    )


def get_image_source_for_args(args: argparse.Namespace) -> T_ImageSource:
    image_size = None
    if args.source_width and args.source_height:
        image_size = ImageSize(height=args.source_height, width=args.source_width)
    image_source = get_image_source(
        args.source,
        image_size=image_size,
        fourcc=args.source_fourcc,
        fps=args.source_fps
    )
    if args.source_threaded:
        return get_threaded_image_source(image_source)
    return image_source


def get_output_sink(args: argparse.Namespace) -> T_OutputSink:
    if args.show_output:
        return get_show_image_output_sink()
    if args.output:
        return get_image_output_sink_for_path(args.output)
    raise RuntimeError('no output sink')


def load_bodypix_model(args: argparse.Namespace) -> BodyPixModelWrapper:
    local_model_path = download_model(args.model_path)
    if args.model_path != local_model_path:
        LOGGER.info('loading model: %r (downloaded from %r)', local_model_path, args.model_path)
    else:
        LOGGER.info('loading model: %r', local_model_path)
    return load_model(
        local_model_path,
        internal_resolution=args.internal_resolution,
        output_stride=args.output_stride,
        architecture_name=args.model_architecture
    )


def get_mask(
    bodypix_result: BodyPixResultWrapper,
    masks: List[np.ndarray],
    timer: LoggingTimer,
    args: argparse.Namespace,
    resize_method: str = DEFAULT_RESIZE_METHOD
) -> np.ndarray:
    mask = bodypix_result.get_mask(args.threshold, dtype=np.float32, resize_method=resize_method)
    if args.mask_blur:
        timer.on_step_start('mblur')
        mask = box_blur_image(mask, args.mask_blur)
    if args.mask_mean_count >= 2:
        timer.on_step_start('mmean')
        masks.append(mask)
        if len(masks) > args.mask_mean_count:
            masks.pop(0)
        if len(masks) >= 2:
            mask = np.mean(masks, axis=0)
    LOGGER.debug('mask.shape: %s (%s)', mask.shape, mask.dtype)
    return mask


class ListModelsSubCommand(SubCommand):
    def __init__(self):
        super().__init__("list-models", "Lists available bodypix models (original models)")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        parser.add_argument(
            "--storage-url",
            default="https://storage.googleapis.com/tfjs-models",
            help="The base URL for the storage containing the models"
        )

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        bodypix_model_json_files = [
            file_url
            for file_url in iter_s3_file_urls(args.storage_url)
            if re.match(r'.*/bodypix/.*/model.*\.json', file_url)
        ]
        print('\n'.join(bodypix_model_json_files))


class ConvertToTFLiteSubCommand(SubCommand):
    def __init__(self):
        super().__init__("convert-to-tflite", "Converts the model to a tflite model")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        parser.add_argument(
            "--model-path",
            default=DEFAULT_MODEL_PATH,
            help="The path or URL to the bodypix model."
        )
        parser.add_argument(
            "--output-model-file",
            required=True,
            help="The path to the output file (tflite model)."
        )
        parser.add_argument(
            "--optimize",
            action='store_true',
            help="Enable optimization (quantization)."
        )
        parser.add_argument(
            "--quantization-type",
            choices=['float16', 'float32', 'int8'],
            help="The quantization type to use."
        )

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        LOGGER.info('converting model: %s', args.model_path)
        converter = get_tflite_converter_for_model_path(download_model(
            args.model_path
        ))
        tflite_model = converter.convert()
        if args.optimize:
            LOGGER.info('enabled optimization')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.quantization_type:
            LOGGER.info('quanization type: %s', args.quantization_type)
            quantization_type = getattr(tf, args.quantization_type)
            converter.target_spec.supported_types = [quantization_type]
            converter.inference_input_type = quantization_type
            converter.inference_output_type = quantization_type
        LOGGER.info('saving tflite model to: %s', args.output_model_file)
        Path(args.output_model_file).write_bytes(tflite_model)


class AbstractWebcamFilterApp(ABC):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bodypix_model = None
        self.output_sink = None
        self.image_source = None
        self.image_iterator = None
        self.timer = LoggingTimer()
        self.masks: List[np.ndarray] = []
        self.exit_stack = ExitStack()
        self.bodypix_result_cache_time = None
        self.bodypix_result_cache = None

    @abstractmethod
    def get_output_image(self, image_array: np.ndarray) -> np.ndarray:
        pass

    def get_mask(self, *args, **kwargs):
        return get_mask(
            *args, masks=self.masks, timer=self.timer, args=self.args, **kwargs
        )

    def get_bodypix_result(self, image_array: np.ndarray) -> BodyPixResultWrapper:
        assert self.bodypix_model is not None
        current_time = time()
        if (
            self.bodypix_result_cache is not None
            and current_time < self.bodypix_result_cache_time + self.args.mask_cache_time
        ):
            return self.bodypix_result_cache
        self.bodypix_result_cache = self.bodypix_model.predict_single(image_array)
        self.bodypix_result_cache_time = current_time
        return self.bodypix_result_cache

    def __enter__(self):
        self.exit_stack.__enter__()
        self.bodypix_model = load_bodypix_model(self.args)
        self.output_sink = self.exit_stack.enter_context(get_output_sink(self.args))
        self.image_source = self.exit_stack.enter_context(get_image_source_for_args(self.args))
        self.image_iterator = iter(self.image_source)
        return self

    def __exit__(self, *args, **kwargs):
        self.exit_stack.__exit__(*args, **kwargs)

    def next_frame(self):
        self.timer.on_frame_start(initial_step_name='in')
        try:
            image_array = next(self.image_iterator)
        except StopIteration:
            return False
        self.timer.on_step_start('model')
        output_image = self.get_output_image(image_array)
        self.timer.on_step_start('out')
        self.output_sink(output_image)
        self.timer.on_frame_end()
        return True

    def run(self):
        try:
            self.timer.start()
            while self.next_frame():
                pass
            if self.args.show_output:
                LOGGER.info('waiting for window to be closed')
                while not self.output_sink.is_closed:
                    sleep(0.5)
        except KeyboardInterrupt:
            LOGGER.info('exiting')


class AbstractWebcamFilterSubCommand(SubCommand):
    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_arguments(parser)
        add_source_arguments(parser)
        add_output_arguments(parser)

    @abstractmethod
    def get_app(self, args: argparse.Namespace) -> AbstractWebcamFilterApp:
        pass

    def run(self, args: argparse.Namespace):
        with self.get_app(args) as app:
            app.run()


class DrawMaskApp(AbstractWebcamFilterApp):
    def get_output_image(self, image_array: np.ndarray) -> np.ndarray:
        resize_method = DEFAULT_RESIZE_METHOD
        result = self.get_bodypix_result(image_array)
        self.timer.on_step_start('get_mask')
        mask = self.get_mask(result, resize_method=resize_method)
        if self.args.colored:
            self.timer.on_step_start('get_cpart_mask')
            mask_image = result.get_colored_part_mask(
                mask, part_names=self.args.parts, resize_method=resize_method
            )
        elif self.args.parts:
            self.timer.on_step_start('get_part_mask')
            mask_image = result.get_part_mask(
                mask, part_names=self.args.parts, resize_method=resize_method
            ) * 255
        else:
            mask_image = mask * 255
        if self.args.mask_alpha is not None:
            self.timer.on_step_start('overlay')
            LOGGER.debug('mask.shape: %s (%s)', mask.shape, mask.dtype)
            alpha = self.args.mask_alpha
            try:
                if mask_image.dtype == tf.int32:
                    mask_image = tf.cast(mask, tf.float32)
            except TypeError:
                pass
            output = np.clip(
                image_array * (1 - alpha) + mask_image * alpha,
                0.0, 255.0
            )
            return output
        return mask_image


class DrawMaskSubCommand(AbstractWebcamFilterSubCommand):
    def __init__(self):
        super().__init__("draw-mask", "Draws the mask for the input")

    def add_arguments(self, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument(
            "--mask-alpha",
            type=float,
            help="The opacity of mask overlay to add."
        )
        parser.add_argument(
            "--add-overlay-alpha",
            dest='mask_alpha',
            type=float,
            help="Deprecated, please use --mask-alpha instead."
        )
        parser.add_argument(
            "--colored",
            action="store_true",
            help="Enable generating the colored part mask"
        )
        parser.add_argument(
            "--parts",
            nargs="*",
            choices=PART_CHANNELS,
            help="Select the parts to output"
        )

    def get_app(self, args: argparse.Namespace) -> AbstractWebcamFilterApp:
        return DrawMaskApp(args)


class DrawPoseApp(AbstractWebcamFilterApp):
    def get_output_image(self, image_array: np.ndarray) -> np.ndarray:
        result = self.get_bodypix_result(image_array)
        self.timer.on_step_start('get_pose')
        poses = result.get_poses()
        LOGGER.debug('number of poses: %d', len(poses))
        output_image = draw_poses(
            image_array.copy(), poses,
            keypoints_color=(255, 100, 100),
            skeleton_color=(100, 100, 255)
        )
        return output_image


class DrawPoseSubCommand(AbstractWebcamFilterSubCommand):
    def __init__(self):
        super().__init__("draw-pose", "Draws the pose estimation")

    def get_app(self, args: argparse.Namespace) -> AbstractWebcamFilterApp:
        return DrawPoseApp(args)


class BlurBackgroundApp(AbstractWebcamFilterApp):
    def get_output_image(self, image_array: np.ndarray) -> np.ndarray:
        result = self.get_bodypix_result(image_array)
        self.timer.on_step_start('get_mask')
        mask = self.get_mask(result)
        self.timer.on_step_start('bblur')
        background_image_array = box_blur_image(image_array, self.args.background_blur)
        self.timer.on_step_start('compose')
        output = np.clip(
            background_image_array * (1 - mask)
            + image_array * mask,
            0.0, 255.0
        )
        return output


class BlurBackgroundSubCommand(AbstractWebcamFilterSubCommand):
    def __init__(self):
        super().__init__("blur-background", "Blurs the background of the webcam image")

    def add_arguments(self, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument(
            "--background-blur",
            type=int,
            default=15,
            help="The blur radius for the background."
        )

    def get_app(self, args: argparse.Namespace) -> AbstractWebcamFilterApp:
        return BlurBackgroundApp(args)


class ReplaceBackgroundApp(AbstractWebcamFilterApp):
    def __init__(self, *args, **kwargs):
        self.background_image_iterator = None
        super().__init__(*args, **kwargs)

    def get_next_background_image(self, image_array: np.ndarray) -> np.ndarray:
        if self.background_image_iterator is None:
            background_image_source = self.exit_stack.enter_context(get_image_source(
                self.args.background,
                image_size=get_image_size(image_array)
            ))
            self.background_image_iterator = iter(cycle(background_image_source))
        return next(self.background_image_iterator)

    def get_output_image(self, image_array: np.ndarray) -> np.ndarray:
        background_image_array = self.get_next_background_image(image_array)
        result = self.get_bodypix_result(image_array)
        self.timer.on_step_start('get_mask')
        mask = self.get_mask(result)
        self.timer.on_step_start('compose')
        background_image_array = resize_image_to(
            background_image_array, get_image_size(image_array)
        )
        output = np.clip(
            background_image_array * (1 - mask)
            + image_array * mask,
            0.0, 255.0
        )
        return output


class ReplaceBackgroundSubCommand(AbstractWebcamFilterSubCommand):
    def __init__(self):
        super().__init__("replace-background", "Replaces the background of a person")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_arguments(parser)
        add_source_arguments(parser)

        parser.add_argument(
            "--background",
            required=True,
            help="The path or URL to the background image."
        )

        add_output_arguments(parser)

    def get_app(self, args: argparse.Namespace) -> AbstractWebcamFilterApp:
        return ReplaceBackgroundApp(args)


SUB_COMMANDS: List[SubCommand] = [
    ListModelsSubCommand(),
    ConvertToTFLiteSubCommand(),
    DrawMaskSubCommand(),
    DrawPoseSubCommand(),
    BlurBackgroundSubCommand(),
    ReplaceBackgroundSubCommand()
]

SUB_COMMAND_BY_NAME: Dict[str, SubCommand] = {
    sub_command.name: sub_command for sub_command in SUB_COMMANDS
}


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'TensorFlow BodyPix (TF BodyPix)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    for sub_command in SUB_COMMANDS:
        sub_parser = subparsers.add_parser(
            sub_command.name, help=sub_command.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        sub_command.add_arguments(sub_parser)

    args = parser.parse_args(argv)
    return args


def run(args: argparse.Namespace):
    sub_command = SUB_COMMAND_BY_NAME[args.command]
    sub_command.run(args)


def main(argv: List[str] = None):
    args = parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    LOGGER.debug("args: %s", args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
