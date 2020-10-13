import argparse
import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from itertools import cycle
from typing import Dict, List

import tensorflow as tf
import numpy as np

from tf_bodypix.utils.timer import LoggingTimer
from tf_bodypix.utils.image import (
    ImageSize,
    resize_image_to,
    get_image_size,
    box_blur_image
)
from tf_bodypix.download import download_model
from tf_bodypix.model import load_model, PART_CHANNELS, BodyPixModelWrapper, BodyPixResultWrapper
from tf_bodypix.source import get_image_source, T_ImageSource
from tf_bodypix.sink import (
    T_OutputSink,
    get_image_output_sink_for_path,
    get_show_image_output_sink
)


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
        help="the fourcc code to select the source to, e.g. MJPG"
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
    return get_image_source(args.source, image_size=image_size, fourcc=args.source_fourcc)


def get_output_sink(args: argparse.Namespace) -> T_OutputSink:
    if args.show_output:
        return get_show_image_output_sink()
    if args.output:
        return get_image_output_sink_for_path(args.output)
    raise RuntimeError('no output sink')


def load_bodypix_model(args: argparse.Namespace) -> BodyPixModelWrapper:
    local_model_path = download_model(args.model_path)
    LOGGER.debug('local_model_path: %r', local_model_path)
    return load_model(
        local_model_path,
        internal_resolution=args.internal_resolution,
        output_stride=args.output_stride
    )


def get_mask(
    bodypix_result: BodyPixResultWrapper,
    masks: List[np.ndarray],
    timer: LoggingTimer,
    args: argparse.Namespace
) -> np.ndarray:
    mask = bodypix_result.get_mask(args.threshold, dtype=np.float32)
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


class ImageToMaskSubCommand(SubCommand):
    def __init__(self):
        super().__init__("image-to-mask", "Converts an image to its mask")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        add_model_arguments(parser)
        add_source_arguments(parser)
        add_output_arguments(parser)

        parser.add_argument(
            "--add-overlay-alpha",
            type=float,
            help="The opacity of mask overlay to add."
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

    def get_output_image(
        self,
        bodypix_model: BodyPixModelWrapper,
        image_array: np.ndarray,
        args: argparse.Namespace,
        masks: List[np.ndarray],
        timer: LoggingTimer
    ) -> np.ndarray:
        result = bodypix_model.predict_single(image_array)
        timer.on_step_start('get_mask')
        mask = get_mask(result, masks=masks, timer=timer, args=args)
        if args.colored:
            timer.on_step_start('get_cpart_mask')
            mask = result.get_colored_part_mask(mask, part_names=args.parts)
        elif args.parts:
            timer.on_step_start('get_part_mask')
            mask = result.get_part_mask(mask, part_names=args.parts)
        if args.add_overlay_alpha is not None:
            timer.on_step_start('overlay')
            LOGGER.debug('mask.shape: %s (%s)', mask.shape, mask.dtype)
            alpha = args.add_overlay_alpha
            if not args.colored:
                alpha *= 255
            try:
                if mask.dtype == tf.int32:
                    mask = tf.cast(mask, tf.float32)
            except TypeError:
                pass
            output = np.clip(
                image_array + mask * alpha,
                0.0, 255.0
            )
            return output
        return mask

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        bodypix_model = load_bodypix_model(args)
        timer = LoggingTimer()
        masks = []
        try:
            with ExitStack() as exit_stack:
                output_sink = exit_stack.enter_context(get_output_sink(args))
                image_source = exit_stack.enter_context(get_image_source_for_args(args))
                image_iterator = iter(image_source)
                timer.start()
                while True:
                    timer.on_frame_start(initial_step_name='in')
                    try:
                        image_array = next(image_iterator)
                    except StopIteration:
                        break
                    timer.on_step_start('model')
                    output_image = self.get_output_image(
                        bodypix_model,
                        image_array,
                        args,
                        masks=masks,
                        timer=timer
                    )
                    timer.on_step_start('out')
                    output_sink(output_image)
                    timer.on_frame_end()
        except KeyboardInterrupt:
            LOGGER.info('exiting')


class ReplaceBackgroundSubCommand(SubCommand):
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

    def get_output_image(
        self,
        bodypix_model: BodyPixModelWrapper,
        image_array: np.ndarray,
        args: argparse.Namespace,
        masks: List[np.ndarray],
        background_image_array: np.ndarray,
        timer: LoggingTimer
    ) -> np.ndarray:
        result = bodypix_model.predict_single(image_array)
        timer.on_step_start('get_mask')
        mask = get_mask(result, masks=masks, timer=timer, args=args)
        timer.on_step_start('compose')
        background_image_array = resize_image_to(
            background_image_array, get_image_size(image_array)
        )
        output = np.clip(
            background_image_array * (1 - mask)
            + image_array * mask,
            0.0, 255.0
        )
        return output

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        bodypix_model = load_bodypix_model(args)
        timer = LoggingTimer()
        masks = []
        try:
            background_image_iterator = None
            with ExitStack() as exit_stack:
                output_sink = exit_stack.enter_context(get_output_sink(args))
                image_source = exit_stack.enter_context(get_image_source_for_args(args))
                image_iterator = iter(image_source)
                timer.start()
                while True:
                    timer.on_frame_start(initial_step_name='in')
                    try:
                        image_array = next(image_iterator)
                    except StopIteration:
                        break
                    timer.on_frame_start(initial_step_name='bg')
                    if background_image_iterator is None:
                        background_image_source = exit_stack.enter_context(get_image_source(
                            args.background,
                            image_size=get_image_size(image_array)
                        ))
                        background_image_iterator = iter(cycle(background_image_source))
                    background_image_array = next(background_image_iterator)
                    timer.on_step_start('model')
                    output_image = self.get_output_image(
                        bodypix_model,
                        image_array,
                        args,
                        masks=masks,
                        background_image_array=background_image_array,
                        timer=timer
                    )
                    timer.on_step_start('out')
                    output_sink(output_image)
                    timer.on_frame_end()
        except KeyboardInterrupt:
            LOGGER.info('exiting')


SUB_COMMANDS: List[SubCommand] = [
    ImageToMaskSubCommand(),
    ReplaceBackgroundSubCommand()
]

SUB_COMMAND_BY_NAME: Dict[str, SubCommand] = {
    sub_command.name: sub_command for sub_command in SUB_COMMANDS
}


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    for sub_command in SUB_COMMANDS:
        sub_parser = subparsers.add_parser(
            sub_command.name, help=sub_command.description
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
