import argparse
import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Dict, List

import tensorflow as tf
import numpy as np

from tf_bodypix.utils.timer import LoggingTimer
from tf_bodypix.download import download_model
from tf_bodypix.model import load_model, PART_CHANNELS, BodyPixModelWrapper
from tf_bodypix.source import get_image_source
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


class ImageToMaskSubCommand(SubCommand):
    def __init__(self):
        super().__init__("image-to-mask", "Converts an image to its mask")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        parser.add_argument(
            "--image",
            required=True,
            help="The path or URL to the source image."
        )
        parser.add_argument(
            "--model-path",
            default=DEFAULT_MODEL_PATH,
            help="The path or URL to the bodypix model."
        )

        output_group = parser.add_mutually_exclusive_group(required=True)
        output_group.add_argument(
            "--show-output",
            action="store_true",
            help="Shows the output in a window."
        )
        output_group.add_argument(
            "--output-mask",
            help="The path to the output mask."
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

    def get_output_sink(self, args: argparse.Namespace) -> T_OutputSink:
        if args.show_output:
            return get_show_image_output_sink()
        if args.output_mask:
            return get_image_output_sink_for_path(args.output_mask)
        raise RuntimeError('no output sink')

    def get_output_image(
        self,
        bodypix_model: BodyPixModelWrapper,
        image_array: np.ndarray,
        args: argparse.Namespace,
        timer: LoggingTimer
    ) -> np.ndarray:
        result = bodypix_model.predict_single(image_array)
        timer.on_step_start('get_mask')
        mask = result.get_mask(args.threshold)
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
        local_model_path = download_model(args.model_path)
        LOGGER.debug('local_model_path: %r', local_model_path)
        bodypix_model = load_model(
            local_model_path,
            internal_resolution=args.internal_resolution,
            output_stride=args.output_stride
        )
        timer = LoggingTimer()
        try:
            with ExitStack() as exit_stack:
                output_sink = exit_stack.enter_context(self.get_output_sink(args))
                image_source = exit_stack.enter_context(get_image_source(args.image))
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
                        timer=timer
                    )
                    timer.on_step_start('out')
                    output_sink(output_image)
                    timer.on_frame_end()
        except KeyboardInterrupt:
            LOGGER.info('exiting')


SUB_COMMANDS: List[SubCommand] = [
    ImageToMaskSubCommand()
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
