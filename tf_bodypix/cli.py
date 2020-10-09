import argparse
import logging
import os
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Dict, List

import tensorflow as tf

from tf_bodypix.download import download_model
from tf_bodypix.model import load_model, PART_CHANNELS


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


def get_file(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    local_path = tf.keras.utils.get_file(
        md5(file_path.encode('utf-8')).hexdigest() + '-' + os.path.basename(file_path),
        file_path
    )
    return local_path


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
        parser.add_argument(
            "--output-mask",
            required=True,
            help="The path to the output mask."
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.75,
            help="The mask threshold."
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

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        local_image_path = get_file(args.image)
        LOGGER.debug('local_image_path: %r', local_image_path)
        local_model_path = download_model(args.model_path)
        LOGGER.debug('local_model_path: %r', local_model_path)
        bodypix_model = load_model(local_model_path)
        image = tf.keras.preprocessing.image.load_img(
            local_image_path
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        result = bodypix_model.predict_single(image_array)
        mask = result.get_mask(args.threshold)
        if args.colored:
            mask = result.get_colored_part_mask(mask, part_names=args.parts)
        elif args.parts:
            mask = result.get_part_mask(mask, part_names=args.parts)
        LOGGER.info('writing mask to: %r', args.output_mask)
        os.makedirs(os.path.dirname(args.output_mask), exist_ok=True)
        tf.keras.preprocessing.image.save_img(
            args.output_mask,
            mask
        )


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
