import logging
from pathlib import Path

from tf_bodypix.download import BodyPixModelPaths
from tf_bodypix.model import ModelArchitectureNames
from tf_bodypix.cli import main


LOGGER = logging.getLogger(__name__)


EXAMPLE_IMAGE_URL = (
    r'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/'
    r'Person_Of_Interest_-_Panel_%289353656298%29.jpg/'
    r'640px-Person_Of_Interest_-_Panel_%289353656298%29.jpg'
)


EXAMPLE_BACKGROUND_IMAGE_URL = (
    r'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa'
    r'/Gold_Coast_skyline.jpg/640px-Gold_Coast_skyline.jpg'
)


class TestMain:
    def test_should_not_fail_to_draw_mask(self, temp_dir: Path):
        output_image_path = temp_dir / 'mask.jpg'
        main([
            'draw-mask',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path
        ])

    def test_should_not_fail_to_draw_selected_mask(self, temp_dir: Path):
        output_image_path = temp_dir / 'mask.jpg'
        main([
            'draw-mask',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path,
            '--parts', 'left_face', 'right_face'
        ])

    def test_should_not_fail_to_draw_colored_mask(self, temp_dir: Path):
        output_image_path = temp_dir / 'mask.jpg'
        main([
            'draw-mask',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path,
            '--colored'
        ])

    def test_should_not_fail_to_draw_selected_colored_mask(self, temp_dir: Path):
        output_image_path = temp_dir / 'mask.jpg'
        main([
            'draw-mask',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path,
            '--parts', 'left_face', 'right_face',
            '--colored'
        ])

    def test_should_not_fail_to_draw_single_person_pose(self, temp_dir: Path):
        output_image_path = temp_dir / 'output.jpg'
        main([
            'draw-pose',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path,
            '--parts', 'left_face', 'right_face'
        ])

    def test_should_not_fail_to_blur_background(self, temp_dir: Path):
        output_image_path = temp_dir / 'output.jpg'
        main([
            'blur-background',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path
        ])

    def test_should_not_fail_to_replace_background(self, temp_dir: Path):
        output_image_path = temp_dir / 'output.jpg'
        main([
            'replace-background',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--background=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path
        ])

    def test_should_list_all_default_model_urls(self, capsys):
        expected_urls = [
            value
            for key, value in BodyPixModelPaths.__dict__.items()
            if not key.startswith('_')
        ]
        main(['list-models'])
        captured = capsys.readouterr()
        output_urls = captured.out.splitlines()
        LOGGER.debug('output_urls: %s', output_urls)
        missing_urls = set(expected_urls) - set(output_urls)
        assert not missing_urls

    def test_should_be_able_to_convert_to_tflite_and_use_model(self, temp_dir: Path):
        output_model_file = temp_dir / 'model.tflite'
        main([
            'convert-to-tflite',
            '--model-path=%s' % BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16,
            '--optimize',
            '--quantization-type=int8',
            '--output-model-file=%s' % output_model_file
        ])
        output_image_path = temp_dir / 'mask.jpg'
        main([
            'draw-mask',
            '--model-path=%s' % output_model_file,
            '--model-architecture=%s' % ModelArchitectureNames.MOBILENET_V1,
            '--output-stride=16',
            '--source=%s' % EXAMPLE_IMAGE_URL,
            '--output=%s' % output_image_path
        ])
