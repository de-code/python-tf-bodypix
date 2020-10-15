from pathlib import Path

from tf_bodypix.cli import main


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
