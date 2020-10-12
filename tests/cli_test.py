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
    def test_should_not_fail_converting_to_mask(self, temp_dir: Path):
        output_mask_path = temp_dir / 'mask.jpg'
        main([
            'image-to-mask',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_mask_path
        ])

    def test_should_not_fail_converting_to_selected_mask(self, temp_dir: Path):
        output_mask_path = temp_dir / 'mask.jpg'
        main([
            'image-to-mask',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_mask_path,
            '--parts', 'left_face', 'right_face'
        ])

    def test_should_not_fail_converting_to_colored_mask(self, temp_dir: Path):
        output_mask_path = temp_dir / 'mask.jpg'
        main([
            'image-to-mask',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_mask_path,
            '--colored'
        ])

    def test_should_not_fail_converting_to_selected_colored_mask(self, temp_dir: Path):
        output_mask_path = temp_dir / 'mask.jpg'
        main([
            'image-to-mask',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_mask_path,
            '--parts', 'left_face', 'right_face',
            '--colored'
        ])

    def test_should_not_fail_replacing_background(self, temp_dir: Path):
        output_image_path = temp_dir / 'output.jpg'
        main([
            'replace-background',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--background=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_image_path
        ])
