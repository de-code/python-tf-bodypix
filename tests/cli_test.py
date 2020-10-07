from pathlib import Path

from tf_bodypix.cli import main


EXAMPLE_IMAGE_URL = (
    r'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/'
    r'Person_Of_Interest_-_Panel_%289353656298%29.jpg/'
    r'640px-Person_Of_Interest_-_Panel_%289353656298%29.jpg'
)


class TestMain:
    def test_should_not_fail(self, temp_dir: Path):
        output_mask_path = temp_dir / 'mask.jpg'
        main([
            'image-to-mask',
            '--image=%s' % EXAMPLE_IMAGE_URL,
            '--output-mask=%s' % output_mask_path
        ])
