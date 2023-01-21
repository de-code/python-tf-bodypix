import logging

import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    for name in ['tests', 'tf_bodypix']:
        logging.getLogger(name).setLevel('DEBUG')
