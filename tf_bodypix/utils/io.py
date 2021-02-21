import os
from hashlib import md5

import tensorflow as tf


def strip_url_suffix(path: str) -> str:
    qs_index = path.find('?')
    if qs_index > 0:
        return path[:qs_index]
    return path


def get_file(file_path: str, download: bool = True) -> str:
    if not download:
        return file_path
    if os.path.exists(file_path):
        return file_path
    local_path = tf.keras.utils.get_file(
        (
            md5(file_path.encode('utf-8')).hexdigest()
            + '-'
            + os.path.basename(strip_url_suffix(file_path))
        ),
        file_path
    )
    return local_path
