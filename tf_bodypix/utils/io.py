import os
from hashlib import md5
from pathlib import Path
from typing import Optional

import requests

import tensorflow as tf


DEFAULT_KERAS_CACHE_DIR = '~/.keras'


def strip_url_suffix(path: str) -> str:
    qs_index = path.find('?')
    if qs_index > 0:
        return path[:qs_index]
    return path


def get_default_cache_dir(
    cache_dir: Optional[str] = None,
    cache_subdir: Optional[str] = None
):
    result = os.path.expanduser(cache_dir or DEFAULT_KERAS_CACHE_DIR)
    if cache_subdir:
        result = os.path.join(result, cache_subdir)
    return result


def download_file_to(
    source_url: str,
    local_path: str,
    skip_if_exists: bool = True
):
    if skip_if_exists and os.path.exists(local_path):
        return local_path
    response = requests.get(source_url)
    response.raise_for_status()
    Path(local_path).write_bytes(response.content)
    return local_path


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
