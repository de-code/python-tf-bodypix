import logging
import json
import os
import re

from hashlib import md5

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


_DOWNLOAD_URL_PREFIX = r'https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/'


class BodyPixModelPaths:
    MOBILENET_FLOAT_50_STRIDE_8 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/050/model-stride8.json'
    )
    MOBILENET_FLOAT_50_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/050/model-stride16.json'
    )
    MOBILENET_FLOAT_75_STRIDE_8 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/075/model-stride8.json'
    )
    MOBILENET_FLOAT_75_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/075/model-stride16.json'
    )
    MOBILENET_FLOAT_100_STRIDE_8 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/100/model-stride8.json'
    )
    MOBILENET_FLOAT_100_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'mobilenet/float/100/model-stride16.json'
    )
    MOBILENET_RESNET50_FLOAT_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'savedmodel/bodypix/resnet50/float/model-stride16.json'
    )
    MOBILENET_RESNET50_FLOAT_STRIDE_32 = (
        _DOWNLOAD_URL_PREFIX + 'savedmodel/bodypix/resnet50/float/model-stride32.json'
    )


def download_model(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith('.json'):
        raise ValueError('remote model path needs to end with .json')
    model_base_path = os.path.dirname(model_path)
    local_name_part = re.sub(
        r'[^a-zA-Z0-9]+',
        r'-',
        os.path.splitext(model_path)[0]
    )
    local_name = (
        md5(model_path.encode('utf-8')).hexdigest() + '-'
        + os.path.basename(local_name_part)
    )
    LOGGER.debug('local_name: %r', local_name)
    cache_subdir = os.path.join('tf-bodypix', local_name)
    local_model_json_path = tf.keras.utils.get_file(
        'model.json',
        model_path,
        cache_subdir=cache_subdir,
    )
    local_model_path = os.path.dirname(local_model_json_path)
    LOGGER.debug('local_model_json_path: %r', local_model_json_path)
    with open(local_model_json_path, 'r') as model_json_fp:
        model_json = json.load(model_json_fp)
    LOGGER.debug('model_json.keys: %s', model_json.keys())
    weights_manifest = model_json['weightsManifest']
    weights_manifest_paths = sorted({
        path
        for item in weights_manifest
        for path in item.get('paths', [])
    })
    LOGGER.debug('weights_manifest_paths: %s', weights_manifest_paths)
    for weights_manifest_path in weights_manifest_paths:
        local_model_json_path = tf.keras.utils.get_file(
            os.path.basename(weights_manifest_path),
            os.path.join(model_base_path, weights_manifest_path),
            cache_subdir=cache_subdir,
        )
    return local_model_path
