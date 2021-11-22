import logging
import json
import os
import re

from hashlib import md5

from tf_bodypix.utils.io import download_file_to, get_default_cache_dir


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

    RESNET50_FLOAT_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'resnet50/float/model-stride16.json'
    )
    RESNET50_FLOAT_STRIDE_32 = (
        _DOWNLOAD_URL_PREFIX + 'resnet50/float/model-stride32.json'
    )

    # deprecated (shouldn't have mobilenet in the name)
    MOBILENET_RESNET50_FLOAT_STRIDE_16 = (
        _DOWNLOAD_URL_PREFIX + 'resnet50/float/model-stride16.json'
    )
    MOBILENET_RESNET50_FLOAT_STRIDE_32 = (
        _DOWNLOAD_URL_PREFIX + 'resnet50/float/model-stride32.json'
    )


class DownloadError(RuntimeError):
    pass


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
    cache_dir = get_default_cache_dir(
        cache_subdir=os.path.join('tf-bodypix', local_name)
    )
    local_model_json_path = download_file_to(
        source_url=model_path,
        local_path=os.path.join(cache_dir, 'model.json'),
        skip_if_exists=True
    )
    local_model_path = os.path.dirname(local_model_json_path)
    LOGGER.debug('local_model_json_path: %r', local_model_json_path)
    try:
        with open(local_model_json_path, 'r', encoding='utf-8') as model_json_fp:
            model_json = json.load(model_json_fp)
    except UnicodeDecodeError as exc:
        LOGGER.error(
            'failed to process %r due to %r',
            local_model_json_path, exc, exc_info=True
        )
        raise DownloadError(
            'failed to process %r due to %r' % (
                local_model_json_path, exc
            )
        ) from exc
    LOGGER.debug('model_json.keys: %s', model_json.keys())
    weights_manifest = model_json['weightsManifest']
    weights_manifest_paths = sorted({
        path
        for item in weights_manifest
        for path in item.get('paths', [])
    })
    LOGGER.debug('weights_manifest_paths: %s', weights_manifest_paths)
    for weights_manifest_path in weights_manifest_paths:
        local_model_json_path = download_file_to(
            source_url=model_base_path + '/' + weights_manifest_path,
            local_path=os.path.join(cache_dir, os.path.basename(weights_manifest_path)),
            skip_if_exists=True
        )
    return local_model_path
