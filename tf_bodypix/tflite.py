import logging

import tensorflow as tf

try:
    import tfjs_graph_converter
except ImportError:
    tfjs_graph_converter = None


LOGGER = logging.getLogger(__name__)


def get_tflite_converter_for_tfjs_model_path(model_path: str) -> tf.lite.TFLiteConverter:
    if tfjs_graph_converter is None:
        raise ImportError('tfjs_graph_converter required')
    graph = tfjs_graph_converter.api.load_graph_model(model_path)
    tf_fn = tfjs_graph_converter.api.graph_to_function_v2(graph)
    return tf.lite.TFLiteConverter.from_concrete_functions([tf_fn])


def get_tflite_converter_for_model_path(model_path: str) -> tf.lite.TFLiteConverter:
    LOGGER.debug('converting model_path: %s', model_path)
    # if model_path.endswith('.json'):
    return get_tflite_converter_for_tfjs_model_path(model_path)
