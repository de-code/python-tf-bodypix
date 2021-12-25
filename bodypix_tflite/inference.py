import numpy as np
from PIL import Image
from pathlib import Path

from tf_bodypix.draw import draw_poses
from tf_bodypix.model import load_tflite_model, get_poses
from tf_bodypix.bodypix_js_utils.util import pad_and_resize_to

# Setup input and output paths
output_path = Path('./data/example-output')
output_path.mkdir(parents=True, exist_ok=True)

# Load model
model_name = 'mobilenet_075_stride16'
stride = int(model_name.split('stride')[-1])
model_path = './models/mobilenet_075_stride16/model.tflite'
bodypix_model = load_tflite_model(model_path)

# Load image
image = Image.open('image.jpg')
image_array = np.asarray(image, dtype='float32')

# Get image shape
image_height, image_width, _ = image_array.shape

# Resize, pad, and normalize the image
image_padded, padding = pad_and_resize_to(image_array, target_height=209, target_width=321)
image_normalized = (image_padded / 127.5) - 1

# Get input image shape
input_height, input_width, _ = image_normalized.shape

# Get prediction result
result = bodypix_model(image_normalized)

# Get poses from result
poses = get_poses(heatmap_logits=result['float_heatmaps'], short_offsets=result['float_short_offsets'],
                  displacement_fwd=result['MobilenetV1/displacement_fwd_2/BiasAdd'],
                  displacement_bwd=result['MobilenetV1/displacement_bwd_2/BiasAdd'], padding=padding,
                  image_height=image_height, image_width=image_width, output_stride=stride,
                  model_input_height=input_height, model_input_width=input_width)

# Draw poses on the image
image_with_poses = draw_poses(image_array.copy(), poses, keypoints_color=(255, 100, 100),
                              skeleton_color=(100, 100, 255))

image_with_poses = Image.fromarray(image_with_poses.astype(np.uint8))
image_with_poses.save('./data/example-output/output-poses.png')
