# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

IMAGE_SIZE = 224
FULL_IMAGE_SIZE = 256
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                bbox_image=None,
                                hm_bytes=None,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(
          scope,
          'distorted_bounding_box_crop',
          [image_bytes, bbox]):
    offsets = tf.random_uniform(
        [2],
        minval=0,
        maxval=FULL_IMAGE_SIZE - IMAGE_SIZE,
        dtype=tf.int32)
    offset_y = offsets[0]
    offset_x = offsets[1]
    image = tf.decode_raw(image_bytes, tf.float32)
    image = tf.reshape(image, [FULL_IMAGE_SIZE, FULL_IMAGE_SIZE, 3])
    image = tf.image.crop_to_bounding_box(
        image,
        offset_y,
        offset_x,
        IMAGE_SIZE,
        IMAGE_SIZE)
    if bbox_image is None:
        return image
    else:
        bbox_image = tf.image.crop_to_bounding_box(
            bbox_image,
            offset_y,
            offset_x,
            IMAGE_SIZE,
            IMAGE_SIZE)
        hm_image = tf.decode_raw(hm_bytes, tf.float32)
        hm_image = tf.reshape(hm_image, [FULL_IMAGE_SIZE, FULL_IMAGE_SIZE, 1])
        hm_image = tf.image.crop_to_bounding_box(
            hm_image,
            offset_y,
            offset_x,
            IMAGE_SIZE,
            IMAGE_SIZE)
        return (image, bbox_image, hm_image)


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes):
  """Make a random crop of IMAGE_SIZE."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes),
      lambda: tf.image.resize_bicubic(
          [image],
          [IMAGE_SIZE, IMAGE_SIZE])[0])
  return image


def _decode_and_center_crop(image_bytes):
  """Crops to center of image with padding then scales IMAGE_SIZE."""
  padded_center_crop_size = tf.cast(
      ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
       tf.cast(FULL_IMAGE_SIZE, tf.float32)),
      tf.int32)
  offset_height = ((FULL_IMAGE_SIZE - padded_center_crop_size) + 1) // 2
  offset_width = ((FULL_IMAGE_SIZE - padded_center_crop_size) + 1) // 2
  image = tf.decode_raw(image_bytes, tf.float32)
  image = tf.reshape(image, [FULL_IMAGE_SIZE, FULL_IMAGE_SIZE, 3])
  image = tf.image.crop_to_bounding_box(
      image,
      offset_height,
      offset_width,
      IMAGE_SIZE,
      IMAGE_SIZE)
  return image


def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset
  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def create_bb(bb_coors, im_size):
  """DEPRECIATED: Convert ClickMe mask into a bounding box mask for an image."""
  bb_coors = tf.cast(tf.squeeze(bb_coors), tf.int32)
  he = bb_coors[0]
  wi = bb_coors[1]
  hmax = bb_coors[2]
  wmax = bb_coors[3]
  he = tf.maximum(he, 0)
  wi = tf.maximum(wi, 0)
  height = tf.abs(hmax - he)
  width = tf.abs(wmax - wi)
  height = tf.maximum(height, 1)
  width = tf.maximum(width, 1)
  ones_im = tf.ones(tf.stack([height, width, 1]))
  height_offset = im_size[0] - height
  width_offset = im_size[1] - width
  paddings = [[he, height_offset], [wi, width_offset], [0, 0]]
  padded_im = tf.pad(ones_im, paddings, name='padded_im')
  padded_im.set_shape(im_size)
  return padded_im


def _clickme_decode_and_random_crop(image_bytes, hm_bytes, bbox_image):
  """Apply the same random crop to image and clickme map."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  bbox_image = create_bb(
      bbox_image,
      [FULL_IMAGE_SIZE, FULL_IMAGE_SIZE, 1])
  image, bbox_image, hm_image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      bbox_image=bbox_image,
      hm_bytes=hm_bytes,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  return image, bbox_image, hm_image


def preprocess_for_train(image_bytes, hm_bytes, bbox):
  """Preprocesses the given image for evaluation.
  1. Make BBox
  2. Cat BBox, image, and HM

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image, bbox, hm = _clickme_decode_and_random_crop(
      image_bytes,
      hm_bytes, bbox)
  image = _normalize(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  if bbox is None:
    image = _flip(image)
    return image
  else:
    bbox = tf.reshape(bbox, [IMAGE_SIZE, IMAGE_SIZE, 1])
    hm = tf.reshape(hm, [IMAGE_SIZE, IMAGE_SIZE, 1])

    image_tensor = tf.concat([image, bbox, hm], axis=-1)
    image_tensor = _flip(image_tensor)
    image = image_tensor[:, :, :3]
    bbox = tf.expand_dims(image_tensor[:, :, 3], axis=-1)
    hm = tf.expand_dims(image_tensor[:, :, 4], axis=-1)
    return image, hm, bbox


def preprocess_for_eval(image_bytes):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes)
  image = _normalize(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  return image


def preprocess_image(image_bytes, hm_bytes, bbox, is_training=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return preprocess_for_train(image_bytes, hm_bytes, bbox)
  else:
    return preprocess_for_eval(image_bytes)
