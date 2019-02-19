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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import resnet_preprocessing


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = resnet_preprocessing.preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu-demos/blob/master/cloud_tpu/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data
    transpose_input: 'bool' for whether to use the double transpose trick
  """

  def __init__(self, is_training, data_dir, use_bfloat, transpose_input=True):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.data_dir = data_dir
    self.use_bfloat = use_bfloat
    self.transpose_input = transpose_input

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    if self.is_training:
        keys_to_features = {
            'image':
                tf.FixedLenFeature((), tf.string),
            'label':
                tf.FixedLenFeature([], tf.int64),
            'heatmap':
                tf.FixedLenFeature([], tf.string, default_value=''),
            'click_count':
                tf.FixedLenFeature([], tf.int64),
            'bbox':
                tf.FixedLenFeature(
                    [4], dtype=tf.int64, default_value=[0, 0, 0, 0])
        }
        parsed = tf.parse_single_example(value, keys_to_features)
        ccount = tf.cast(
            tf.reshape(parsed['click_count'], shape=[]), dtype=tf.int32)
        image_bytes = tf.reshape(parsed['image'], shape=[])
        hm_bytes = tf.reshape(parsed['heatmap'], shape=[])
        bbox = tf.cast(tf.reshape(parsed['bbox'], shape=[4]), tf.int64)
        image, hm, bbox = self.image_preprocessing_fn(
            image_bytes=image_bytes,
            hm_bytes=hm_bytes,
            bbox=bbox,
            is_training=self.is_training,
        )
        if self.use_bfloat:
            image = tf.cast(image, tf.bfloat16)
            hm = tf.cast(hm, tf.bfloat16)
            bbox = tf.cast(bbox, tf.bfloat16)
        image = {'image': image, 'hm': hm, 'bbox': bbox, 'ccount': ccount}
    else:
        keys_to_features = {
            'image':
                tf.FixedLenFeature((), tf.string),
            'label':
                tf.FixedLenFeature([], tf.int64),
            # 'heatmap':
            #     tf.FixedLenFeature([], tf.string),
            #  'bbox':
            #     tf.FixedLenFeature([4], dtype=tf.int64)
        }
        parsed = tf.parse_single_example(value, keys_to_features)
        image_bytes = tf.reshape(parsed['image'], shape=[])
        # hm_bytes = tf.reshape(parsed['heatmap'], shape=[])
        # bbox = tf.reshape(parsed['bbox'], shape=[4])
        image = self.image_preprocessing_fn(
            image_bytes=image_bytes,
            hm_bytes=None,
            bbox=None,
            is_training=self.is_training,
        )
        if self.use_bfloat:
            image = tf.cast(image, tf.bfloat16)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['label'], shape=[]), dtype=tf.int32)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'val-*')
    dataset = tf.data.Dataset.list_files(
        file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))
    if self.is_training:
        dataset = dataset.shuffle(1024)

    # Parse, preprocess, and batch the data in parallel
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=8,    # 8 == num_cores per host
            drop_remainder=True))

    def tfun(images, dim_order):
        """Function for transposing."""
        if isinstance(images, dict):
            hm = images['hm']
            image = images['image']
            bbox = images['bbox']
            images['hm'] = tf.transpose(hm, dim_order)
            images['image'] = tf.transpose(image, dim_order)
            images['bbox'] = tf.transpose(bbox, dim_order)
        else:
            images = tf.transpose(images, dim_order)
        return images

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tfun(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=8)

    def set_shapes(images, labels):
      """Statically set the batch_size dimension."""
      if isinstance(images, dict):
          hm = images['hm']
          image = images['image']
          bbox = images['bbox']
          if self.transpose_input:
            image.set_shape(image.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            bbox.set_shape(bbox.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            hm.set_shape(hm.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
          else:
            image.set_shape(image.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            bbox.set_shape(bbox.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            hm.set_shape(hm.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
          images['image'] = image
          images['hm'] = hm
          images['bbox'] = bbox
      else:
          if self.transpose_input:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
          else:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
      return images, labels

    # Assign static batch size dimension
    dataset = dataset.map(set_shapes)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
