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
"""Train a ResNet-50 model on ImageNet on TPU.



RESNETDEPTH=v2_152; \
BUCKET=gs://performances-tpu-$RESNETDEPTH; \
gsutil rm -r $BUCKET; gsutil mkdir $BUCKET; \
git pull; python3 models/official/resnet/resnet_main.py \
  --tpu_name=$TPU_NAME \
  --data_dir=gs://imagenet_data/train \
  --model_dir=gs://performances-tpu-$RESNETDEPTH \
  --resnet_depth=$RESNETDEPTH | tee -a performances-tpu-$RESNETDEPTH

RESNETDEPTH=fc-v2_152; \
BUCKET=gs://performances-tpu-$RESNETDEPTH; \
gsutil rm -r $BUCKET; gsutil mkdir $BUCKET; \
git pull; python3 models/official/resnet/resnet_main.py \
  --tpu_name=$TPU_NAME \
  --data_dir=gs://imagenet_data/train \
  --model_dir=gs://performances-tpu-$RESNETDEPTH \
  --clip_gradients 100 \
  --resnet_depth=$RESNETDEPTH | tee -a performances-tpu-$RESNETDEPTH

RESNETDEPTH=paper-v2_152; \
BUCKET=gs://performances-tpu-$RESNETDEPTH; \
gsutil rm -r $BUCKET; gsutil mkdir $BUCKET; \
git pull; python3 models/official/resnet/resnet_main.py \
  --tpu_name=$TPU_NAME \
  --data_dir=gs://imagenet_data/train \
  --model_dir=gs://performances-tpu-$RESNETDEPTH \
  --resnet_depth=$RESNETDEPTH | tee -a performances-tpu-$RESNETDEPTH


RESNETDEPTH=fc-v2_152; \
BUCKET=gs://performances-tpu-$RESNETDEPTH-aa; \
gsutil rm -r $BUCKET; gsutil mkdir $BUCKET; \
git pull; python3 models/official/resnet/resnet_main.py \
  --tpu_name=$TPU_NAME \
  --data_dir=gs://imagenet_data/train \
  --model_dir=$BUCKET \
  --clip_gradients 5 \
  --resnet_depth=$RESNETDEPTH | tee -a performances-tpu-$RESNETDEPTH



"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from absl import flags
import tensorflow as tf

import imagenet_input
import resnet_model
import resnet_v2_model
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'use_tpu', True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

flags.DEFINE_string(
    'master', default=None,
    help='gRPC URL of the master (i.e. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=None,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_string(
    'resnet_depth', default="50",
    help=('either the depth to use or [v2_50, ...]'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=200000,
    help=('The number of steps to use for training.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=512, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'steps_per_eval', default=5000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=100,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the model on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

# TODO(chrisying): remove this flag once --transpose_tpu_infeed flag is enabled
# by default for TPU
flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_integer(
    'clip_gradients', default=0,
    help='if 0 dont clip')


flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_string(
    'annotation',
    default='hms',
    help=('Annotation: hms or bboxs'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.1,
    help=('base LR when batch size = 256.'))


# Dataset constants
LABEL_CLASSES = 1000
NUM_TRAIN_IMAGES = 1281167
NUM_EVAL_IMAGES = 50000

# Learning hyperparameters
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def l2_channel_norm(x, eps=1e-12):
    denom = tf.sqrt(
        tf.maximum(
            tf.reduce_sum(
                x,
                reduction_indices=[1, 2, 3],
                keep_dims=True), eps))
    return x / denom


def blur(image, kernel=3, sigma=7, dtype=tf.bfloat16):
    """Apply blur to image."""
    if FLAGS.data_format == 'channels_last':
        df = 'NHWC'
    else:
        df = 'NCHW'

    def gauss_filter(size, sigma, ndims=1):
        """Create a gaussian filter."""
        x = int(size)
        if x % 2 == 0:
            x = x + 1
        x_zeros = np.zeros((size, size))

        center = int(np.floor(x / 2.))

        x_zeros[center, center] = 1
        y = gaussian_filter(
            x_zeros, sigma=sigma)[:, :, None, None]
        y = np.repeat(y, ndims, axis=2)
        return y
    out_dims = int(image.get_shape()[-1])
    k = tf.constant(
        gauss_filter(
            kernel,
            sigma,
            ndims=int(image.get_shape()[-1])).astype(
                np.float32).repeat(out_dims, axis=-1), dtype=dtype)
    return tf.nn.conv2d(image, k, [1, 1, 1, 1], padding='SAME', data_format=df)


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

  decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    images = features['image']
    hms = features['hm']
    bboxs = features['bbox']
    ccount = features['ccount']
  else:
    images = features
    hms = None
    bboxs = None

  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    images = tf.transpose(images, [0, 3, 1, 2])
    if hms is not None:
      hms = tf.transpose(hms, [0, 3, 1, 2])
      bboxs = tf.transpose(bboxs, [0, 3, 1, 2])

  if FLAGS.transpose_input:
    images = tf.transpose(images, [3, 0, 1, 2])  # HWCN to NHWC
    if hms is not None:
      hms = tf.transpose(hms, [3, 0, 1, 2])
      bboxs = tf.transpose(bboxs, [3, 0, 1, 2])

  if FLAGS.use_tpu:
    import bfloat16
    scope_fn = lambda: bfloat16.bfloat16_scope()
  else:
    scope_fn = lambda: tf.variable_scope("")

  with scope_fn():
    resnet_size = int(FLAGS.resnet_depth.split("_")[-1])
    if FLAGS.resnet_depth.startswith("v1_"):
      print("\n\n\n\n\nUSING RESNET V1 {}\n\n\n\n\n".format(FLAGS.resnet_depth))
      network = resnet_model.resnet_v1(
          resnet_depth=int(resnet_size),
          num_classes=LABEL_CLASSES,
          attention=None,
          apply_to="outputs",
          use_tpu=FLAGS.use_tpu,
          data_format=FLAGS.data_format)
    elif FLAGS.resnet_depth.startswith("se-v1_"):
      print(
          "\n\n\n\n\nUSING RESNET V1 (Squeeze-and-excite) {}\n\n\n\n\n".format(resnet_size))
      network = resnet_model.resnet_v1(
          resnet_depth=int(resnet_size),
          num_classes=LABEL_CLASSES,
          attention="se",
          apply_to="outputs",
          use_tpu=FLAGS.use_tpu,
          data_format=FLAGS.data_format)
    elif FLAGS.resnet_depth.startswith("GALA-v1_"):
      print(
          "\n\n\n\n\nUSING RESNET V1 (GALA) {}\n\n\n\n\n".format(resnet_size))
      network = resnet_model.resnet_v1(
          resnet_depth=int(resnet_size),
          num_classes=LABEL_CLASSES,
          attention="gala",
          apply_to="outputs",
          use_tpu=FLAGS.use_tpu,
          data_format=FLAGS.data_format)
    elif FLAGS.resnet_depth.startswith("v2_"):
      print(
          "\n\n\n\n\nUSING RESNET V2 {}\n\n\n\n\n".format(resnet_size))
      network = resnet_v2_model.resnet_v2(
          resnet_size=resnet_size,
          num_classes=LABEL_CLASSES,
          feature_attention=False,
          extra_convs=0,
          data_format=FLAGS.data_format,
          use_tpu=FLAGS.use_tpu)
    elif FLAGS.resnet_depth.startswith("SE-v2_"):
      print(
          "\n\n\n\n\nUSING RESNET V2 (Squeeze-and-excite) {}\n\n\n\n\n".format(resnet_size))
      network = resnet_v2_model.resnet_v2(
          resnet_size=resnet_size,
          num_classes=LABEL_CLASSES,
          feature_attention="se",
          extra_convs=0,
          apply_to="output",
          data_format=FLAGS.data_format,
          use_tpu=FLAGS.use_tpu)
    elif FLAGS.resnet_depth.startswith("GALA-v2_"):
      print(
          "\n\n\n\n\nUSING RESNET V2 (GALA) {}\n\n\n\n\n".format(resnet_size))
      network = resnet_v2_model.resnet_v2(
          resnet_size=resnet_size,
          num_classes=LABEL_CLASSES,
          feature_attention="gala",
          extra_convs=1,
          data_format=FLAGS.data_format,
          use_tpu=FLAGS.use_tpu)
    else:
      assert False

    logits, attention = network(
        inputs=images, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.cast(logits, tf.float32)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  batch_size = params['batch_size']

  # Calculate softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, LABEL_CLASSES)
  loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=one_hot_labels)

  # Switch hms/bboxs
  if FLAGS.annotation == 'hms':
      pass
  elif FLAGS.annotation == 'bboxs':
      hms = bboxs
  elif FLAGS.annotation == 'none':
      hms = None
  else:
      raise NotImplementedError(FLAGS.annotation)

  # Add attention losses
  if hms is not None:
    map_loss_list = []
    blur_click_maps = 49  # 0 = no, > 0 blur kernel
    blur_click_maps_sigma = 28  # 14

    # Blur the heatmaps
    hms = blur(
        hms,
        kernel=blur_click_maps,
        sigma=blur_click_maps_sigma,
        dtype=images.dtype)

    mask = tf.cast(tf.greater(ccount, 0), tf.float32)
    mask = tf.reshape(mask, [int(hms.get_shape()[0]), 1, 1, 1])
    for layer in attention:
        layer_shape = [int(x) for x in layer.get_shape()[1:3]]
        layer = tf.cast(layer, tf.float32)
        hms = tf.cast(hms, tf.float32)
        resized_maps = tf.image.resize_bilinear(
            hms,
            layer_shape,
            align_corners=True)
        if layer.get_shape().as_list()[-1] > 1:
            layer = tf.reduce_mean(
                tf.pow(layer, 2),
                axis=-1,
                keep_dims=True)
        resized_maps = l2_channel_norm(resized_maps)
        layer = l2_channel_norm(layer)
        dist = resized_maps - layer
        d = tf.nn.l2_loss(dist * mask)
        map_loss_list += [d]
    denominator = len(attention)
    map_loss = (
        tf.add_n(map_loss_list) / float(denominator)) * 1.  # (1. / 20.)
    loss += map_loss
  loss += (WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name and
       'ATTENTION' not in v.name and
       'block' not in v.name and
       'training' not in v.name]))
  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    batches_per_epoch = NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    current_epoch = (tf.cast(global_step, tf.float32) /
                     batches_per_epoch)
    learning_rate = learning_rate_schedule(current_epoch)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=True)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      if FLAGS.clip_gradients == 0:
        print("\nnot clipping gradients\n")
        train_op = optimizer.minimize(loss, global_step)
      else:
        print("\nclipping gradients\n")
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_gradients)
        train_op = optimizer.apply_gradients(
            zip(gradients, variables),
            global_step=global_step)

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, loss, lr, ce):  # , hm=None, image=None):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        with summary.create_file_writer(FLAGS.model_dir).as_default():
          with summary.always_record_summaries():
            summary.scalar('loss', loss[0], step=gs)
            summary.scalar('learning_rate', lr[0], step=gs)
            summary.scalar('current_epoch', ce[0], step=gs)
            # summary.image('image', hm, step=gs)
            # summary.image('heatmap', image, step=gs)
            return summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])
      # im_t = tf.cast(images, tf.float32)
      # hm_t = tf.cast(hms, tf.float32)
      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])
      # host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t, im_t, hm_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])
  # logging_hook = tf.train.LoggingTensorHook(
  #   {"logging_hook_loss": loss}, every_n_iter=1)

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      # training_hooks=[logging_hook]
  )


def main(unused_argv):
  """Run job."""
  tpu_grpc_url = None
  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    # Determine the gRPC URL of the TPU device to use
    if not FLAGS.master and not FLAGS.tpu_name:
      raise RuntimeError('You must specify either --master or --tpu_name.')

    if FLAGS.master:
      if FLAGS.tpu_name:
        tf.logging.warn('Both --master and --tpu_name are set. Ignoring'
                        ' --tpu_name and using --master.')
      tpu_grpc_url = FLAGS.master
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name,
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
  else:
    # URL is unused if running locally without TPU
    tpu_grpc_url = None

  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.iterations_per_loop,
      cluster=tpu_cluster_resolver,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = imagenet_input.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      use_bfloat=FLAGS.use_tpu,
      transpose_input=FLAGS.transpose_input)
  imagenet_eval = imagenet_input.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      use_bfloat=FLAGS.use_tpu,
      transpose_input=FLAGS.transpose_input)

  if FLAGS.mode == 'eval':
    eval_steps = NUM_EVAL_IMAGES // FLAGS.eval_batch_size

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(FLAGS.model_dir):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d' %
                        (eval_results, elapsed_time))

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d' % current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    batches_per_epoch = NUM_TRAIN_IMAGES / FLAGS.train_batch_size
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.' % (FLAGS.train_steps,
                                   FLAGS.train_steps / batches_per_epoch,
                                   current_step))

    start_timestamp = time.time()  # This time will include compilation time
    if FLAGS.mode == 'train':
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn, max_steps=int(FLAGS.train_steps))

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < FLAGS.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        resnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=int(next_checkpoint))
        current_step = next_checkpoint

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be consistently excluded modulo the batch size.
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=NUM_EVAL_IMAGES // FLAGS.eval_batch_size)
        tf.logging.info('Eval results: %s' % eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                    (FLAGS.train_steps, elapsed_time))

    if FLAGS.export_dir is not None:
      # The guide to serve a exported TensorFlow model is at:
      #    https://www.tensorflow.org/serving/serving_basic
      tf.logging.info('Starting to export model.')
      resnet_classifier.export_savedmodel(
          export_dir_base=FLAGS.export_dir,
          serving_input_receiver_fn=imagenet_input.image_serving_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
