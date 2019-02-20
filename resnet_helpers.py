from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class LayerHelper:

  def __init__(self, use_batchnorm, use_tpu, attention_losses):
    self.use_batchnorm = use_batchnorm
    self.use_tpu = use_tpu
    self.attention_losses = attention_losses

  def feature_attention_se(
          self,
          bottom,
          global_pooling=tf.reduce_mean,
          intermediate_nl=tf.nn.relu,
          squash=tf.sigmoid,
          name=None,
          training=True,
          combine='sum_p',
          _BATCH_NORM_DECAY=0.997,
          _BATCH_NORM_EPSILON=1e-5,
          r=4,
          return_map=False):
    """https://arxiv.org/pdf/1709.01507.pdf"""
    # 1. Global pooling
    mu = global_pooling(
        bottom, reduction_indices=[1, 2], keep_dims=True)

    # 2. FC layer with c / r channels + a nonlinearity
    c = int(mu.get_shape()[-1])
    intermediate_size = int(c / r)
    intermediate_activities = intermediate_nl(
        self.fc_layer(
            bottom=tf.contrib.layers.flatten(mu),
            out_size=intermediate_size,
            name='%s_ATTENTION_intermediate' % name,
            training=training))

    # 3. FC layer with c / r channels + a nonlinearity
    out_size = c
    output_activities = self.fc_layer(
        bottom=intermediate_activities,
        out_size=out_size,
        name='%s_ATTENTION_output' % name,
        training=training)
    if squash is not None:
        output_activities = self.fc_layer(
            bottom=intermediate_activities,
            out_size=out_size,
            name='%s_ATTENTION_output_squash_' % name,
            training=training)

    # 5. Scale bottom with output_activities
    exp_activities = tf.expand_dims(
        tf.expand_dims(output_activities, 1), 1)
    if return_map:
      return exp_activities

    # 4. Add batch_norm to scaled activities
    if self.use_batchnorm:
      bottom = tf.layers.batch_normalization(
          inputs=bottom,
          axis=3,
          momentum=_BATCH_NORM_DECAY,
          epsilon=_BATCH_NORM_EPSILON,
          center=True,
          scale=True,
          training=training,
          fused=True)
    scaled_bottom = bottom * exp_activities

    # 6. Add a loss term to compare scaled activity to clickmaps
    if combine == 'sum_abs':
      salience_bottom = tf.reduce_sum(
          tf.abs(scaled_bottom), axis=-1, keep_dims=True)
    elif combine == 'sum_p':
      salience_bottom = tf.reduce_sum(
          tf.pow(scaled_bottom, 2), axis=-1, keep_dims=True)
    else:
      raise NotImplementedError(
          '%s combine not implmented.' % combine)
    self.attention_losses += [salience_bottom]
    return scaled_bottom

  def feature_attention_gala(
          self,
          bottom,
          intermediate_nl=tf.nn.relu,
          squash=tf.sigmoid,
          name=None,
          training=True,
          extra_convs=1,
          extra_conv_size=5,
          dilation_rate=(1, 1),
          intermediate_kernel=1,
          normalize_output=False,
          include_fa=True,
          interaction='both',  # 'additive',  # 'both',
          r=4):
    """Fully convolutional form of https://arxiv.org/pdf/1709.01507.pdf"""
    # 1. FC layer with c / r channels + a nonlinearity
    c = int(bottom.get_shape()[-1])
    intermediate_channels = int(c / r)
    intermediate_activities = tf.layers.conv2d(
        inputs=bottom,
        filters=intermediate_channels,
        kernel_size=intermediate_kernel,
        activation=intermediate_nl,
        padding='SAME',
        use_bias=True,
        kernel_initializer=tf.variance_scaling_initializer(),
        trainable=training,
        name='%s_ATTENTION_intermediate' % name)

    # 1a. Optionally add convolutions with spatial dimensions
    if extra_convs:
      for idx in range(extra_convs):
        intermediate_activities = tf.layers.conv2d(
            inputs=intermediate_activities,
            filters=intermediate_channels,
            kernel_size=extra_conv_size,
            activation=intermediate_nl,
            padding='SAME',
            use_bias=True,
            dilation_rate=dilation_rate,
            kernel_initializer=tf.variance_scaling_initializer(),
            trainable=training,
            name='%s_ATTENTION_intermediate_%s' % (name, idx))

    # 2. Spatial attention map
    output_activities = tf.layers.conv2d(
        inputs=intermediate_activities,
        filters=1,  # c,
        kernel_size=1,
        padding='SAME',
        use_bias=True,
        activation=None,
        kernel_initializer=tf.variance_scaling_initializer(),
        trainable=training,
        name='%s_ATTENTION_output' % name)
    if self.use_batchnorm:
        output_activities = self.batch_norm_relu(
            inputs=output_activities,
            training=training,
            use_relu=False)

    # Also calculate se attention
    if include_fa:
      fa_map = self.feature_attention_se(
          bottom=bottom,
          intermediate_nl=intermediate_nl,
          squash=None,
          name=name,
          training=training,
          r=r,
          return_map=True)
    if interaction == 'both':
      k = fa_map.get_shape().as_list()[-1]
      alpha = tf.get_variable(
          name='alpha_%s' % name,
          shape=[1, 1, 1, k],
          initializer=tf.variance_scaling_initializer(),
          dtype=tf.bfloat16 if self.use_tpu else tf.float32)
      beta = tf.get_variable(
          name='beta_%s' % name,
          shape=[1, 1, 1, k],
          initializer=tf.variance_scaling_initializer(),
          dtype=tf.bfloat16 if self.use_tpu else tf.float32)
      additive = output_activities + fa_map
      multiplicative = output_activities * fa_map
      output_activities = alpha * additive + beta * multiplicative
    elif interaction == 'multiplicative':
      output_activities = output_activities * fa_map
    elif interaction == 'additive':
      output_activities = output_activities + fa_map
    else:
      raise NotImplementedError(interaction)
    output_activities = squash(output_activities)

    # 3. Scale bottom with output_activities
    scaled_bottom = bottom * output_activities

    # 4. Use attention for a clickme loss
    if normalize_output:
      norm = tf.sqrt(
          tf.reduce_sum(
              tf.pow(output_activities, 2),
              axis=[1, 2],
              keep_dims=True))
      self.attention_losses += [
          output_activities / (norm + 1e-12)]
    else:
      self.attention_losses += [output_activities]
    return scaled_bottom

  def fc_layer(
          self,
          bottom,
          in_size=None,
          out_size=None,
          name=None,
          activation=True,
          training=True):
    """Wrapper for a fully connected layer."""
    assert name is not None, 'Supply a name for your operation.'
    in_size = int(bottom.get_shape()[-1])
    x = tf.reshape(
        bottom,
        [-1, in_size]) if len(bottom.get_shape()) > 2 else bottom
    out = tf.contrib.layers.fully_connected(x, out_size, scope=name)
    return out
