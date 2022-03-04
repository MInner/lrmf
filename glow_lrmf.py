import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tqdm.auto as tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

def ForwardChain(bijectors, name=None):
  return tfb.Chain(list(reversed(bijectors)), name=name)

sigmoid_nvp_scale_bijector = ForwardChain([tfb.Shift(2), tfb.Sigmoid()], 'sigmoid_p2_scale')
softplus_nvp_scale_bijector = ForwardChain([tfb.Shift(2), tfb.Softplus()], 'softplus_p2_scale')
default_nvp_scale_bijector = sigmoid_nvp_scale_bijector
default_actnorm_scale_bijector = softplus_nvp_scale_bijector

import tensorflow.compat.v1 as tf1

# https://github.com/openai/glow/blob/master/tfops.py#L203

def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
    name = "_".join([str(dim) for dim in [a, b, *(x.shape[1:3])]])
    pads = tf1.get_collection(name)
    if not pads:
        print("Creating pad", name)
        pad = np.zeros([1] + x.shape[1:3] + [1], dtype='float32')
        pad[:, :a, :, 0] = 1.
        pad[:, -a:, :, 0] = 1.
        pad[:, :, :b, 0] = 1.
        pad[:, :, -b:, 0] = 1.
        pad = tf.convert_to_tensor(pad)
        tf1.add_to_collection(name, pad)
    else:
        pad = pads[0]
    pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
    x = tf.concat([x, pad], axis=3)
    return x


def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
      if not reverse:
          x = actnorm_center(name+"_center", x, reverse)
          x = actnorm_scale(name+"_scale", x, scale, logdet,
                            logscale_factor, batch_variance, reverse, init)
          if logdet != None:
              x, logdet = x
      else:
          x = actnorm_scale(name + "_scale", x, scale, logdet,
                            logscale_factor, batch_variance, reverse, init)
          if logdet != None:
              x, logdet = x
          x = actnorm_center(name+"_center", x, reverse)
      if logdet != None:
          return x, logdet
      return x


def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf1.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))


def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf1.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, int_shape(x)[3]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x


def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf1.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:
            x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
            logdet_factor = int(shape[1])*int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        inv_std = scale/(tf.sqrt(x_var)+1e-6)/logdet_factor
        remapped_init = default_actnorm_scale_bijector.inverse(inv_std)*logscale_factor
        remapped = get_variable_ddi("logs", _shape, initial_value=remapped_init)
        un_mapped = default_actnorm_scale_bijector.forward(remapped)
  
        if not reverse:
            x = x * un_mapped
        else:
            x = x / un_mapped

        if logdet != None:
            dlogdet = tf.reduce_sum(tf.log(un_mapped)) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x


__glow_conv_scope_name = "glow_conv_net_template_tf1"

# https://github.com/openai/glow/blob/master/model.py#L420
# https://github.com/openai/glow/blob/master/tfops.py#L236
def glow_conv_net_template_tf1(
        image_shape,
        filters=(512, 512),
        kernel_size=(3, 3),
        activation=tf.nn.relu,
        actnorm_scale_norm_bijector=None,
        name=None):
  
  activation = tf.nn.relu if activation == 'relu' else activation

  with tf.name_scope(name or __glow_conv_scope_name):
    def _fn(x, output_units=None):

      output_units = output_units or image_shape[-1]

      x = tf.reshape(
          x, (-1, int(image_shape[2])//2, *image_shape[:2].as_list()))

      # x_np = x.numpy()
      # print(x_np.shape)
      # print(x_np.mean(axis=(0, 2, 3)))
      # print(x_np.var(axis=(0, 2, 3)))
      # x_np_imgs = x_np.reshape(-1, *x_np.shape[2:])[..., None]
      # show_panel(x_np_imgs, n=16)

      # [B, C, H, W] -> [B, H, W, C]
      x = tf.transpose(x, (0, 2, 3, 1))
      
      kernel_sizes = [kernel_size] + [(1, 1)] * (len(filters) - 1)
      for i, (filter_size_i, kernel_size_i) in enumerate(zip(filters, kernel_sizes)):
        x = add_edge_padding(x, kernel_size_i)
        x = tf1.layers.conv2d(
            inputs=x,
            filters=filter_size_i,
            kernel_size=kernel_size_i,
            strides=(1, 1),
            padding='valid',
            # kernel_initializer=tf.random_normal_initializer(0.0, 0.05),
            use_bias=False,
            # kernel_constraint=lambda kernel: (
            #     tf.nn.l2_normalize(
            #         kernel, list(range(kernel.shape.ndims-1))))
            )

        # x = tf1.layers.batch_normalization(x, axis=-1)
        x = actnorm('actnorm_%d' % i, x)
        
        x = activation(x)
      
      x = add_edge_padding(x, kernel_size)
      output_chans = (output_units // np.prod(image_shape[:2]))
      
      x = tf1.layers.conv2d(
          inputs=x,
          filters=2 * output_chans,
          kernel_size=kernel_size,
          strides=(1, 1),
          padding='valid',
          use_bias=True,
          kernel_initializer=tf.zeros_initializer())
      
      # x = tf1.layers.batch_normalization(x, axis=-1)
      # x = ActNorm(use_get_variable=True).forward(x)
      x = actnorm('actnorm_last', x)
      
      shift_log_scale = tf.transpose(x, (0, 3, 1, 2))  # [B, 2*C, H, W]
      shift, log_scale = tf.split(shift_log_scale, 2, axis=1)  # 2*[B, C, H, W]
      shift = tf.reshape(shift, [-1, np.prod(image_shape[:2]) * output_chans])
      log_scale = tf.reshape(log_scale, [-1, np.prod(image_shape[:2]) * output_chans])
      
      return shift, log_scale

  return tf1.make_template("glow_conv_net_template", _fn)


tfk = tf.keras
tfkl = tf.keras.layers


class ActNormLayer(tfkl.Layer):
  def __init__(self, scale_norm_bijector=default_actnorm_scale_bijector, **kwargs):
    super().__init__(**kwargs)
    self._actnorm = ActNorm(scale_norm_bijector=scale_norm_bijector)

  def call(self, inputs):
    return self._actnorm.forward(inputs)


class GlowConvNet(object):
  def __init__(self, image_shape, filters=(512, 512), kernel_size=(3, 3), 
               actnorm_scale_norm_bijector=default_actnorm_scale_bijector, 
               activation='relu', name=None):
    self._image_shape = image_shape
    self._filters = filters
    self._kernel_size = kernel_size
    self._activation = activation
    self._actnorm_scale_norm_bijector = actnorm_scale_norm_bijector
    self.built = False

  def build_net(self, x, output_chans):
    layers = []
    kernel_sizes = [self._kernel_size] + [(1, 1)] * (len(self._filters) - 1)
    for filters_i, kernel_size_i in zip(self._filters, kernel_sizes):
      layers.extend([
        tfkl.Conv2D(filters_i, kernel_size_i, padding='same', use_bias=False),
        ActNormLayer(self._actnorm_scale_norm_bijector),
        tfkl.Activation(self._activation)
      ])

    layers.extend([
      tfkl.Conv2D(2 * output_chans, 
                  self._kernel_size, 
                  padding='same', 
                  use_bias=True, 
                  kernel_initializer=tf.zeros_initializer()),
      ActNormLayer(self._actnorm_scale_norm_bijector)
    ])
    return tfk.Sequential(layers)

  def __call__(self, x, output_units=None):
    output_units = output_units or self._image_shape[-1]
    output_chans = output_units // np.prod(self._image_shape[:2])

    if not self.built:
      self.net = self.build_net(x, output_chans)
      self.built = True

    image_shape = self._image_shape
    chw_shape = (-1, int(image_shape[2])//2, *image_shape[:2].as_list())
    x = tf.reshape(x, chw_shape)

    # x_np = x.numpy()
    # print(x_np.shape)
    # print(x_np.mean(axis=(0, 2, 3)))
    # print(x_np.var(axis=(0, 2, 3)))
    # x_np_imgs = x_np.reshape(-1, *x_np.shape[2:])[..., None]
    # show_panel(x_np_imgs, n=16)

    x = tf.transpose(x, (0, 2, 3, 1)) # [B, C, H, W] -> [B, H, W, C]

    output = self.net(x)  # [B, H, W, 2*C]
    shift_log_scale = tf.transpose(output, (0, 3, 1, 2))  # [B, 2*C, H, W]
    shift, log_scale = tf.split(shift_log_scale, 2, axis=1)  # 2*[B, C, H, W]
    shift = tf.reshape(shift, [-1, np.prod(image_shape[:2]) * output_chans])
    log_scale = tf.reshape(log_scale, [-1, np.prod(image_shape[:2]) * output_chans])
    return shift, log_scale

from functools import partial


class BuildBijector(tfb.Bijector):
    def __init__(self,
                 forward_min_event_ndims=3,
                 inverse_min_event_ndims=3,
                 keeps_dims=False,
                 inits_from_data=False,
                 *args,
                 **kwargs):
      
      self._keeps_dims = keeps_dims
      self._inits_from_data = inits_from_data
      self.built = False

      super().__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          inverse_min_event_ndims=inverse_min_event_ndims,
          *args,
          **kwargs)

    def build_chain(self, input_shape):
      raise NotImplementedError()

    def invert_shape(self, output_shape):
      if self._keeps_dims:
        return output_shape
      else:
        raise NotImplementedError()

    def build(self, input_shape):
      self.flow = self.build_chain(input_shape)
      self.built = True
    
    def init_from_data(self, x):
      pass
      
    def _forward(self, x, **kwargs):
        if not self.built:
          self.build(x.get_shape())
          self.init_from_data(x)

        return self.flow.forward(x, **kwargs)

    def _inverse(self, y, **kwargs):
        if not self.built:
          if self._inits_from_data:
            raise RuntimeError('This bijector needs a forward pass to init.')
          else:
            input_shape = self.invert_shape(y.get_shape())
            self.build(input_shape)

        return self.flow.inverse(y, **kwargs)

    def _forward_log_det_jacobian(self, x, **kwargs):
        if not self.built:
            self.build(x.get_shape())
            self.init_from_data(x)

        return self.flow.forward_log_det_jacobian(
            x, event_ndims=self.forward_min_event_ndims)

    def _inverse_log_det_jacobian(self, y, **kwargs):
        if not self.built:
          if self._inits_from_data:
            raise RuntimeError('This bijector needs a forward pass to init.')
          else:
            input_shape = self.invert_shape(y.get_shape())
            self.build(input_shape)

        return self.flow.inverse_log_det_jacobian(
            y, event_ndims=self.inverse_min_event_ndims)


class Squeeze(BuildBijector):
  def __init__(self, factor=2, name=None):
    self._factor = factor
    super().__init__(name=name)

  def build_chain(self, input_shape):
    factor = self._factor
    (H, W, C) = input_shape[-3:]
    intermediate_shape = (H//factor, factor, W//factor, factor, C)
    perm = [0, 2, 4, 1, 3]
    output_input_shape = [intermediate_shape[x] for x in perm]
    output_shape = (H//factor, W//factor, C*factor**2)

    flow = ForwardChain([
        tfb.Reshape(intermediate_shape, input_shape[-3:]),
        tfb.Transpose(perm),
        tfb.Reshape(output_shape, output_input_shape),
    ])
    
    return flow

  def invert_shape(self, inverse_shape):
      factor = self._factor
      input_shape = (inverse_shape[0], 
                      inverse_shape[1]*factor, inverse_shape[2]*factor, 
                      inverse_shape[3]//factor//factor)
      
      return input_shape


class WrapCHWBijectorToFlat(BuildBijector):
  def __init__(self, bijector, image_shape, name=None):
    self.bijector = bijector
    self.image_shape = image_shape
    super().__init__(keeps_dims=True, name=name,
                     forward_min_event_ndims=1, 
                     inverse_min_event_ndims=1)

  def build_chain(self, input_shape):
    image_shape = self.image_shape
    fwd_trans = [2, 0, 1]  # [B, C, H, W]
    bwd_trans = [1, 2, 0]  # [B, H, W, C]
    image_shape_trans = [image_shape[i] for i in fwd_trans]
    flow = ForwardChain([
        tfb.Reshape(event_shape_in=(np.prod(image_shape_trans),), # [B, C*H*W]
                    event_shape_out=image_shape_trans),        # [B, C, H, W]) 
        tfb.Transpose(bwd_trans),  # [B, H, W, C]
        self.bijector,
        tfb.Transpose(fwd_trans),  # [B, C, H, W]
        tfb.Reshape(event_shape_in=image_shape_trans,           # [B, C, H, W]
                    event_shape_out=(np.prod(image_shape_trans),)) # [B, C*H*W]
    ])

    flow._forward_min_event_ndims = 1
    flow._inverse_min_event_ndims = 1
    return flow


class WrapFlatBijectorToCHW(BuildBijector):
  def __init__(self, bijector, name=None):
    self.bijector = bijector
    super().__init__(keeps_dims=True, name=name)

  def build_chain(self, input_shape):
    image_shape = input_shape[1:]
    fwd_trans = [2, 0, 1]  # [B, C, H, W]
    bwd_trans = [1, 2, 0]  # [B, H, W, C]

    fwd_trans_shape = [image_shape[i] for i in fwd_trans]

    flow = ForwardChain([
        tfb.Transpose(fwd_trans),  # -> [B, C, H, W]
        tfb.Reshape(event_shape_in=fwd_trans_shape, 
                    event_shape_out=(np.prod(fwd_trans_shape),)),  # [B, C*H*W]
        self.bijector,
        tfb.Reshape(event_shape_out=fwd_trans_shape, 
                    event_shape_in=(np.prod(fwd_trans_shape),)),  # [B, C, H, W]
        tfb.Transpose(bwd_trans),  # [B, H, W, C]
    ])

    return flow


class BlockwiseSplitChan(BuildBijector):
  def __init__(self, bijectors, name=None):
    assert len(bijectors) == 2
    self.bijectors = bijectors
    super().__init__(keeps_dims=True, name=name)

  def build_chain(self, input_shape):
    assert input_shape[-1] >= 2  # [B, H, W, C]
    image_shape = input_shape[1:]
    sub_img_shapes = [
        [*image_shape[:-1], image_shape[-1] // 2],
        [*image_shape[:-1], image_shape[-1] - image_shape[-1] // 2]
    ]
    # each b in bijectors receives / produces in the [B, H, W, C] order
    # each fb in flat_bijectors in the [B, C*H*W] order
    flat_bijectors = [WrapCHWBijectorToFlat(bb, sub_shape) for bb, sub_shape 
                      in zip(self.bijectors, sub_img_shapes)]
    block_sizes = [np.prod(shape) for shape in sub_img_shapes]
    flat_blockwise = tfp.bijectors.Blockwise(flat_bijectors, block_sizes, name='bl_fl')
    # flat_blockwise receives input / produces output in the [B, C*H*W] order
    # final blockwise receives / produces in the [B, H, W, C] order
    blockwise = WrapFlatBijectorToCHW(flat_blockwise)
    return blockwise


from tensorflow.keras import layers

def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
  with tf.name_scope(name or 'trainable_lu_factorization'):
    event_size = tf.convert_to_tensor(
        event_size, dtype_hint=tf.int32, name='event_size')
    batch_shape = tf.convert_to_tensor(
        batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
    kern_shape = tf.concat([batch_shape, [event_size, event_size]], axis=0)
    # print('kern_shape', kern_shape)
    random_matrix = tf.random.uniform(shape=kern_shape, dtype=dtype, seed=seed)
    random_orthonormal = tf.linalg.qr(random_matrix)[0]
    lower_upper, permutation = tf.linalg.lu(random_orthonormal)
    lower_upper = tf.Variable(
        initial_value=lower_upper,
        trainable=True,
        name='lower_upper')
    # Initialize a non-trainable variable for the permutation indices so
    # that its value isn't re-sampled from run-to-run.
    permutation = tf.Variable(
        initial_value=permutation,
        trainable=False,
        name='permutation')
    # print('conv weight shapes', lower_upper.shape, permutation.shape)
    return lower_upper, permutation


class ActNorm(BuildBijector):
  def __init__(self, name=None, scale_norm_bijector=default_actnorm_scale_bijector):
    super().__init__(inits_from_data=True, name=name)
    self.scale_norm_bijector = scale_norm_bijector
    # self.log_scale_factor = log_scale_factor

  def _bijector_fn(self, x0, input_depth, **condition_kwargs):
    shift, re_scale = self.bias, self.re_scale
    scale = self.scale_norm_bijector.forward(re_scale)
    return tfb.AffineScalar(shift=shift, scale=scale)

  def build_chain(self, input_shape):
    self.bias = tf.Variable(tf.zeros(input_shape[-1]), name='bias', trainable=True)
    self.re_scale = tf.Variable(tf.zeros(input_shape[-1]), name='re_scale', trainable=True)
    # tfb.Affine did not allow trainable tf.exp(log_scale)
    # not using Shift()(Scale()) directly to hopefully enable caching?
    return tfb.RealNVP(num_masked=0, bijector_fn=self._bijector_fn)
  
  def init_from_data(self, x):
    mean, variance = tf.nn.moments(x, [0, 1, 2], keepdims=False)
    # log_scale = -0.5 * tf.math.log(variance + 1e-6)
    scale = 1.0 / tf.math.sqrt(variance + 1e-6)
    re_scale = self.scale_norm_bijector.inverse(scale)
    # because y = x * exp(log_scale) + bias
    self.bias.assign(-mean*scale)
    self.re_scale.assign(re_scale)


class ConvRealNVP(BuildBijector):
  def __init__(self,
              filters=512,
              kernel_size=(3,3),
              resnet_depth=2,
              nvp_scale_norm_bijector=default_nvp_scale_bijector,
              actnorm_scale_norm_bijector=default_actnorm_scale_bijector,
              shift_scale_fn_builder=glow_conv_net_template_tf1,
              name=None):
    super().__init__(keeps_dims=True, name=name)
    self._filters = filters
    self._kernel_size = kernel_size
    self._resnet_depth = resnet_depth
    self._nvp_scale_norm_bijector = nvp_scale_norm_bijector
    self._actnorm_scale_norm_bijector = actnorm_scale_norm_bijector
    self._shift_scale_fn_builder = shift_scale_fn_builder

  def _bijector_fn(self, x0, input_depth, **condition_kwargs):
    shift, re_scale = self._shift_and_log_scale_fn(x0, input_depth)
    scale = self._nvp_scale_norm_bijector.forward(re_scale)
    return tfb.AffineScalar(shift=shift, scale=scale)

  def build_chain(self, input_shape):
    image_shape = input_shape[1:]
    # GlowConvNet
    self._shift_and_log_scale_fn = self._shift_scale_fn_builder(
        image_shape=image_shape,
        filters=(self._filters,)*self._resnet_depth,
        kernel_size=self._kernel_size,
        activation='relu',
        actnorm_scale_norm_bijector=self._actnorm_scale_norm_bijector)

    affine_coupling = tfb.RealNVP(
        num_masked=np.prod(image_shape[:2])*(image_shape[2]//2),
        bijector_fn=self._bijector_fn)
    
    wrapped_real_nvp = WrapFlatBijectorToCHW(affine_coupling)
    return wrapped_real_nvp


class GlowStep(BuildBijector):

  def __init__(self,
               depth=3,
               filters=512,
               kernel_size=(3,3),
               resnet_depth=2,
               normalization='actnorm',
               nvp_scale_norm_bijector=default_nvp_scale_bijector,
               actnorm_scale_norm_bijector=default_actnorm_scale_bijector,
               shift_scale_fn_builder=glow_conv_net_template_tf1,
               name=None):
    
    self._depth = depth
    self._normalization = normalization
    self._actnorm_scale_norm_bijector = actnorm_scale_norm_bijector
    self._shift_scale_args = {
        'filters': filters,
        'kernel_size': kernel_size,
        'resnet_depth': resnet_depth,
        'nvp_scale_norm_bijector': nvp_scale_norm_bijector,
        'actnorm_scale_norm_bijector': actnorm_scale_norm_bijector,
        'shift_scale_fn_builder': shift_scale_fn_builder
    }
    super().__init__(keeps_dims=True, name=name)

  def get_normalization(self):
    norm = self._normalization
    if norm == 'actnorm':
      return ActNorm(scale_norm_bijector=self._actnorm_scale_norm_bijector)
    elif norm == 'batch':
      return tfb.BatchNormalization(
          batchnorm_layer=tf.keras.layers.BatchNormalization(axis=-1))
    elif norm is None:
      return tfb.Identity()
    else:
      raise ValueError('norm must be in [act, batch, None]')

  def get_conv1x1(self):
    w_perm = trainable_lu_factorization(self._input_shape[-1])
    conv1x1 = tfb.ScaleMatvecLU(*w_perm, validate_args=True)
    return conv1x1

  def get_real_nvp(self):
    return ConvRealNVP(**self._shift_scale_args)

  def build_chain(self, input_shape):
    self._input_shape = input_shape
    self._image_shape = input_shape[1:]

    flow_parts = []
    for i in range(self._depth):
      flow_parts.extend([
          self.get_normalization(),
          self.get_conv1x1(),
          self.get_real_nvp()
      ])

    return ForwardChain(flow_parts)


class GlowFlow(BuildBijector):
    def __init__(self,
                 levels=2,
                 level_depth=2,
                 factor=2,
                 filters=512,
                 kernel_size=(3,3),
                 resnet_depth=2,
                 nvp_scale_norm_bijector=default_nvp_scale_bijector,
                 actnorm_scale_norm_bijector=default_actnorm_scale_bijector,
                 shift_scale_fn_builder=glow_conv_net_template_tf1,
                 normalization='actnorm',
                 name=None):
      
        self._level = levels
        self._level_depth = level_depth
        self._factor = factor
        self._glow_step_args = {
            'filters': filters,
            'kernel_size': kernel_size,
            'resnet_depth': resnet_depth,
            'normalization': normalization,
            'nvp_scale_norm_bijector': nvp_scale_norm_bijector,
            'actnorm_scale_norm_bijector': actnorm_scale_norm_bijector,
            'shift_scale_fn_builder': shift_scale_fn_builder
        }
        super().__init__(keeps_dims=True, name=name)

    def build_chain(self, input_shape):
      squeeze = Squeeze(factor=self._factor)
      flow_steps = [
        squeeze, 
        GlowStep(
          depth=self._level_depth,
          **self._glow_step_args
        )
      ]

      if self._level > 1:
        flow_steps.append(
          BlockwiseSplitChan([
            tfb.Identity(),
            GlowFlow(
              levels=self._level - 1,
              level_depth=self._level_depth,
              factor=self._factor,
              **self._glow_step_args
            )
          ])
        )

      flow_steps.append(tfb.Invert(squeeze))

      return ForwardChain(flow_steps)

def build_deep_conv_glow(shape, **glow_flow_args):
  base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros(np.prod(shape)),
        scale_diag=tf.ones(np.prod(shape)))
  
  base_resh_dist = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=tfp.bijectors.Reshape(event_shape_out=shape, 
                                   event_shape_in=[np.prod(shape)]),
  )

  glow_flow = GlowFlow(**glow_flow_args)
  inv_glow_flow = tfb.Invert(glow_flow)

  # this way _images_ are passed _forward_ through glow_flow 
  # during model.log_prob
  # z = glow.forward(image)
  # z = inv_glow.inverse(image)

  glow_flow_dist = tfd.TransformedDistribution(
      distribution=base_resh_dist,
      bijector=inv_glow_flow)

  # (for some reason) required for initialization
  # test = np.random.rand(2, *shape).astype('float32')
  # assert glow_flow_dist.bijector.forward(test).shape
  # assert glow_flow_dist.log_prob(test).shape == (2,)
  return glow_flow_dist

def build_domain_classifier(input_shape=(16, 16, 1), dense=False, 
                            lr=1e-3, epochs=50, val_size=128, batch_size=64, 
                            verbose=False, patience=3, eps=1e-30):
  if dense:
    part = [tfkl.Flatten(), tfkl.Dense(256, activation='relu')]
  else:
    part = [
      tfkl.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
      tfkl.Conv2D(32, (3, 3), activation='relu'),
      tfkl.MaxPooling2D(pool_size=(2, 2)),
      tfkl.Dropout(0.25),
      tfkl.Flatten(),
    ]

  domain_clf = tfk.Sequential([
    *part,
    tfkl.Dense(10, activation='relu'),
    tfkl.Dropout(0.5),
    tfkl.Dense(1, activation='sigmoid')
  ])

  _from_tensor_slices = tf.data.Dataset.from_tensor_slices

  def train_dc(dataset_a, dataset_b):
    domain_dataset = tf.data.experimental.sample_from_datasets([
      tf.data.Dataset.zip((dataset_a, _from_tensor_slices([0.0]).repeat())),
      tf.data.Dataset.zip((dataset_b, _from_tensor_slices([1.0]).repeat()))
    ])

    opt = tfk.optimizers.Adam(learning_rate=lr)
    domain_clf.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    train_ds = domain_dataset.skip(val_size).batch(batch_size)
    val_ds = domain_dataset.take(val_size).batch(batch_size)

    cb = tfk.callbacks.EarlyStopping('val_accuracy', min_delta=0.01, patience=patience) 
    domain_clf.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cb], verbose=verbose)
    eval_results = domain_clf.evaluate(val_ds, verbose=verbose)
    return eval_results

  def log_odds(dataset_a, dataset_b):
    log_odds = []
    for ds in [dataset_a, dataset_b]:
      p = np.concatenate([np.ravel(domain_clf(x).numpy()) for x in ds.batch(batch_size)])
      logodds = np.log(np.maximum(p, eps)) - np.log(np.maximum(1-p, eps))
      x = logodds[np.isfinite(logodds)]
      log_odds.append(x)

    return log_odds

  domain_clf.train = train_dc
  domain_clf.log_odds = log_odds

  return domain_clf

from datetime import datetime
import contextlib
import os
import pprint


def set_polyak_step_size(opt, loss, grads, max_sps, min_sps, loss_opt):
  c = opt.polyak_c
  g_norm_sq = tf.add_n([tf.reduce_sum(tf.pow(g, 2)) for g in grads])
  step_size = (loss - loss_opt) / (c * g_norm_sq)
  step_size_clip = tf.clip_by_value(step_size, min_sps, max_sps)
  opt.learning_rate.assign(step_size_clip)
  

def fit_loss_iter(loss_fn, optimizer, batched_dataset, epochs=1000,
                  reducer_fn=tf.reduce_mean, compile=True):
  
  use_ployak = getattr(optimizer, 'use_ployak', False)
  max_sps = getattr(optimizer, 'polyak_max_sps', 1.0)
  min_sps = getattr(optimizer, 'polyak_min_sps', 1e-20)
  loss_opt = getattr(optimizer, 'polyak_loss_opt', 0.0)

  def train_step(target_sample):
    with tf.GradientTape() as tape:
      losses = loss_fn(target_sample)
      total_loss = reducer_fn(losses)

    if tf.math.is_nan(total_loss):
      return losses
    else:
      variables = tape.watched_variables()
      grads = tape.gradient(total_loss, variables)

      if use_ployak:
        set_polyak_step_size(optimizer, total_loss, grads, max_sps, min_sps, loss_opt)
      
      optimizer.apply_gradients(zip(grads, variables))
      return losses

  if compile:
    train_step = tf.function(train_step)
  else:
    print('running eagerly!')

  tqit = tqdm.trange(epochs)
  for epoch in tqit:
    for batch in batched_dataset:
      yield train_step(batch)


def add_noise_aug(arr, level=1/128):
  return arr + level*tf.random.uniform(arr.shape, dtype=arr.dtype)

def add_two_sided_noise_aug(arr, level=1/128):
  return arr + level*(2*tf.random.uniform(arr.shape, dtype=arr.dtype) - 1)

def loss_to_bpd(loss, shape):
  ll_per_dim = -loss / np.prod(shape)
  bits_per_dim = -(ll_per_dim - np.log(256)) / np.log(2)
  return bits_per_dim

def ckpt_restore_latest(save_path, mode='existing', **kwargs):
  ckpt = tf.train.Checkpoint(**kwargs)
  ckpt_mng = tf.train.CheckpointManager(ckpt, save_path, 1)
  print('restoring %s : %s' % (list(kwargs.keys()), ckpt_mng.latest_checkpoint))
  restore = ckpt.restore(ckpt_mng.latest_checkpoint)
  if mode == 'existing':
    restore.assert_existing_objects_matched().expect_partial()
  elif mode == 'partial':
    restore.expect_partial()
  elif mode == 'consumed':
    restore.assert_consumed()


def model_summary_str(model, dataset, optimizer):
  glow_flow = model.bijector._bijector
  keys = ['_level', '_level_depth', '_factor', '_glow_step_args']
  dd = {'model': dict({key[1:]: getattr(glow_flow, key) for key in keys})}
  dd['model']['glow_step_args'] = dd['model']['glow_step_args'].copy()
  dd['model']['glow_step_args']['actnorm_scale_norm_bijector'] = dd['model']['glow_step_args']['actnorm_scale_norm_bijector'].name
  dd['model']['glow_step_args']['nvp_scale_norm_bijector'] = dd['model']['glow_step_args']['nvp_scale_norm_bijector'].name
  dd['model']['glow_step_args']['shift_scale_fn_builder'] = dd['model']['glow_step_args']['shift_scale_fn_builder'].__name__

  dd['data'] = {'shape': tuple(next(iter(dataset)).shape)}
  dd['opt'] = optimizer.get_config()
  return pprint.pformat(dd)


def fit_density_model(model, dataset, lr=1e-4, epochs=50,
                      show_every=10, print_every=100, save_every=1000, 
                      clf_every=1000,  
                      tb_path=None, save_path=None, run_path=None, 
                      domain_clf_builder=None, clf_dataset_length=2048,
                      restore_on_start=True,
                      check_numerics=False, compile=True):
  if check_numerics:
    tf.debugging.enable_check_numerics()

  nll_loss = lambda x: -1 * tf.reduce_mean(model.log_prob(x))
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr) if not callable(lr) else lr()
  fit_iter = fit_loss_iter(nll_loss, optimizer, dataset, 
                           epochs=epochs, compile=compile)
  # to init actnorms; when computing log_prob
  single_batch = next(iter(dataset))
  print('init loss', nll_loss(single_batch))

  if run_path is not None:
    tb_path = os.path.join(run_path, 'logs')
    save_path = os.path.join(run_path, 'ckpt')

  if tb_path is not None:
    logdir = os.path.join(tb_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    tb_ctx = file_writer.as_default()
  else:
    tb_ctx = contextlib.nullcontext()

  if save_path is not None:
    step_var = tf.Variable(0)
    ckpt = tf.train.Checkpoint(model=model, step=step_var, opt=optimizer)
    ckpt_mng = tf.train.CheckpointManager(ckpt, save_path, 1)
    if restore_on_start and ckpt_mng.latest_checkpoint:
      print('restoring from ', ckpt_mng.latest_checkpoint)
      ckpt.restore(ckpt_mng.latest_checkpoint)
      print('restored step =', int(step_var))
  else:
    step_var = 0

  if domain_clf_builder is not None:
    clf = domain_clf_builder(single_batch.shape[1:])
    clf_weights = clf.get_weights()
  
  with tb_ctx:
    desc = model_summary_str(model, dataset, optimizer)
    print(desc)
    tf.summary.text('text_summary', desc, step=0)
    start = getattr(model, 'quit_iter_i', int(step_var))
    for iter_i, loss in enumerate(fit_iter, start=start):
      if tf.math.is_nan(loss):
        model.quit_iter_i = iter_i
        print('nan loss, quitting')
        return -1
      
      if iter_i % print_every == 0:
        bpd = float(loss_to_bpd(loss, single_batch.shape[1:]))
        print('%5d %.3f %.3f %g' % (iter_i, float(loss), bpd, float(optimizer.learning_rate.numpy())))

        if tb_path is not None:
          tf.summary.scalar('loss', loss, step=iter_i)
          tf.summary.scalar('bpd', bpd, step=iter_i)
          tf.summary.scalar('lr', optimizer.learning_rate, step=iter_i)
      
      if iter_i % show_every == 0 and tb_path is not None:
        if check_numerics:
          tf.debugging.disable_check_numerics()
        
        tf.summary.image('sample', model.sample(10), step=iter_i)                
        tf.summary.image('inv_real', model.bijector.inverse(single_batch), step=iter_i)

      if iter_i % clf_every == 0 and domain_clf_builder is not None:
        if clf_dataset_length is None:
          clf_dataset_length = len([None for _ in dataset])
        
        sampled_n_batches = clf_dataset_length // single_batch.shape[0] + 1
        sample_batches = [model.sample(single_batch.shape[0]) 
                          for _ in range(sampled_n_batches)]
        sampled_dataset = tf.data.Dataset.from_tensor_slices(sample_batches)
        sampled_unb = sampled_dataset.unbatch().take(clf_dataset_length)
        dataset_unb = dataset.unbatch().take(clf_dataset_length)
        clf.set_weights(clf_weights)
        train_results = clf.train(sampled_unb, dataset_unb)
        print('loss/acc', train_results)
        
        log_odds_s, log_odds_d = clf.log_odds(sampled_unb, dataset_unb)
        tf.summary.histogram('clf/log_odds_s', log_odds_s, step=iter_i)
        tf.summary.histogram('clf/log_odds_d', log_odds_d, step=iter_i)
        tf.summary.scalar('clf/acc', train_results[-1], step=iter_i)

        if check_numerics:
          tf.debugging.enable_check_numerics()

      if (save_path is not None
          and iter_i % save_every == 0 
          and iter_i > int(step_var)):
        print('saving step =', iter_i, 'to', save_path)
        step_var.assign(iter_i)
        ckpt_mng.save()
        print('saved as', ckpt_mng.latest_checkpoint)


def equal_z_mean_reg(model_s, batch_a, gen_t):
  z_a = model_s.bijector.inverse(batch_a)
  z_bt = model_s.bijector.inverse(gen_t)
  mean_z_a = tf.reduce_mean(z_a, axis=0)
  mean_z_bt = tf.reduce_mean(z_bt, axis=0)
  loss = tf.reduce_mean((mean_z_a - mean_z_bt)**2)
  return loss


def lrmf_loss(model_s, model_t, batch_a, batch_b):
  gen_t = model_t.bijector.forward(batch_b)
  nll_s_a = -1 * tf.reduce_mean(model_s.log_prob(batch_a))
  nll_s_t = -1 * tf.reduce_mean(model_s.log_prob(gen_t))
  event_ndims = len(batch_a.shape) - 1
  fjs = model_t.bijector.forward_log_det_jacobian(batch_b, event_ndims)
  neg_fwd_jac = -1 * tf.reduce_mean(fjs)
  fwd_jac_var = tf.reduce_mean((fjs + neg_fwd_jac)**2)
  eq_mean_reg = equal_z_mean_reg(model_s, batch_a, gen_t)
  return (nll_s_a, nll_s_t, neg_fwd_jac, eq_mean_reg, fwd_jac_var)


def reduce_losses_dict(losses, loss_names, loss_weights):
  assert len(loss_names) == len(losses)
  loss_dict = dict(zip(loss_names, losses))
  losses_weighted = [x*loss_weights.get(n, 1.0) 
                     for n, x in loss_dict.items()
                     if loss_weights.get(n, 1.0) > 0]
  total_loss = tf.math.add_n(losses_weighted)
  loss_dict = {'total_loss': total_loss, **loss_dict}
  return loss_dict


def compute_dataset_stats(prior, data_b, mean_nll_a, mean_nll_b):
  prior_logp = np.mean([np.mean(prior.log_prob(x)) for x in data_b])
  dataset_stats = {
      'const': prior_logp + mean_nll_a + mean_nll_b,
      'prior': prior_logp,
      'nll_a': mean_nll_a,
      'nll_b': mean_nll_b
  }
  return dataset_stats


def build_glow_translation_model(shape, **glow_flow_args):
  base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros(np.prod(shape)),
        scale_diag=tf.ones(np.prod(shape)))
  
  base_resh_dist = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=tfp.bijectors.Reshape(event_shape_out=shape, 
                                   event_shape_in=[np.prod(shape)]),
  )

  glow_flow1 = GlowFlow(**glow_flow_args)
  glow_flow2 = GlowFlow(**glow_flow_args)
  inv_glow_flow2 = tfb.Invert(glow_flow2)
  flow = ForwardChain([glow_flow1, inv_glow_flow2])

  glow_t_dist = tfd.TransformedDistribution(
      distribution=base_resh_dist,
      bijector=flow)
  
  return glow_t_dist

def pretrain_t_on_prior(model, batch_size, n_samples, lr=1e-3):
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  nll_loss = lambda x: tf.reduce_mean(model.log_prob(x))
  n_repeats = n_samples // batch_size
  prior_data = [model.distribution.sample(batch_size) for _ in range(n_repeats)]
  train_it = fit_loss_iter(nll_loss, optimizer, prior_data, epochs=1, compile=False)
  for loss_value in train_it:
    print(loss_value)

  print('-- finished pre-training model_t --')


def pretrain_t_on_b(model, dataset_b, epochs, lr=1e-3):
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  nll_loss = lambda x: tf.reduce_mean((model.bijector.forward(x) - x)**2)
  train_it = fit_loss_iter(nll_loss, optimizer, dataset_b, epochs=epochs, compile=False)
  print('staring loss', next(train_it))
  for loss_value in train_it:
    print(loss_value)

  print('end loss', loss_value)


def pretrain_t_on_inverse_checkpoint(model_t):
  glow1 = model_t.bijector.bijectors[0]._bijector
  glow2 = model_t.bijector.bijectors[1]

  assert glow1.flow.bijectors[-2].flow.bijectors[-1].built
  assert glow2.flow.bijectors[-2].flow.bijectors[-1].built

  ckpt = tf.train.Checkpoint(glow=glow1)
  ckpt.save('/tmp/glow_init/ckpt')
  ckpt2 = tf.train.Checkpoint(glow=glow2)
  ckpt2.restore(tf.train.latest_checkpoint('/tmp/glow_init/'))


def init_models_from_data(model_s, model_si, model_t, single_batch_a, single_batch_b):
  # _single_ batch to ensure that actnorm inits optimally for both
  # we can use batch_b instead of model_t.forward(batch_b) since model_t 
  # is initialized to be idx on batch_b
  single_batch_c = tf.concat([single_batch_a, single_batch_b], axis=0)
  print(model_t.bijector.bijectors[1].forward(single_batch_c).shape)
  print(model_t.bijector.bijectors[0].inverse(single_batch_c).shape)
  print(model_s.bijector.inverse(single_batch_c).shape)
  if model_si is not None:
    print(model_si.bijector.inverse(single_batch_c).shape)

def get_glow_config(glow_flow):
  keys = ['_level', '_level_depth', '_factor', '_glow_step_args']
  dd = {'model': dict({key[1:]: getattr(glow_flow, key) for key in keys})}
  dd['model']['glow_step_args'] = dd['model']['glow_step_args'].copy()
  dd['model']['glow_step_args']['actnorm_scale_norm_bijector'] = dd['model']['glow_step_args']['actnorm_scale_norm_bijector'].name
  dd['model']['glow_step_args']['nvp_scale_norm_bijector'] = dd['model']['glow_step_args']['nvp_scale_norm_bijector'].name
  dd['model']['glow_step_args']['shift_scale_fn_builder'] = dd['model']['glow_step_args']['shift_scale_fn_builder'].__name__
  return dd


def lrmf_summary_str(model_s, model_si, model_t, dataset_a, dataset_b, optimizer, loss_weights):
  dd = {
      'model_s': get_glow_config(model_s.bijector._bijector),
      'model_si': get_glow_config(model_si.bijector._bijector) if model_si is not None else None,
      'model_t_0': get_glow_config(model_t.bijector.bijectors[1]),
      'model_t_1': get_glow_config(model_t.bijector.bijectors[0]._bijector),
      'dataset_a': {'shape': tuple(next(iter(dataset_a)).shape)},
      'dataset_b': {'shape': tuple(next(iter(dataset_b)).shape)},
      'opt': optimizer.get_config(),
      'loss_weights': loss_weights
  }
  if  getattr(optimizer, 'use_ployak', False):
    c = optimizer.polyak_c.numpy()
    max_sps = getattr(optimizer, 'polyak_max_sps', 1.0)
    loss_opt = getattr(optimizer, 'polyak_loss_opt', 0.0)
    dd['polyak'] = {'c': c, 'max_sps': max_sps, 'loss_opt': loss_opt}
  else:
    dd['polyak'] = False

  return pprint.pformat(dd)


def with_bits_per_dim(loss_dict, shape):
  loss_dict.update({
      'bpd_a': loss_to_bpd(loss_dict['nll_s_a'], shape),
      'bpd_b': loss_to_bpd(loss_dict['nll_s_t'], shape),
  })
  return loss_dict


def with_other_metrics(loss_dict, dataset_stats):
  loss_dict.update({
      'nll_p/tb_est': (-loss_dict['neg_fwd_jac'] + dataset_stats['nll_b']),
      'nll_p/a': dataset_stats['nll_a'],
      'zm/nll_s_a': loss_dict['nll_s_a'] - dataset_stats['nll_a'],
      'zm/nll_s_tb': loss_dict['nll_s_t'] - dataset_stats['nll_a'],
  })

  loss_dict.update({
    'gap/tb_est': loss_dict['nll_s_t'] - loss_dict['nll_p/tb_est'],
    'gap/a': loss_dict['nll_s_a'] - loss_dict['nll_p/a']
  })

  if 'nll_si_ai' in loss_dict:
      loss_dict.update({
          'nll_pi/b': (dataset_stats['nll_b']),
          'nll_pi/ai_est': (-loss_dict['neg_fwd_jac_ai'] + dataset_stats['nll_a']),
      })
      loss_dict.update({
        'gap/ai_est': loss_dict['nll_si_ai'] - loss_dict['nll_pi/ai_est'],
        'gap/bi': loss_dict['nll_si_b'] - loss_dict['nll_pi/b']
      })

  return loss_dict


def print_stats(iter_i, loss_dict):
  fmt = {'lr': '%g'}
  losses_strs = [('%s = ' % n + fmt.get(n, '%.3f') % v) for n, v in loss_dict.items()]  
  print('%5d' % iter_i, ' '.join(losses_strs))


def grid_merge_imgs(rows, img_batch_size=4):
  flat_rows = [tf.reshape(tf.transpose(x, (1, 0, 2, 3)), 
                          [x.shape[1], -1, x.shape[-1]]) for x in rows]
  single_image = tf.concat(flat_rows, axis=0)
  if img_batch_size is None:
    return single_image
  else:
    n_batches = rows[0].shape[0] // img_batch_size
    with_batch_shape = (single_image.shape[0], n_batches, -1, *single_image.shape[2:])
    with_batch_dim = tf.reshape(single_image, with_batch_shape)
    return tf.transpose(with_batch_dim, (1, 0, 2, 3))


def fit_lrmf_model(model_s, model_t, dataset_a, dataset_b, dataset_stats,
                   lr=1e-4, epochs=50, pretrain_t=None,
                   eq_mean_reg_lam=0, fwd_jac_var_lam=0, 
                   show_every=10, print_every=100, save_every=1000, 
                   clf_every=1000,
                   tb_path=None, save_path=None, run_path=None, 
                   restore_on_start=True,
                   check_numerics=False, compile=True, 
                   domain_clf_builder=None, domain_clf_size=1024,
                   show_panel=None, restart_on_nan_callback=None,
                   supervised_lam=0, supervised_loss=None,
                   model_si=None):
  
  show_panel = show_panel or (lambda *_, **__: None) 
  const = -1 * dataset_stats['nll_a'] - dataset_stats['nll_b']
  if check_numerics:
    tf.debugging.enable_check_numerics()

  const = tf.convert_to_tensor(const, dtype=tf.float32)
  loss_names = ['nll_s_a', 'nll_s_t', 'neg_fwd_jac', 'eq_mean_reg', 
                'fwd_jac_var', 'const']
  loss_weights = {'eq_mean_reg': eq_mean_reg_lam, 
                  'fwd_jac_var': fwd_jac_var_lam,
                  'fwd_jac_var_ai': fwd_jac_var_lam,
                  'eq_mean_reg_ai': eq_mean_reg_lam,
                  'supervised_loss': supervised_lam}

  lrmf_loss_fn = lambda pair: (*lrmf_loss(model_s, model_t, *pair), const)
  loss_reducer_fn = lambda losses: reduce_losses_dict(losses, loss_names, loss_weights)['total_loss']
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr) if not callable(lr) else lr()

  if supervised_lam:
    loss_names.append('supervised_loss')
    lrmf_loss_fn_old = lrmf_loss_fn
    lrmf_loss_fn = lambda pair: (*lrmf_loss_fn_old(pair), supervised_loss(*pair))

  if model_si is not None:
    loss_names.extend(['nll_si_b', 'nll_si_ai', 'neg_fwd_jac_ai', 'eq_mean_reg_ai', 'fwd_jac_var_ai', 'const2'])
    lrmf_loss_fn_old2 = lrmf_loss_fn
    model_t_inv = tfd.TransformedDistribution(model_t.distribution, tfb.Invert(model_t.bijector))
    lrmf_loss_fn = lambda pair: (*lrmf_loss_fn_old2(pair), *lrmf_loss(model_si, model_t_inv, pair[1], pair[0]), const)

  dataset = tf.data.Dataset.zip((dataset_a, dataset_b))
  fit_iter = fit_loss_iter(lrmf_loss_fn, optimizer, dataset, 
                           reducer_fn=loss_reducer_fn,
                           epochs=epochs, compile=compile)
  
  single_batch_a = next(iter(dataset_a))
  single_batch_b = next(iter(dataset_b))

  # to init actnorms
  init_models_from_data(model_s, model_si, model_t, single_batch_a, single_batch_b)

  show_panel(model_t.bijector.forward(single_batch_a), n=4)
  show_panel(model_t.bijector.forward(single_batch_b), n=4)

  losses = lrmf_loss_fn((single_batch_a, single_batch_b))
  loss_dict = reduce_losses_dict(losses, loss_names, loss_weights)
  print('init pre-init', loss_dict)
  
  if pretrain_t == 'prior':
    print('pre-training model_t on prior')
    pretrain_t_on_prior(model_t, 256, 2560)
  elif pretrain_t == 'b':
    print('pre-training model_t on id(b)')
    pretrain_t_on_b(model_t, dataset_b, 1, 1e-4)
  elif pretrain_t == 'ckpt':
    pretrain_t_on_inverse_checkpoint(model_t)
  elif pretrain_t is None:
    pass
  else:
    raise ValueError('pretrain_t must be in [prior, b, ckpt, None]')

  losses = lrmf_loss_fn((single_batch_a, single_batch_b))
  loss_dict = reduce_losses_dict(losses, loss_names, loss_weights)
  print('loss post pre-train b', loss_dict)
  show_panel(model_t.bijector.forward(single_batch_a), n=4)
  show_panel(model_t.bijector.forward(single_batch_b), n=4)

  if run_path is not None:
    tb_path = os.path.join(run_path, 'logs')
    save_path = os.path.join(run_path, 'ckpt')

  if tb_path is not None:
    logdir = os.path.join(tb_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    print('saving tensorboard logs to ', logdir)
    tb_ctx = file_writer.as_default()
  else:
    tb_ctx = contextlib.nullcontext()

  if save_path is not None:
    step_var = tf.Variable(0)
    ckpt_dict = dict(model_s=model_s, model_t=model_t, step=step_var, opt=optimizer)
    if model_si is not None:
        ckpt_dict['model_si'] = model_si

    ckpt = tf.train.Checkpoint(**ckpt_dict)
    ckpt_mng = tf.train.CheckpointManager(ckpt, save_path, 1)
    if restore_on_start and ckpt_mng.latest_checkpoint:
      print('restoring from ', ckpt_mng.latest_checkpoint)
      ckpt.restore(ckpt_mng.latest_checkpoint)
      print('restored step =', int(step_var))
    if restart_on_nan_callback:
      restart_path = ckpt.save(save_path+'_restart')
      print('saved restart ckpt at', restart_path)
  else:
    step_var = 0

  if domain_clf_builder is not None:
    domain_clf_fixed = domain_clf_builder(single_batch_a.shape[1:])
    domain_clf_fixed.train(dataset_a.unbatch(), dataset_b.unbatch())

    domain_clf_adap = domain_clf_builder(single_batch_a.shape[1:])
    clf_adap_init_w = domain_clf_adap.get_weights()

  with tb_ctx:
    desc = lrmf_summary_str(model_s, model_si, model_t, dataset_a, dataset_b, optimizer, loss_weights)
    tf.summary.text('text_summary', desc, step=0)
    print(desc)

    start_iter_i = getattr(model_s, 'quit_iter_i', int(step_var))
    for iter_i, losses in enumerate(fit_iter, start=start_iter_i):
      loss_dict = reduce_losses_dict(losses, loss_names, loss_weights)
      
      if tf.math.is_nan(loss_dict['total_loss']):
        loss_dict = with_bits_per_dim(loss_dict, single_batch_a.shape[1:])
        loss_dict = with_other_metrics(loss_dict, dataset_stats)
        loss_dict['lr'] = float(optimizer.learning_rate.numpy())
        model_s.quit_iter_i = iter_i
        print_stats(iter_i, loss_dict)
        if restart_on_nan_callback is None:
            print('nan loss, quitting')
            return -1
        else:
            print('nan loss, restoring from start')
            ckpt.restore(restart_path)
            restart_on_nan_callback(optimizer)

      if iter_i % print_every == 0:
        tf.summary.eixperimental.set_step(iter_i)
        stat_dict = with_bits_per_dim(loss_dict, single_batch_a.shape[1:])
        stat_dict = with_other_metrics(loss_dict, dataset_stats)
        stat_dict['lr'] = float(optimizer.learning_rate.numpy())
        stat_dict['polyak_c'] = optimizer.polyak_c.numpy() if hasattr(optimizer, 'polyak_c') else -1 
        
        print_stats(iter_i, stat_dict)

        if tb_path is not None:
          for n, l in loss_dict.items():
            tf.summary.scalar(n, l, step=iter_i)
      
      if iter_i % show_every == 0 and tb_path is not None:
        if check_numerics:
          tf.debugging.disable_check_numerics()
        
        tf.summary.experimental.set_step(iter_i)
        
        gen_b = model_t.bijector.forward(single_batch_b)
        gen_inv_a = model_t.bijector.inverse(single_batch_a)
        gen_s_a = model_s.bijector.inverse(single_batch_a)
        gen_s_b = model_s.bijector.inverse(single_batch_b)

        batch_size = len(single_batch_b)
        tf.summary.image('sample_s', grid_merge_imgs(tf.split(model_s.sample(batch_size*2), 2)))
        tf.summary.image('forward_b', grid_merge_imgs([single_batch_b, gen_b]))
        tf.summary.image('inverse_a', grid_merge_imgs([single_batch_a, gen_inv_a]))
        tf.summary.image('inv_s_a', grid_merge_imgs([single_batch_a, gen_s_a])) 
        tf.summary.image('inv_s_tb', grid_merge_imgs([single_batch_b, gen_s_b])) 
        
        if model_si is not None:
          tf.summary.image('sample_si', grid_merge_imgs(tf.split(model_si.sample(batch_size*2), 2)))
        
        if check_numerics:
          tf.debugging.enable_check_numerics()

      if iter_i % clf_every == 0 and domain_clf_builder is not None:
        if check_numerics:
          tf.debugging.disable_check_numerics()

        dataset_a_unbatch = dataset_a.unbatch().take(domain_clf_size)
        dataset_b_clf_size = dataset_b.take(domain_clf_size // single_batch_b.shape[0] + 1)
        gen_b_batches = [model_t.bijector.forward(x) for x in dataset_b_clf_size]
        gen_b_ds = tf.data.Dataset.from_tensor_slices(gen_b_batches).unbatch()
        gen_b_ds_take = gen_b_ds.take(domain_clf_size)
        log_odds_a, log_odds_b = domain_clf_fixed.log_odds(dataset_a_unbatch, gen_b_ds_take)
        tf.summary.histogram('fixed_log_odds/a', log_odds_a)
        tf.summary.histogram('fixed_log_odds/b', log_odds_b)

        domain_clf_adap.set_weights(clf_adap_init_w)
        eval_results = domain_clf_adap.train(dataset_a_unbatch, gen_b_ds_take)
        print('loss/acc', eval_results)
        
        log_odds_a, log_odds_b = domain_clf_adap.log_odds(dataset_a_unbatch, gen_b_ds_take)
        tf.summary.histogram('adapt_log_odds/a', log_odds_a)
        tf.summary.histogram('adapt_log_odds/b', log_odds_b)
        tf.summary.scalar('clf/adapt_acc', eval_results[-1])

        if check_numerics:
          tf.debugging.enable_check_numerics()

      if (save_path is not None
          and iter_i % save_every == 0 
          and iter_i > int(step_var)):
        print('saving step =', iter_i, 'to', save_path)
        step_var.assign(iter_i)
        ckpt_mng.save()
        print('saved as', ckpt_mng.latest_checkpoint)


def tqdm_tfd(dataset):
  return tqdm.tqdm(dataset, total=int(tf.data.experimental.cardinality(dataset)))


def tfds_batch_map(fn, dataset, batch_size, tqdm=True):
  maybe_tqdm = tqdm_tfd if tqdm else lambda x: x
  return tf.data.Dataset.from_tensor_slices(np.concatenate(list(map(
    lambda x: fn(x).numpy(), maybe_tqdm(dataset.batch(batch_size))))))


def write_dataset(dataset, fn):
  serialized_dataset = dataset.map(tf.io.serialize_tensor)
  writer = tf.data.experimental.TFRecordWriter(fn)
  writer.write(serialized_dataset)

def tf_record_tensor_dataset(fn, shape):
  str_ds = tf.data.TFRecordDataset([fn])
  tensor_ds = str_ds.map(lambda s: tf.io.parse_tensor(s, tf.float32))
  reshaped_ds = tensor_ds.map(lambda x: (x.set_shape(shape), x)[1])
  return reshaped_ds

