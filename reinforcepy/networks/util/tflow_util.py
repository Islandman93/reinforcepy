# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# IslandMan93: Some of this code comes from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# ==============================================================================
import tensorflow as tf
import numpy as np
import tflearn


def nn_layer(input_tensor, shape, layer_name, act, conv, stride=None, variable_summaries=True):
    """Reusable code for making a simple neural net layer.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Have to use variable scope because we can have multiple networks that need
    # to reuse these weights
    # https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html
    with tf.variable_scope(layer_name) as scope:
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights, stdv = weight_variable_torch('weights', shape)
            if variable_summaries:
                compute_variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable_torch('biases', shape[-1], stdv)
            if variable_summaries:
                compute_variable_summaries(biases, layer_name + '/biases')
        if conv:
            with tf.name_scope('conv2d'):
                conv_out = tf.nn.conv2d(input_tensor, weights,
                                        strides=[1, stride, stride, 1],
                                        padding='VALID')
                preactivate = tf.nn.bias_add(conv_out, biases)
        else:
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases

        tf.summary.histogram(scope._name_scope + 'pre_activations', preactivate)
        if act is not None:
            activations = act(preactivate, name='activation')
            tf.summary.histogram(scope._name_scope + 'activations', activations)
            return activations
        else:
            return preactivate


def compute_variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def weight_variable_torch(name, shape, init='xavier', uniform=True):
    # conv
    if len(shape) == 4:
        # comes from https://github.com/torch/nn/blob/master/SpatialConvolution.lua#L38
        # need only width height input in channels
        # [filter_height, filter_width, in_channels, out_channels]
        if init == 'he':
            stdv = np.sqrt(2) * np.sqrt(1. / np.prod(shape[0:3]))
        else:
            stdv = 1. / np.sqrt(np.prod(shape[0:3]))
    elif len(shape) == 2:
        # looks like torch linear only looks at input channels
        # https://github.com/torch/nn/blob/master/Linear.lua#L25
        if init == 'he':
            stdv = np.sqrt(2) * np.sqrt(1. / shape[0])
        else:
            stdv = 1. / np.sqrt(shape[0])

    if uniform:
        weight_tensor = tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv))
    else:
        weight_tensor = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=stdv))
    return weight_tensor, stdv


def bias_variable_torch(name, shape, stdv):
    bias_tensor = tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv))
    # bias_tensor = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), trainable=True)
    return bias_tensor


def torch_init(input_tensor):
    shape_ints = [int(x) for x in input_tensor.get_shape()[1:]]
    stdv = 1.0 / np.sqrt(np.prod(shape_ints))
    return tf.random_uniform_initializer(minval=-stdv, maxval=stdv)


def one_hot(select_from_tensor, index_tensor, output_num):
    # Because of https://github.com/tensorflow/tensorflow/issues/206
    # we cannot use numpy like indexing so we convert to a one hot
    # multiply then take the sum over last dim
    # NumPy/Theano select_from_tensor[:, index_tensor]
    one_hot = tf.one_hot(index_tensor, depth=output_num, name='one-hot',
                         on_value=1.0, off_value=0.0, dtype=tf.float32)
    # we reduce sum here because the output could be negative we can't take the max
    # the other indecies will be 0
    return tf.reduce_sum(tf.multiply(select_from_tensor, one_hot), axis=1)


def dueling_to_q_vals(value_tensor, advantage_tensor):
    # dueling redfines Q(s, a) as V(s) + (A(s, a) - mean(A(s)))
    t_advantage_mean = tf.reduce_mean(advantage_tensor, axis=-1, keep_dims=True)
    return value_tensor + (advantage_tensor - t_advantage_mean)


def create_nips_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    out = tflearn.fully_connected(l_hid3, output_num, scope='denseout')
    return out


def create_dueling_nips_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    value_out = tflearn.fully_connected(l_hid3, 1, scope='valueout')
    advantage_out = tflearn.fully_connected(l_hid3, output_num, scope='advantageout')
    return value_out, advantage_out


def create_a3c_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', padding='valid', scope='conv1')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', padding='valid', scope='conv2')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    actor_out = tflearn.fully_connected(l_hid3, output_num, activation='softmax', scope='actorout')
    critic_out = tflearn.fully_connected(l_hid3, 1, activation='linear', scope='criticout')

    return actor_out, critic_out


def global_norm_clip_grads_vars(grads_vars, norm_clip):
    """
    Accepts the list of tuples (gradient, tensor) returned by compute_gradients and applies global norm clipping
    Returns list of tuples (clipped_gradients, tensor)
    """
    # kinda lame that clip by global norm doesn't accept the list of tuples returned from compute_gradients
    # so we unzip then zip
    tensors = [tensor for gradient, tensor in grads_vars]
    grads = [gradient for gradient, tensor in grads_vars]
    clipped_gradients, _ = tf.clip_by_global_norm(grads, norm_clip)  # returns list[tensors], norm
    return zip(clipped_gradients, tensors)


def nstep_rewards_nd(rewards_placeholder, last_reward, q_discount, rewards_shape=[]):
    """ Calculates nstep_rewards for a nd tensor with unkown 1st dimension static size.
    To use a 1d tensor simply don't pass a shape to rewards_shape
    To use a nd tensor pass a shape like [5, 5] to rewards_shape, don't include the None dimension
    Args:
        rewards_placeholder: Placeholder for rewards earned from the environment, should be shape [None]
        last_reward: The estimated reward from the last state_tp1, should be 0 if last state is terminal
            Must be at least 1d, ie: tf.Constant([0.0])
        q_discount: The q discount value (also referred to as lambda) generally ~0.9 - 0.99
    Returns:
        Returns nstep_rewards of shape rewards_placeholder
    """
    # 1d case
    if len(rewards_shape) == 0:
        stack_shape_invariant = tf.TensorShape([None])
        output_shape = [-1]
    # nd case
    else:
        stack_shape_invariant = tf.TensorShape([None] + rewards_shape)
        output_shape = [-1] + rewards_shape

    num_iter = tf.squeeze(tf.shape(rewards_placeholder)[0])
    condition = lambda i, s: tf.greater(i, 0)  # noqa: E731
    def body(counter, stack):  # noqa: E301,E306
        stack = tf.concat([stack, [rewards_placeholder[counter - 1] + stack[-1] * q_discount]], axis=0)
        return (tf.subtract(counter, 1), stack)

    _, tf_nstep_rewards = tf.while_loop(condition, body, (num_iter, last_reward),
                                        shape_invariants=(num_iter.get_shape(), stack_shape_invariant),
                                        parallel_iterations=1, back_prop=False)

    # remove the first item from stack, it will be the rewards_placeholder
    # reverse the nstep rewards, they are computed backward this puts them forward looking
    return tf.reshape(tf_nstep_rewards[1:], output_shape)[::-1]
