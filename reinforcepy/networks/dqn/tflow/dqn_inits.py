import tensorflow as tf
from ...util.tflow_util import nn_layer


def dqn_nips_network(network_parms, training_parms):
    network_parms.required(['input_shape', 'stride', 'output_num'])
    training_parms.required(['minibatch_size', 'discount', 'rms_decay', 'rms_epsilon'])
    # we need to fix the conv input shape from theano (batch, filter, height, width) to
    # tensorflow which is (batch, height, width, filter)
    image_input_shape = network_parms.get('input_shape')
    input_shape = [image_input_shape[1], image_input_shape[2], image_input_shape[0]]
    # Input placeholders
    with tf.name_scope('input'):
        x_input_channel_firstdim = tf.placeholder(tf.uint8, [None] + image_input_shape, name='x-input')
        # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
        x_input = tf.cast(tf.transpose(x_input_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        x_input_tp1_channel_firstdim = tf.placeholder(tf.uint8, [training_parms.get('minibatch_size')] + image_input_shape, name='x-input-tp1')
        # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
        x_input_tp1 = tf.cast(tf.transpose(x_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        x_actions = tf.placeholder(tf.int32, [training_parms.get('minibatch_size')], name='x-actions')
        x_rewards = tf.placeholder(tf.float32, [training_parms.get('minibatch_size')], name='x-rewards')
        x_terminals = tf.placeholder(tf.bool, [training_parms.get('minibatch_size')], name='x-terminals')
        x_discount = training_parms.get('discount')
        tf.image_summary('input', x_input, max_images=10)
        tf.image_summary('input_tp1', x_input_tp1, max_images=10)

    with tf.variable_scope('network-outputs') as scope:
        with tf.name_scope('output-train-t'):
            network_output = create_nips_network(x_input, network_parms, input_shape)

        # must use variable scope to reuse weights
        scope.reuse_variables()

        with tf.name_scope('output-train-tp1'):
            # don't let gradients prop through the tp1 step
            # we also can't recalculate weight/bias summaries or SummaryWriter complains (we don't want to either, they are shared
            network_output_tp1 = tf.stop_gradient(create_nips_network(x_input_tp1, network_parms, input_shape,
                                                                      variable_summaries=False))

    with tf.name_scope('loss'):
        with tf.name_scope('estimated-reward-tp1'):
            one_minus_term = tf.mul(1.0 - tf.cast(x_terminals, tf.float32), x_discount)
            est_rew_tp1 = tf.mul(one_minus_term, tf.reduce_max(network_output_tp1, reduction_indices=1))

        y = x_rewards + est_rew_tp1

        with tf.name_scope('estimated-reward'):
            # Because of https://github.com/tensorflow/tensorflow/issues/206
            # we cannot use numpy like indexing so we convert to a one hot
            # multiply then take the max over last dim
            # NumPy/Theano est_rew = network_output[:, x_actions]
            x_actions_one_hot = tf.one_hot(x_actions, depth=network_parms.get('output_num'), name='one-hot')
            est_rew = tf.reduce_max(tf.mul(network_output, x_actions_one_hot), reduction_indices=1)

        with tf.name_scope('qloss'):
            diff = (y - est_rew)**2.0
            mse = tf.reduce_mean(diff)
        tf.scalar_summary('loss', mse)

    with tf.name_scope('optimizer'):
        tf_learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.RMSPropOptimizer(tf_learning_rate, decay=training_parms.get('rms_decay'), epsilon=training_parms.get('rms_epsilon'))
        train_step = optimizer.minimize(mse)

    # Merge all the summaries
    merged_summaries = tf.merge_all_summaries()

    def train(sess, learning_rate, state, action, reward, state_tp1, terminal, run_summaries=False, **kwargs):
        feed_dict = {x_input_channel_firstdim: state, x_input_tp1_channel_firstdim: state_tp1,
                     x_actions: action, x_rewards: reward, x_terminals: terminal,
                     tf_learning_rate: learning_rate}

        # summaries take time to run, only compute when run_summaries
        if run_summaries:
            op_list = [mse, merged_summaries, train_step]
        else:
            op_list = [mse, train_step]

        return sess.run(op_list, feed_dict=feed_dict, **kwargs)

    def get_output(sess, state):
        feed_dict = {x_input_channel_firstdim: state}
        return sess.run([network_output], feed_dict=feed_dict)

    return train, get_output


def create_nips_network(input_tensor, network_parms, input_shape, variable_summaries=True):
    # [filter_height, filter_width, in_channels, out_channels]
    l_hid1 = nn_layer(input_tensor, [8, 8, input_shape[2], 16], 'conv1', act=tf.nn.relu,
                      conv=True, stride=network_parms.get('stride')[0], variable_summaries=variable_summaries)

    l_hid2 = nn_layer(l_hid1, [4, 4, 16, 32], 'conv2', act=tf.nn.relu,
                      conv=True, stride=network_parms.get('stride')[1], variable_summaries=variable_summaries)

    # flatten for dense layer
    l_hid2_flat = tf.contrib.layers.flatten(l_hid2, scope='conv2flatten')

    # dense layer
    l_hid3 = nn_layer(l_hid2_flat, [l_hid2_flat.get_shape()[1].value, 256], 'dense3', act=tf.nn.relu,
                      conv=False, variable_summaries=variable_summaries)

    # output layer
    l_hid4 = nn_layer(l_hid3, [l_hid3.get_shape()[1].value, network_parms.get('output_num')], 'output', act=None,
                      conv=False, variable_summaries=variable_summaries)
    return l_hid4
