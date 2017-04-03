import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten as tf_flatten
import tflearn
import tflearn.helpers.summarizer as summarizer
from ..target_dqn import TargetDQN
import reinforcepy.networks.util.tflow_util as tf_util


def create_a3c_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    actor_out = tflearn.fully_connected(l_hid3, output_num, activation='softmax', scope='actorout')
    critic_out = tflearn.fully_connected(l_hid3, 1, activation='linear', scope='criticout')

    return actor_out, critic_out, l_hid3


def create_aux_deconv_from_hidden_output(hidden_output, output_num):
    # create aux deconv net
    # the paper says 7x7 but that doesn't work in tensorflow, to get a deconv of 20x20 I think you actually need 9x9
    # also the inverse of l_hid2 would be from 9x9->20x20 so this makes more sense
    # first map lstm output batchsizex256 -> 9x9x32 = 2592
    deconv_linear = tflearn.fully_connected(hidden_output, 2592, activation='relu', scope='deconv-linear')
    deconv_linear = tf.reshape(deconv_linear, (-1, 9, 9, 32))  # since this layer is learned it doesn't matter how we reshape

    deconv_value = tflearn.conv_2d_transpose(deconv_linear, 1, 4, [20, 20], strides=2, padding='valid', scope='deconv-value')
    deconv_advantage = tflearn.conv_2d_transpose(deconv_linear, output_num, 4, [20, 20], strides=2, padding='valid', scope='deconv-advantage')
    return deconv_value, deconv_advantage


class NStepA3CUNREAL(TargetDQN):
    def __init__(self, input_shape, output_num, optimizer=None, network_generator=create_a3c_network, q_discount=0.99,
                 auxiliary_pc_q_discount=0.99, entropy_regularization=0.01, global_norm_clipping=40.0, initial_learning_rate=0.001,
                 learning_rate_decay=None):
        self._entropy_regularization = entropy_regularization
        self.aux_pc_q_discount = auxiliary_pc_q_discount
        super().__init__(input_shape, output_num, None, optimizer=optimizer, network_generator=network_generator,
                         q_discount=q_discount, loss_clipping=None, global_norm_clipping=global_norm_clipping,
                         initial_learning_rate=initial_learning_rate, learning_rate_decay=learning_rate_decay)

    def create_network_graph(self):
        input_shape = self._input_shape
        output_num = self._output_num
        # Input placeholders
        with tf.name_scope('input'):
            # we need to fix the input shape from (batch, filter, height, width) to
            # tensorflow which is (batch, height, width, filter)
            x_input_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            x_input = tf.cast(tf.transpose(x_input_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            x_input_tp1_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input-tp1')
            x_input_tp1 = tf.cast(tf.transpose(x_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            x_actions = tf.placeholder(tf.int32, shape=[None], name='x-actions')
            x_rewards = tf.placeholder(tf.float32, shape=[None], name='x-rewards')
            x_last_state_terminal = tf.placeholder(tf.bool, name='x-last-state-terminal')

            x_input_reward_prediction_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input-reward-pred')
            x_input_aux_rew_pred = tf.cast(tf.transpose(x_input_reward_prediction_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        with tf.variable_scope('network') as var_scope:
            actor_output, critic_output, hidden3_output = self._network_generator(x_input, output_num)
            # flatten the critic_output NOTE: THIS IS VERY IMPORTANT
            # otherwise critic_output will be (batch_size, 1) and all ops with it and x_rewards will create a
            # tensor of shape (batch_size, batch_size)
            critic_output = tf.reshape(critic_output, [-1])

            var_scope.reuse_variables()

            # used to calculate nstep rewards and auxiliary nstep rewards
            _, critic_output_tp1, hidden3_output_tp1 = self._network_generator(x_input_tp1, output_num)

            # summarize a histogram of each action output
            for output_ind in range(output_num):
                summarizer.summarize(actor_output[:, output_ind], 'histogram', 'network-actor-output/{0}'.format(output_ind))
            # summarize critic output
            summarizer.summarize(tf.reduce_mean(critic_output), 'scalar', 'network-critic-output')

            # get the trainable variables for this network, later used to overwrite target network vars
            network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

            # summarize activations
            summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='network'))

            # add network summaries
            summarizer.summarize_variables(train_vars=network_trainables)

            # reward prediction
            l_hid1 = tflearn.conv_2d(x_input_aux_rew_pred, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
            cnn_encoding = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')

        # nstep rewards
        with tf.name_scope('nstep_reward'):
            estimated_reward = tf.cond(x_last_state_terminal, lambda: tf.constant([0.0]), lambda: critic_output_tp1[-1])
            tf_nstep_rewards = tf_util.nstep_rewards_nd(x_rewards, estimated_reward, self._q_discount)

        # caculate losses
        with tf.name_scope('loss'):
            with tf.name_scope('critic-reward-diff'):
                critic_diff = tf.subtract(critic_output, tf_nstep_rewards)

            with tf.name_scope('log-of-actor-policy'):
                # Because of https://github.com/tensorflow/tensorflow/issues/206
                # we cannot use numpy like indexing so we convert to a one hot
                # multiply then take the max over last dim
                # NumPy/Theano est_rew = network_output[:, x_actions]
                x_actions_one_hot = tf.one_hot(x_actions, depth=output_num, name='one-hot',
                                               on_value=1.0, off_value=0.0, dtype=tf.float32)
                # we reduce sum here because the output could be negative we can't take the max
                # the other indecies will be 0
                log_policy = tf.log(actor_output + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, x_actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(actor_output, log_policy))
                summarizer.summarize(actor_entropy, 'scalar', 'actor-entropy')

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                summarizer.summarize(actor_loss, 'scalar', 'actor-loss')

            with tf.name_scope('critic-loss'):
                critic_loss = tf.nn.l2_loss(critic_diff)
                summarizer.summarize(critic_loss, 'scalar', 'critic-loss')
                value_replay_critic_loss_summary = tf.summary.scalar('value-replay-critic-loss', critic_loss)

            with tf.name_scope('total-loss'):
                # NOTICE: we are summing gradients
                # NOTE: we are maximizing entropy
                # We want the network to not be sure of it's actions (entropy is highest with outputs not at 0 or 1)
                # https://www.wolframalpha.com/input/?i=log(x)+*+x
                total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self._entropy_regularization))
                summarizer.summarize(total_loss, 'scalar', 'total-loss')

        with tf.variable_scope('aux-deconv-output') as var_scope:
            deconv_value, deconv_advantage = create_aux_deconv_from_hidden_output(hidden3_output, output_num)
            var_scope.reuse_variables()
            deconv_value_tp1, deconv_advantage_tp1 = create_aux_deconv_from_hidden_output(hidden3_output_tp1, output_num)

        with tf.name_scope('pixel-control'):
            # add input for states_tp1
            aux_pc_input_tp1_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='aux-pc-input-tp1')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            aux_pc_input_tp1 = tf.cast(tf.transpose(aux_pc_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0

            # crop to 80x80 TODO: assumes 84x84
            states_cropped = x_input[:, 2:82, 2:82, :]
            states_tp1_cropped = aux_pc_input_tp1[:, 2:82, 2:82, :]
            # extract 20x20 4x4x4 (flattened) image patches
            # for a batch_sizex80x80x4 image this gives a batch_sizex20x20x(4h*4w*4c)
            states_patches = tf.extract_image_patches(states_cropped, [1, 4, 4, 1], [1, 4, 4, 1], [1, 1, 1, 1], padding='VALID')
            states_tp1_patches = tf.extract_image_patches(states_tp1_cropped, [1, 4, 4, 1], [1, 4, 4, 1], [1, 1, 1, 1], padding='VALID')
            # the average abs distance is taken over pixels and channels which is the last dimension
            reward_pixel_difference = tf.reduce_mean(tf.abs(states_patches - states_tp1_patches), axis=-1)
            # now we have rewards over batch_sizex20x20
            reward_pixel_diff_summary = tf.summary.image('pixel-difference', tf.expand_dims(reward_pixel_difference, axis=-1))
            deconv_value_summary = tf.summary.image('deconv-value', deconv_value)
            for i in range(output_num):
                if i == 0:
                    deconv_advantage_summary = tf.summary.image('deconv-advantage-{0}'.format(i),
                                                                tf.expand_dims(deconv_advantage[:, :, :, i], axis=-1))
                else:
                    deconv_advantage_summary = tf.summary.merge([deconv_advantage_summary, tf.summary.image('deconv-advantage-{0}'.format(i),
                                                                 tf.expand_dims(deconv_advantage[:, :, :, i], axis=-1))])

            # get the dueling Q output values, V(s) + (A(s,a) - mean(A(s)))
            deconv_q_values = tf_util.dueling_to_q_vals(deconv_value, deconv_advantage)
            # shape is batch_sizex20x20xoutput_num

            # get deconv_q_values(s,a) from actions, sum over last dimension to get the one hot
            deconv_q_s_a = tf.reduce_sum(tf.multiply(deconv_q_values, x_actions_one_hot), axis=-1)
            # shape is batch_sizex20x20

            # deconv_tp1 get estimated reward from q_vals, max(V(s) + (A(s,a) - mean(A(s))), axis=-1)
            deconv_q_values_tp1 = tf_util.dueling_to_q_vals(deconv_value_tp1, deconv_advantage_tp1)
            # shape is batch_sizex20x20xoutput_num, reduce_max last dimension
            deconv_est_reward_tp1 = tf.reduce_max(deconv_q_values_tp1, axis=-1, keep_dims=True)
            # deconv_est_reward_tp1[-1] is shape 20x20x1 need to convert to 1x20x20
            deconv_value_est_reward = tf.transpose(deconv_est_reward_tp1[-1], [2, 0, 1])
            estimated_reward = tf.cond(x_last_state_terminal,
                                       lambda: tf.constant(np.zeros((1, 20, 20)), dtype=tf.float32),
                                       lambda: deconv_value_est_reward)
            tf_aux_pc_nstep_rewards = tf_util.nstep_rewards_nd(reward_pixel_difference, estimated_reward,
                                                               self.aux_pc_q_discount, rewards_shape=[20, 20])

            aux_pixel_loss_not_agg = tf_flatten(tf.stop_gradient(tf_aux_pc_nstep_rewards) - deconv_q_s_a)
            # not sure if original paper uses mse or mse * 0.5
            # TODO: not sure if gradients are summed or meaned
            aux_pixel_loss_weight_placeholder = tf.placeholder(tf.float32)
            # aux_pixel_loss = tf.reduce_sum(tf.reduce_mean(tf.square(aux_pixel_loss_not_agg), axis=1))
            aux_pixel_loss = tf.reduce_sum(tf.square(aux_pixel_loss_not_agg)) * aux_pixel_loss_weight_placeholder
            aux_pixel_summaries = tf.summary.merge([reward_pixel_diff_summary, deconv_value_summary,
                                                    deconv_advantage_summary, tf.summary.scalar('aux-pixel-loss', aux_pixel_loss)])

        with tf.name_scope('reward-prediction'):
            cnn_encoding = tf.reshape(cnn_encoding, (1, 3*32*9*9))
            rp_fc4 = tflearn.fully_connected(cnn_encoding, 128, activation='relu', scope='rp-fc4')
            reward_prediction = tflearn.fully_connected(rp_fc4, 3, activation='softmax', scope='reward-pred-output')
            # TODO: this is hack because rewards are clipped to -1 and 1
            one_hot_reward_classes = tf.one_hot(tf.cast(x_rewards, tf.int32) + 1, 3, on_value=1.0, off_value=0.0, dtype=tf.float32)
            rp_loss = tf.reduce_sum(tflearn.categorical_crossentropy(reward_prediction, one_hot_reward_classes))
            reward_prediction_loss_summary = tf.summary.scalar('reward-prediction-loss', rp_loss)

        # optimizer
        with tf.name_scope('shared-optimizer'):
            tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = self._optimizer_fn(learning_rate=tf_learning_rate)
            # only train the network vars
            with tf.name_scope('compute-clip-grads'):
                gradients = optimizer.compute_gradients(total_loss)
                clipped_grads_tensors = tf_util.global_norm_clip_grads_vars(gradients, self.global_norm_clipping)
                tf_train_step = optimizer.apply_gradients(clipped_grads_tensors)
                # summarizer.summarize_gradients(clipped_grads_tensors)
            # TODO: it's unknown whether we keep the same rmsprop vars for auxiliary tasks
            # we could create another optimizer that stores separate vars for each
            with tf.name_scope('auxiliary-value-replay-update'):
                # value replay is in fact just the critic loss, it's questionable whether gradients are
                # still multiplied by 0.5, but is the most likely scenario so we just reuse that var
                gradients = optimizer.compute_gradients(critic_loss)
                clipped_grads_tensors = tf_util.global_norm_clip_grads_vars(gradients, self.global_norm_clipping)
                tf_train_step_auxiliary_value_replay = optimizer.apply_gradients(clipped_grads_tensors)
            # TODO: it's unknown whether we keep the same rmsprop vars for auxiliary tasks
            # we could create another optimizer that stores separate vars for each
            with tf.name_scope('auxiliary-pixel-loss-update'):
                gradients = optimizer.compute_gradients(aux_pixel_loss)
                clipped_grads_tensors = tf_util.global_norm_clip_grads_vars(gradients, self.global_norm_clipping)
                tf_train_step_auxiliary_pixel_loss = optimizer.apply_gradients(clipped_grads_tensors)
            # TODO: it's unknown whether we keep the same rmsprop vars for auxiliary tasks
            # we could create another optimizer that stores separate vars for each
            with tf.name_scope('auxiliary-reward-prediciton-update'):
                gradients = optimizer.compute_gradients(rp_loss)
                clipped_grads_tensors = tf_util.global_norm_clip_grads_vars(gradients, self.global_norm_clipping)
                tf_train_step_auxiliary_reward_pred = optimizer.apply_gradients(clipped_grads_tensors)

            # tf learn auto merges all summaries so we just have to grab the last one
            tf_summaries = summarizer.summarize(tf_learning_rate, 'scalar', 'learning-rate')
            auxiliary_summaries = tf.summary.merge([aux_pixel_summaries, value_replay_critic_loss_summary])

        # function to get network output
        def get_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state}
            output = sess.run(actor_output, feed_dict=feed_dict)
            return get_action_from_probabilities(output[0])

        # function to get mse feed dict
        def train_step(sess, states, actions, rewards, states_tp1, terminals, global_step=0, summaries=False):
            self.anneal_learning_rate(global_step)

            # nstep calculate TD reward
            if sum(terminals) > 1:
                raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

            all_states_plus_tp1 = np.concatenate((np.expand_dims(states[0], axis=0), states_tp1), axis=0)
            feed_dict = {x_input_channel_firstdim: states, x_actions: actions, x_rewards: rewards,
                         x_input_tp1_channel_firstdim: all_states_plus_tp1, x_last_state_terminal: terminals[-1],
                         tf_learning_rate: self.current_learning_rate}

            if summaries:
                return sess.run([tf_summaries, tf_train_step], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step], feed_dict=feed_dict)

        # value replay
        def train_auxiliary_value_replay(sess, states, rewards, states_tp1, terminals, task_weight=1, summaries=False):
            # nstep calculate TD reward
            if sum(terminals) > 1:
                raise ValueError('Value replay reward for mutiple terminal states in a batch is undefined')

            all_states_plus_tp1 = np.concatenate((np.expand_dims(states[0], axis=0), states_tp1), axis=0)
            feed_dict = {x_input_channel_firstdim: states, x_rewards: rewards,
                         x_input_tp1_channel_firstdim: all_states_plus_tp1, x_last_state_terminal: terminals[-1],
                         tf_learning_rate: self.current_learning_rate * task_weight}

            if summaries:
                return sess.run([value_replay_critic_loss_summary, tf_train_step_auxiliary_value_replay], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step_auxiliary_value_replay], feed_dict=feed_dict)

        # reward prediction
        def train_auxiliary_reward_preditiction(sess, states, rewards, task_weight=1, summaries=False):
            feed_dict = {x_input_reward_prediction_channel_firstdim: states, x_rewards: rewards,
                         tf_learning_rate: self.current_learning_rate * task_weight}
            if summaries:
                return sess.run([reward_prediction_loss_summary, tf_train_step_auxiliary_reward_pred], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step_auxiliary_reward_pred], feed_dict=feed_dict)

        # pixel control
        def train_auxiliary_pixel_control(sess, states, actions, states_tp1, terminals, task_weight=0.0007, summaries=False):
            all_states_plus_tp1 = np.concatenate((np.expand_dims(states[0], axis=0), states_tp1), axis=0)
            feed_dict = {x_input_channel_firstdim: states, x_actions: actions,
                         x_input_tp1_channel_firstdim: all_states_plus_tp1, x_last_state_terminal: terminals[-1],
                         aux_pc_input_tp1_channel_firstdim: states_tp1, aux_pixel_loss_weight_placeholder: task_weight,
                         tf_learning_rate: self.current_learning_rate * task_weight}
            if summaries:
                return sess.run([aux_pixel_summaries, tf_train_step_auxiliary_pixel_loss], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step_auxiliary_pixel_loss], feed_dict=feed_dict)

        def train_auxiliary_vr_pc(sess, states, actions, rewards, states_tp1, terminals, pixel_control_weight=0.0007, summaries=False):
            # nstep calculate TD reward
            if sum(terminals) > 1:
                raise ValueError('Value replay reward for mutiple terminal states in a batch is undefined')

            all_states_plus_tp1 = np.concatenate((np.expand_dims(states[0], axis=0), states_tp1), axis=0)
            feed_dict = {x_input_channel_firstdim: states, x_actions: actions, x_rewards: rewards,
                         x_input_tp1_channel_firstdim: all_states_plus_tp1, x_last_state_terminal: terminals[-1],
                         aux_pc_input_tp1_channel_firstdim: states_tp1, aux_pixel_loss_weight_placeholder: pixel_control_weight,
                         tf_learning_rate: self.current_learning_rate}
            if summaries:
                return sess.run([auxiliary_summaries, tf_train_step_auxiliary_value_replay, tf_train_step_auxiliary_pixel_loss],
                                feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step_auxiliary_value_replay, tf_train_step_auxiliary_pixel_loss], feed_dict=feed_dict)

        self._get_output = get_output
        self._train_step = train_step
        self._train_auxiliary_reward_preditiction = train_auxiliary_reward_preditiction
        self._train_auxiliary_value_replay = train_auxiliary_value_replay
        self._train_auxiliary_pixel_control = train_auxiliary_pixel_control
        self._train_all_auxiliary = train_auxiliary_vr_pc
        self._save_variables = network_trainables

    def train_step(self, state, action, reward, state_tp1, terminal, global_step=None, summaries=False):
        return self._train_step(self.tf_session, state, action, reward, state_tp1, terminal, global_step=global_step,
                                summaries=summaries)

    def train_auxiliary_value_replay(self, state, reward, state_tp1, terminal, summaries=False):
        return self._train_auxiliary_value_replay(self.tf_session, state, reward, state_tp1, terminal, summaries=summaries)

    def train_auxiliary_pixel_control(self, state, action, state_tp1, terminal, summaries=False):
        return self._train_auxiliary_pixel_control(self.tf_session, state, action, state_tp1, terminal, summaries=summaries)

    def train_auxiliary_reward_preditiction(self, states, rewards, summaries=False):
        return self._train_auxiliary_reward_preditiction(self.tf_session, states, rewards, summaries=summaries)

    def train_auxiliary_vr_pc(self, state, action, reward, state_tp1, terminal, summaries=False):
        # Much faster than training individually, just one gpu copy then free GIL
        # this uses the same data for value replay and pixel control but that shouldn't be a problem
        return self._train_all_auxiliary(self.tf_session, state, action, reward, state_tp1, terminal, summaries=summaries)


def get_action_from_probabilities(cnn_action_probabilities):
        """
        Get action according to policy probabilities
        REF: https://github.com/coreylynch/async-rl/blob/master/a3c.py#L52
        https://github.com/muupan/async-rl/blob/master/policy_output.py#L26
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        cnn_action_probabilities = cnn_action_probabilities - np.finfo(np.float32).epsneg
        # Useful numpy function ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
        sample = np.random.multinomial(1, cnn_action_probabilities)
        # since we only sample once, sample will look like a one hot array
        action_index = int(np.nonzero(sample)[0])  # numpy where returns an array of length 1, we just want the first
        return action_index
