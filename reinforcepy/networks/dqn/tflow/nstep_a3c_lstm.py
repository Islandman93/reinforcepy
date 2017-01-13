from copy import deepcopy
import numpy as np
import tensorflow as tf
import tflearn
import tflearn.helpers.summarizer as summarizer
from .target_dqn import TargetDQN


def create_a3c_lstm_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')

    # reshape l_hid3 to lstm usable shape (1, batch_size, 256)
    l_hid3_reshape = tf.reshape(l_hid3, [1, -1, 256])

    # have to custom make the lstm output here to use tf.nn.dynamic_rnn
    l_lstm = tflearn.BasicLSTMCell(256)
    # BasicLSTMCell lists state size as tuple so we need to pass tuple into dynamic_rnn
    lstm_state_size = tuple([[1, x] for x in l_lstm.state_size])
    # has to specifically be the same type tf.python.ops.rnn_cell.LSTMStateTuple
    from tensorflow.python.ops.nn import rnn_cell as _rnn_cell
    initial_lstm_state = _rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=lstm_state_size[0], name='initial_lstm_state1'),
                                                  tf.placeholder(tf.float32, shape=lstm_state_size[1], name='initial_lstm_state2'))
    # dynamically get the sequence length
    sequence_length = tf.reshape(tf.shape(l_hid3)[0], [1])
    l_lstm4, new_lstm_state = tf.nn.dynamic_rnn(l_lstm, l_hid3_reshape,
                                                initial_state=initial_lstm_state, sequence_length=sequence_length,
                                                time_major=False, scope='lstm4')

    # reshape lstm back to (batch_size, 256)
    l_lstm4_reshape = tf.reshape(l_lstm4, [-1, 256])
    actor_out = tflearn.fully_connected(l_lstm4_reshape, output_num, activation='softmax', scope='actorout')
    critic_out = tflearn.fully_connected(l_lstm4_reshape, 1, activation='linear', scope='criticout')

    return actor_out, critic_out, initial_lstm_state, new_lstm_state


class NStepA3CLSTM(TargetDQN):
    def __init__(self, input_shape, output_num, optimizer=None, network_generator=create_a3c_lstm_network, q_discount=0.99,
                 entropy_regularization=0.01, global_norm_clipping=40, initial_learning_rate=0.001, learning_rate_decay=None):
        self._entropy_regularization = entropy_regularization
        self.prev_lstm_state = None
        super().__init__(input_shape, output_num, None, optimizer=optimizer, network_generator=network_generator,
                         q_discount=q_discount, loss_clipping=None, global_norm_clipping=global_norm_clipping,
                         initial_learning_rate=initial_learning_rate, learning_rate_decay=learning_rate_decay)
        self.reset_lstm_state()

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
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            x_actions = tf.placeholder(tf.int32, shape=[None], name='x-actions')
            x_rewards = tf.placeholder(tf.float32, shape=[None], name='x-rewards')

        with tf.variable_scope('network'):
            actor_output, critic_output, initial_lstm_state, new_lstm_state = self._network_generator(x_input, output_num)
            # flatten the critic_output NOTE: THIS IS VERY IMPORTANT
            # otherwise critic_output will be (batch_size, 1) and all ops with it and x_rewards will create a
            # tensor of shape (batch_size, batch_size)
            critic_output = tf.reshape(critic_output, [-1])

            # # summarize a histogram of each action output
            # for output_ind in range(output_num):
            #     summarizer.summarize(actor_output[:, output_ind], 'histogram', 'network-actor-output/{0}'.format(output_ind))
            # # summarize critic output
            # summarizer.summarize(tf.reduce_mean(critic_output), 'scalar', 'network-critic-output')

            # # get the trainable variables for this network, later used to overwrite target network vars
            network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

            # # summarize activations
            # summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='network'))

            # # add network summaries
            # summarizer.summarize_variables(train_vars=network_trainables)

        # caclulate losses
        with tf.name_scope('loss'):
            with tf.name_scope('critic-reward-diff'):
                critic_diff = tf.sub(critic_output, x_rewards)

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
                log_policy_one_hot = tf.mul(log_policy, x_actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, reduction_indices=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.mul(actor_output, log_policy))
                summarizer.summarize(actor_entropy, 'scalar', 'actor-entropy')

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.mul(log_policy_action, tf.stop_gradient(critic_diff)))
                summarizer.summarize(actor_loss, 'scalar', 'actor-loss')

            with tf.name_scope('critic-loss'):
                critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
                summarizer.summarize(critic_loss, 'scalar', 'critic-loss')

            with tf.name_scope('total-loss'):
                # NOTICE: we are summing gradients
                # NOTE: we are maximizing entropy
                # We want the network to not be sure of it's actions (entropy is highest with outputs not at 0 or 1)
                # https://www.wolframalpha.com/input/?i=log(x)+*+x
                total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self._entropy_regularization))
                summarizer.summarize(total_loss, 'scalar', 'total-loss')

        # optimizer
        with tf.name_scope('shared-optimizer'):
            tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = self._optimizer_fn(learning_rate=tf_learning_rate)
            # only train the network vars
            with tf.name_scope('compute-clip-grads'):
                gradients = optimizer.compute_gradients(total_loss)
                # gradients are stored as a tuple, (gradient, tensor the gradient corresponds to)
                # kinda lame that clip by global norm doesn't accept the list of tuples returned from compute_gradients
                # so we unzip then zip
                tensors = [tensor for gradient, tensor in gradients]
                grads = [gradient for gradient, tensor in gradients]
                clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)  # returns list[tensors], norm
                clipped_grads_tensors = zip(clipped_gradients, tensors)
                tf_train_step = optimizer.apply_gradients(clipped_grads_tensors)
                # tflearn smartly knows how gradients are stored so we just pass in the list of tuples
                # summarizer.summarize_gradients(clipped_grads_tensors)

            # tf learn auto merges all summaries so we just have to grab the last one
            tf_summaries = summarizer.summarize(tf_learning_rate, 'scalar', 'learning-rate')

        # function to get network output
        def get_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state, initial_lstm_state: self.prev_lstm_state}
            output, lstm_state = sess.run([actor_output, new_lstm_state], feed_dict=feed_dict)
            self.prev_lstm_state = lstm_state
            return get_action_from_probabilities(output[0])

        # function to get mse feed dict
        def train_step(sess, states, actions, rewards, states_tp1, terminals, lstm_state, global_step=0, summaries=False):
            self.anneal_learning_rate(global_step)

            # nstep calculate TD reward
            if sum(terminals) > 1:
                raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

            # last state not terminal need to query target network
            curr_reward = 0
            if not terminals[-1]:
                target_feed_dict = {x_input_channel_firstdim: [states_tp1[-1]]}  # make a list to add back the first dim (needs to be 4 dims)
                curr_reward = max(sess.run(critic_output, feed_dict=target_feed_dict))

            # get bootstrap estimate of last state_tp1
            td_rewards = []
            for reward in reversed(rewards):
                curr_reward = reward + self._q_discount * curr_reward
                td_rewards.append(curr_reward)
            # td rewards is computed backward but other lists are stored forward so need to reverse
            td_rewards = list(reversed(td_rewards))
            feed_dict = {x_input_channel_firstdim: states, x_actions: actions, x_rewards: td_rewards,
                            tf_learning_rate: self.current_learning_rate, initial_lstm_state: lstm_state}

            if summaries:
                return sess.run([tf_summaries, tf_train_step], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step], feed_dict=feed_dict)

        def reset_lstm_state(new_state=None):
            if new_state is not None:
                self.prev_lstm_state = new_state
            else:
                self.prev_lstm_state = (np.zeros((1, 256)), np.zeros((1, 256)))

        self._get_output = get_output
        self._train_step = train_step
        self._save_variables = network_trainables
        self.reset_lstm_state = reset_lstm_state

    def train_step(self, state, action, reward, state_tp1, terminal, lstm_state=None, global_step=None, summaries=False):
        return self._train_step(self.tf_session, state, action, reward, state_tp1, terminal, lstm_state=lstm_state, global_step=global_step, summaries=summaries)

    def get_lstm_state(self):
        return deepcopy(self.prev_lstm_state)


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
