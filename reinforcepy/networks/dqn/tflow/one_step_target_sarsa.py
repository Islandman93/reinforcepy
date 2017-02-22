from functools import partial
import tensorflow as tf
import tflearn
import tflearn.helpers.summarizer as summarizer


# TODO: refactor this with target dqn?
class OneStepTargetSARSA:
    def __init__(self, input_shape, output_num, optimizer=None, network_generator=None, q_discount=0.99, loss_clipping=1):
        self.tf_session = None
        self.tf_graph = None
        self.saver = None

        # these functions are created by create_network_graph
        self._get_output = None
        self._train_step = None
        self._update_target_network = None

        # if optimizer is none use default rms prop
        if optimizer is None:
            optimizer = partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)
        elif not callable(optimizer):
            raise AttributeError('Optimizer must be callable. EX: partial(tf.train.RMSPropOptimizer)')

        # if network is none use default nips
        if network_generator is None:
            network_generator = create_nips_network
        elif not callable(network_generator):
            raise AttributeError('Network Generator must be callable. EX: create_nips_network(input_tensor, output_num)')

        with tf.Graph().as_default() as graph:
            self.tf_graph = graph
            self.create_network_graph(input_shape, output_num, network_generator, q_discount, optimizer, loss_clipping=loss_clipping)
            self.init_tf_session()
            self.update_target_network()

    def init_tf_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=config)
        self.tf_session.run(tf.global_variables_initializer())

    def create_network_graph(self, input_shape, output_num, network_generator, q_discount, optimizer, loss_clipping):
        # Input placeholders
        with tf.name_scope('input'):
            # we need to fix the input shape from (batch, filter, height, width) to
            # tensorflow which is (batch, height, width, filter)
            x_input_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            x_input = tf.cast(tf.transpose(x_input_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            x_input_tp1_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input-tp1')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            x_input_tp1 = tf.cast(tf.transpose(x_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            x_actions = tf.placeholder(tf.int32, shape=[None], name='x-actions')
            # TODO: SARSA only change
            x_actions_tp1 = tf.placeholder(tf.int32, shape=[None], name='x-actions-tp1')
            x_rewards = tf.placeholder(tf.float32, shape=[None], name='x-rewards')
            x_terminals = tf.placeholder(tf.bool, shape=[None], name='x-terminals')
            x_discount = q_discount

        # Target network does not reuse variables. so we use two different variable scopes
        with tf.variable_scope('network'):
            network_output = network_generator(x_input, output_num)

            # summarize a histogram of each action output
            for output_ind in range(output_num):
                summarizer.summarize(network_output[:, output_ind], 'histogram', 'network-output/{0}'.format(output_ind))

            # get the trainable variables for this network, later used to overwrite target network vars
            network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

            # summarize activations
            summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='network'))

            # add network summaries
            summarizer.summarize_variables(train_vars=network_trainables)

        with tf.variable_scope('target-network'):
            target_network_output = network_generator(x_input_tp1, output_num)

            # get trainables for target network, used in assign op for the update target network step
            target_network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target-network')

            # summarize activations
            summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='target-network'))

            # add network summaries
            summarizer.summarize_variables(train_vars=target_network_trainables)

        # update target network with network variables
        with tf.name_scope('update-target-network'):
            update_target_network_ops = [target_v.assign(v) for v, target_v in zip(network_trainables, target_network_trainables)]

        # caclulate QLoss
        with tf.name_scope('loss'):
            with tf.name_scope('estimated-reward-tp1'):
                one_minus_term = tf.multiply(1.0 - tf.cast(x_terminals, tf.float32), x_discount)
                # TODO: SARSA only change
                # Sarsa uses the q estimate of the next state given action_tp1. Not the max
                # We must convert to one hot same as below
                # NumPy/Theano est_rew_tp1 = network_output[:, x_actions_tp1]
                x_actions_tp1_one_hot = tf.one_hot(x_actions_tp1, depth=output_num, name='one-hot-tp1',
                                                   on_value=1.0, off_value=0.0, dtype=tf.float32)
                # we reduce sum here because the output could be negative we can't take the max
                # the other indecies will be 0
                network_est_rew_tp1 = tf.reduce_sum(tf.multiply(target_network_output, x_actions_tp1_one_hot), axis=1)
                est_rew_tp1 = tf.multiply(one_minus_term, network_est_rew_tp1)

            y = x_rewards + tf.stop_gradient(est_rew_tp1)

            with tf.name_scope('estimated-reward'):
                # Because of https://github.com/tensorflow/tensorflow/issues/206
                # we cannot use numpy like indexing so we convert to a one hot
                # multiply then take the max over last dim
                # NumPy/Theano est_rew = network_output[:, x_actions]
                x_actions_one_hot = tf.one_hot(x_actions, depth=output_num, name='one-hot',
                                               on_value=1.0, off_value=0.0, dtype=tf.float32)
                # we reduce sum here because the output could be negative we can't take the max
                # the other indecies will be 0
                est_rew = tf.reduce_sum(tf.multiply(network_output, x_actions_one_hot), axis=1)

            with tf.name_scope('qloss'):
                # clip loss but keep linear past clip bounds
                # REFS: https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/q_network.py#L108
                # https://github.com/Jabberwockyll/deep_rl_ale/blob/master/q_network.py#L241
                diff = y - est_rew

                if loss_clipping > 0.0:
                    abs_diff = tf.abs(diff)
                    # same as min(diff, loss_clipping) because diff can never be negative (definition of abs value)
                    quadratic_part = tf.clip_by_value(abs_diff, 0.0, loss_clipping)
                    linear_part = abs_diff - quadratic_part
                    loss = (0.5 * tf.square(quadratic_part)) + (loss_clipping * linear_part)
                else:
                    # But why multiply the loss by 0.5 when not clipping? https://groups.google.com/forum/#!topic/deep-q-learning/hKK0ZM_OWd4
                    loss = 0.5 * tf.square(diff)
                # NOTICE: we are summing gradients
                error = tf.reduce_sum(loss)
            summarizer.summarize(error, 'scalar', 'loss')

        # optimizer
        with tf.name_scope('shared-optimizer'):
            tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = optimizer(learning_rate=tf_learning_rate)
            # only train the network vars not the target network
            tf_train_step = optimizer.minimize(error, var_list=network_trainables)

            # tf learn auto merges all summaries so we just have to grab the last output
            tf_summaries = summarizer.summarize(tf_learning_rate, 'scalar', 'learning-rate')

        # function to get network output
        def get_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state}
            return sess.run([network_output], feed_dict=feed_dict)

        # function to get mse feed dict
        def train_step(sess, current_learning_rate, state, action, reward, state_tp1, action_tp1, terminal, summaries=False):
            feed_dict = {x_input_channel_firstdim: state, x_input_tp1_channel_firstdim: state_tp1,
                         x_actions: action, x_actions_tp1: action_tp1, x_rewards: reward, x_terminals: terminal,
                         # TODO: SARSA only change action_tp1
                         tf_learning_rate: current_learning_rate}
            if summaries:
                return sess.run([tf_summaries, tf_train_step], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step], feed_dict=feed_dict)

        def update_target_net(sess):
            return sess.run([update_target_network_ops])

        self._get_output = get_output
        self._train_step = train_step
        self._update_target_network = update_target_net
        self.saver = tf.train.Saver(var_list=network_trainables)

    def get_output(self, x):
        return self._get_output(self.tf_session, x)

    # TODO: SARSA only change action_tp1
    def train_step(self, current_learning_rate, state, action, reward, state_tp1, action_tp1, terminal, summaries=False):
        return self._train_step(self.tf_session, current_learning_rate, state, action, reward, state_tp1, action_tp1, terminal, summaries=summaries)

    def update_target_network(self):
        self._update_target_network(self.tf_session)

    def save(self, *args, **kwargs):
        self.saver.save(self.tf_session, *args, **kwargs)

    def load(self, path):
        self.saver.restore(self.tf_session, path)


def create_nips_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    out = tflearn.fully_connected(l_hid3, output_num, scope='denseout')

    return out
