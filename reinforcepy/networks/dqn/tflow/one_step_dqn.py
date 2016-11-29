import logging
from functools import partial
import tensorflow as tf
import tflearn
import tflearn.helpers.summarizer as summarizer
import reinforcepy.networks.util.tflow_util as tf_util
from ..base_network import BaseNetwork


class OneStepDQN(BaseNetwork):
    def __init__(self, input_shape, output_num, optimizer=None, network_generator=None, q_discount=0.99, loss_clipping=1,
                 inital_learning_rate=0.004, learning_rate_decay=None, target_network_update_steps=10000):
        # setup vars needed for create_network_graph
        # if optimizer is none use default rms prop
        if optimizer is None:
            optimizer = partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)
        elif not callable(optimizer):
            raise AttributeError('Optimizer must be callable. EX: partial(tf.train.RMSPropOptimizer, arg1, arg2)')

        # if network is none use default nips
        if network_generator is None:
            network_generator = create_nips_network
        elif not callable(network_generator):
            raise AttributeError('Network Generator must be callable. EX: create_nips_network(input_tensor, output_num)')

        self._optimizer_fn = optimizer
        self._network_generator = network_generator
        self._q_discount = q_discount
        self._loss_clipping = loss_clipping
        self.target_network_update_steps = target_network_update_steps
        self._target_network_last_updated_step = 0

        # super calls create_network_graph
        super().__init__(input_shape, output_num)

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
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            x_input_tp1 = tf.cast(tf.transpose(x_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            x_actions = tf.placeholder(tf.int32, shape=[None], name='x-actions')
            x_rewards = tf.placeholder(tf.float32, shape=[None], name='x-rewards')
            x_terminals = tf.placeholder(tf.bool, shape=[None], name='x-terminals')
            x_discount = self._q_discount

        # Target network does not reuse variables
        with tf.variable_scope('network') as var_scope:
            network_output = self._network_generator(x_input, output_num)

            # if double DQN then we need to create network output for s_tp1
            if self.algorithm_type == 'double':
                var_scope.reuse_variables()
                network_output_tp1 = self._network_generator(x_input_tp1, output_num)

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
            target_network_output = self._network_generator(x_input_tp1, output_num)

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
                if self.algorithm_type == 'double':
                    # Target = target_Q(s_tp1, argmax(online_Q(s_tp1)))
                    argmax_tp1 = tf.argmax(network_output_tp1, dimension=1)
                    target = tf_util.one_hot(target_network_output, argmax_tp1, output_num)
                elif self.algorithm_type == 'dqn':
                    # Target = max(target_Q(s_tp1))
                    target = tf.reduce_max(target_network_output, reduction_indices=1)
                elif self.algorithm_type == 'nstep':




                # compute a mask that returns gamma (discount factor) or 0 if terminal
                terminal_discount_mask = tf.mul(1.0 - tf.cast(x_terminals, tf.float32), x_discount)
                est_rew_tp1 = tf.mul(terminal_discount_mask, target)

            y = x_rewards + tf.stop_gradient(est_rew_tp1)

            with tf.name_scope('estimated-reward'):
                est_rew = tf_util.one_hot(network_output, x_actions, output_num)

            with tf.name_scope('qloss'):
                # clip loss but keep linear past clip bounds (huber loss with customizable linear part)
                # REFS: https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/q_network.py#L108
                # https://github.com/Jabberwockyll/deep_rl_ale/blob/master/q_network.py#L241
                diff = y - est_rew

                if self._loss_clipping > 0.0:
                    abs_diff = tf.abs(diff)
                    # same as min(diff, loss_clipping) because diff can never be negative (definition of abs value)
                    quadratic_part = tf.clip_by_value(abs_diff, 0.0, self._loss_clipping)
                    linear_part = abs_diff - quadratic_part
                    loss = (0.5 * tf.square(quadratic_part)) + (self._loss_clipping * linear_part)
                else:
                    # But why multiply the loss by 0.5 when not clipping? https://groups.google.com/forum/#!topic/deep-q-learning/hKK0ZM_OWd4
                    loss = 0.5 * tf.square(diff)
                # NOTICE: we are summing gradients
                error = tf.reduce_sum(loss)
            summarizer.summarize(error, 'scalar', 'loss')

        # optimizer
        with tf.name_scope('shared-optimizer'):
            tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = self._optimizer_fn(learning_rate=tf_learning_rate)
            # only train the network vars not the target network
            tf_train_step = optimizer.minimize(error, var_list=network_trainables)

            # tf learn auto merges all summaries so we just have to grab the last output
            tf_summaries = summarizer.summarize(tf_learning_rate, 'scalar', 'learning-rate')

        # function to get network output
        def get_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state}
            return sess.run([network_output], feed_dict=feed_dict)

        # function to get mse feed dict
        def train_step(sess, state, action, reward, state_tp1, terminal, global_step=0, summaries=False):
            self.possible_update_target_network(global_step)
            self.anneal_learning_rate(global_step)

            feed_dict = {x_input_channel_firstdim: state, x_input_tp1_channel_firstdim: state_tp1,
                         x_actions: action, x_rewards: reward, x_terminals: terminal,
                         tf_learning_rate: self.current_learning_rate}
            if summaries:
                return sess.run([tf_summaries, tf_train_step], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step], feed_dict=feed_dict)

        def update_target_net(sess):
            return sess.run([update_target_network_ops])

        self._get_output = get_output
        self._train_step = train_step
        self._update_target_network = update_target_net
        self._save_variables = network_trainables

    def possible_update_target_network(self, global_step):
        if global_step > self._target_network_last_updated_step + self.target_network_update_steps:
            logger = logging.getLogger(__name__)
            logger.info('Updating target network')
            self._update_target_network(self.tf_session)

    def anneal_learning_rate(self, global_step):
        self.learning_rate_annealer.anneal_to(global_step)

    @property
    def current_learning_rate(self):
        return self.learning_rate_annealer.current_value


def create_nips_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    out = tflearn.fully_connected(l_hid3, output_num, scope='denseout')

    return out
