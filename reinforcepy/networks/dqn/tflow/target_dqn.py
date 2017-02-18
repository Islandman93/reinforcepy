import logging
from functools import partial
import numpy as np
import tensorflow as tf
import tflearn.helpers.summarizer as summarizer
import reinforcepy.networks.util.tflow_util as tf_util
from reinforcepy.handlers.linear_annealer import LinnearAnnealer
from ..base_network import BaseNetwork


class TargetDQN(BaseNetwork):
    """
        Parameters:
            algorithm_type: str one of 'dqn', 'double', 'nstep', 'doublenstep'
    """
    def __init__(self, input_shape, output_num, algorithm_type, optimizer=None, network_generator=tf_util.create_nips_network, q_discount=0.99, loss_clipping=1,
                 global_norm_clipping=40, initial_learning_rate=0.001, learning_rate_decay=None, target_network_update_steps=10000):
        # setup vars needed for create_network_graph
        # if optimizer is none use default rms prop
        if optimizer is None:
            optimizer = partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)
        elif not callable(optimizer):
            raise AttributeError('Optimizer must be callable. EX: partial(tf.train.RMSPropOptimizer, arg1, arg2)')

        if not callable(network_generator):
            raise AttributeError('Network Generator must be callable. EX: create_network(input_tensor, output_num)')

        self.learning_rate_annealer = LinnearAnnealer(initial_learning_rate, 0, learning_rate_decay)
        self.algorithm_type = algorithm_type
        self.global_norm_clipping = global_norm_clipping
        self._optimizer_fn = optimizer
        self._network_generator = network_generator
        self._q_discount = q_discount
        self._loss_clipping = loss_clipping
        self.target_network_update_steps = target_network_update_steps
        self._target_network_next_update_step = 0

        # super calls create_network_graph
        super().__init__(input_shape, output_num)

    def create_network_graph(self):
        input_shape = self._input_shape
        output_num = self._output_num
        # Input placeholders
        with tf.name_scope('input'):
            # we need to fix the input shape from (batch, filter, height, width) to
            # tensorflow which is (batch, height, width, filter)
            self._t_x_input_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            self._t_x_input = tf.cast(tf.transpose(self._t_x_input_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            self._t_x_input_tp1_channel_firstdim = tf.placeholder(tf.uint8, [None] + input_shape, name='x-input-tp1')
            # transpose because tf wants channels on last dim and channels are passed in on 2nd dim
            self._t_x_input_tp1 = tf.cast(tf.transpose(self._t_x_input_tp1_channel_firstdim, perm=[0, 2, 3, 1]), tf.float32) / 255.0
            self._t_x_actions = tf.placeholder(tf.int32, shape=[None], name='x-actions')
            self._t_x_rewards = tf.placeholder(tf.float32, shape=[None], name='x-rewards')
            self._t_x_terminals = tf.placeholder(tf.bool, shape=[None], name='x-terminals')
            self._t_x_discount = self._q_discount

        # Target network does not reuse variables
        with tf.variable_scope('network') as var_scope:
            self._t_network_output = self._network_generator(self._t_x_input, output_num)

            # get the trainable variables for this network, later used to overwrite target network vars
            self._tf_network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

            # summarize activations
            summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='network'))

            # if double DQN then we need to create network output for s_tp1
            if self.algorithm_type == 'double' or self.algorithm_type == 'doublenstep':
                var_scope.reuse_variables()
                self._t_network_output_tp1 = self._network_generator(self._t_x_input_tp1, output_num)

            # summarize a histogram of each action output
            for output_ind in range(output_num):
                summarizer.summarize(self._t_network_output[:, output_ind], 'histogram', 'network-output/{0}'.format(output_ind))

            # add network summaries
            summarizer.summarize_variables(train_vars=self._tf_network_trainables)

        with tf.variable_scope('target-network'):
            self._t_target_network_output = self._network_generator(self._t_x_input_tp1, output_num)

            # get trainables for target network, used in assign op for the update target network step
            target_network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target-network')

        # update target network with network variables
        with tf.name_scope('update-target-network'):
            self._tf_update_target_network_ops = [target_v.assign(v) for v, target_v in zip(self._tf_network_trainables, target_network_trainables)]

        # if double convience function to get target values for online action
        if self.algorithm_type == 'double' or self.algorithm_type == 'doublenstep':
            with tf.name_scope('double_target'):
                # Target = target_Q(s_tp1, argmax(online_Q(s_tp1)))
                argmax_tp1 = tf.argmax(self._t_network_output_tp1, axis=1)
                self._t_target_value_online_action = tf_util.one_hot(self._t_target_network_output, argmax_tp1, output_num)

        # caclulate QLoss
        with tf.name_scope('loss'):
            # nstep rewards are calculated outside the gpu/graph because it requires a loop
            if self.algorithm_type != 'nstep' and self.algorithm_type != 'doublenstep':
                with tf.name_scope('estimated-reward-tp1'):
                    if self.algorithm_type == 'double':
                        # Target = target_Q(s_tp1, argmax(online_Q(s_tp1)))
                        target = self._t_target_value_online_action
                    elif self.algorithm_type == 'dqn':
                        # Target = max(target_Q(s_tp1))
                        target = tf.reduce_max(self._t_target_network_output, axis=1)

                    # compute a mask that returns gamma (discount factor) or 0 if terminal
                    terminal_discount_mask = tf.multiply(1.0 - tf.cast(self._t_x_terminals, tf.float32), self._t_x_discount)
                    est_rew_tp1 = tf.multiply(terminal_discount_mask, target)

                y = self._t_x_rewards + tf.stop_gradient(est_rew_tp1)
            # else nstep
            else:
                y = self._t_x_rewards

            with tf.name_scope('estimated-reward'):
                est_rew = tf_util.one_hot(self._t_network_output, self._t_x_actions, output_num)

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
            self._tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = self._optimizer_fn(learning_rate=self._tf_learning_rate)
            # only train the network vars not the target network
            gradients = optimizer.compute_gradients(error, var_list=self._tf_network_trainables)
            # gradients are stored as a tuple, (gradient, tensor the gradient corresponds to)
            # kinda lame that clip by global norm doesn't accept the list of tuples returned from compute_gradients
            # so we unzip then zip
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)  # returns list[tensors], norm
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            self._tf_train_step = optimizer.apply_gradients(clipped_grads_tensors)
            # tflearn smartly knows how gradients are stored so we just pass in the list of tuples
            summarizer.summarize_gradients(clipped_grads_tensors)

            # tf learn auto merges all summaries so we just have to grab the last output
            self._tf_summaries = summarizer.summarize(self._tf_learning_rate, 'scalar', 'learning-rate')

    def _train_step(self, sess, states, actions, rewards, states_tp1, terminals, global_step=0, summaries=False):
        self.possible_update_target_network(global_step)
        self.anneal_learning_rate(global_step)

        # if not nstep we pass all vars to gpu
        if self.algorithm_type != 'nstep' and self.algorithm_type != 'doublenstep':
            feed_dict = {self._t_x_input_channel_firstdim: states, self._t_x_input_tp1_channel_firstdim: states_tp1,
                         self._t_x_actions: actions, self._t_x_rewards: self._t_rewards, self._t_x_terminals: terminals,
                         self._tf_learning_rate: self.current_learning_rate}
        # else nstep calculate TD reward
        else:
            if sum(terminals) > 1:
                raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

            # last state not terminal need to query target network
            curr_reward = 0
            if not terminals[-1]:
                # make a list to add back the first dim (needs to be 4 dims)
                target_feed_dict = {self._t_x_input_tp1_channel_firstdim: [states_tp1[-1]]}
                if self.algorithm_type == 'nstep':
                    curr_reward = max(sess.run(self._t_target_network_output, feed_dict=target_feed_dict)[0])
                elif self.algorithm_type == 'doublenstep':
                    curr_reward = sess.run(self._t_target_value_online_action, feed_dict=target_feed_dict)[0]

            # get bootstrap estimate of last state_tp1
            td_rewards = []
            for reward in reversed(rewards):
                curr_reward = reward + self._q_discount * curr_reward
                td_rewards.append(curr_reward)
            # td rewards is computed backward but other lists are stored forward so need to reverse
            td_rewards = list(reversed(td_rewards))
            feed_dict = {self._t_x_input_channel_firstdim: states, self._t_x_actions: actions, self._t_x_rewards: td_rewards,
                         self._tf_learning_rate: self.current_learning_rate}

        if summaries:
            return sess.run([self._tf_summaries, self._tf_train_step], feed_dict=feed_dict)[0]
        else:
            return sess.run([self._tf_train_step], feed_dict=feed_dict)

    def update_target_net(self, sess):
        return sess.run([self._tf_update_target_network_ops])

    # function to get network output
    def _get_output(self, sess, state):
        feed_dict = {self._t_x_input_channel_firstdim: state}
        return np.argmax(sess.run([self._t_network_output], feed_dict=feed_dict))

    def possible_update_target_network(self, global_step):
        if global_step > self._target_network_next_update_step:
            self._target_network_next_update_step += self.target_network_update_steps
            logger = logging.getLogger(__name__)
            logger.info('Updating target network')
            self._update_target_network(self.tf_session)

    def anneal_learning_rate(self, global_step):
        self.learning_rate_annealer.anneal_to(global_step)

    @property
    def current_learning_rate(self):
        return self.learning_rate_annealer.curr_val
