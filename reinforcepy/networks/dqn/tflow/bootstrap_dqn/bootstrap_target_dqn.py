import logging
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import tflearn.helpers.summarizer as summarizer
import tflearn
import reinforcepy.networks.util.tflow_util as tf_util
from reinforcepy.handlers.linear_annealer import LinnearAnnealer
from reinforcepy.networks.dqn.base_network import BaseNetwork
from scipy import stats


def create_bootstraps(num_bootstraps, input, output_num):
    bootstrap_outputs = []
    for bootstrap in range(num_bootstraps):
        bootstrap_fc1 = tflearn.fully_connected(input, 256, activation='relu', scope='bootstrap-fc1-{}'.format(bootstrap))
        bootstrap_out = tflearn.fully_connected(bootstrap_fc1, output_num, scope='bootstrap-out-{}'.format(bootstrap))
        bootstrap_outputs.append(bootstrap_out)
    return bootstrap_outputs


def create_nips_network(input_tensor, output_num, conv_only=False):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1', padding='valid')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2', padding='valid')
    if conv_only:
        return l_hid2
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    out = tflearn.fully_connected(l_hid3, output_num, scope='denseout')
    return out


class BootstrapTargetDQN(BaseNetwork):
    """
        Parameters:
            algorithm_type: str one of 'dqn', 'double', 'dueling', 'nstep', 'doublenstep', 'duelingnstep'
                For dueling architectures the network_generator must return Value, Advantage outputs
    """
    def __init__(self, input_shape, output_num, algorithm_type, num_bootstraps, optimizer=None, network_generator=create_nips_network,
                 q_discount=0.99, loss_clipping=1, global_norm_clipping=40.0, initial_learning_rate=0.001,
                 learning_rate_decay=None, target_network_update_steps=10000, **kwargs):
        # setup vars needed for create_network_graph
        # if optimizer is none use default rms prop
        if optimizer is None:
            optimizer = partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)  # momentum 0.95
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
        self.num_bootstraps = num_bootstraps

        # super calls create_network_graph
        super().__init__(input_shape, output_num, **kwargs)

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
            if 'dueling' not in self.algorithm_type:
                self._t_conv_output = self._network_generator(self._t_x_input, output_num, conv_only=True)
                self._bootstrap_outputs = create_bootstraps(self.num_bootstraps, self._t_conv_output, output_num)
            # else:
            #     # dueling redfines Q(s, a) as V(s) + (A(s, a) - mean(A(s)))
            #     value_output, advantage_output = self._network_generator(self._t_x_input, output_num)
            #     self._t_network_output = tf_util.dueling_to_q_vals(value_output, advantage_output)

            # get the trainable variables for this network, later used to overwrite target network vars
            self._tf_network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

            # summarize activations
            summarizer.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='network'))

            # if double or dueling DQN then we need to create network output for s_tp1
            if 'double' in self.algorithm_type or 'dueling' in self.algorithm_type:
                var_scope.reuse_variables()
                if 'double' in self.algorithm_type:
                    self._t_network_output_tp1 = self._network_generator(self._t_x_input_tp1, output_num, conv_only=True)
                    self._bootstrap_outputs_tp1 = create_bootstraps(self.num_bootstraps, self._t_network_output_tp1, output_num)
                # elif 'dueling' in self.algorithm_type:
                #     # dueling redfines Q(s, a) as V(s) + (A(s, a) - mean(A(s)))
                #     value_output_tp1, advantage_output_tp1 = self._network_generator(self._t_x_input_tp1, output_num)
                #     self._t_network_output_tp1 = tf_util.dueling_to_q_vals(value_output_tp1, advantage_output_tp1)

            # summarize a histogram of each action output
            for output_ind in range(output_num):
                for bootstrap in range(self.num_bootstraps):
                    summarizer.summarize(self._bootstrap_outputs[bootstrap][:, output_ind],
                                         'histogram', 'bootstrap-output-{}/{}'.format(bootstrap, output_ind))

            # add network summaries
            summarizer.summarize_variables(train_vars=self._tf_network_trainables)

        with tf.variable_scope('target-network'):
            if 'dueling' not in self.algorithm_type:
                self._t_target_network_output = self._network_generator(self._t_x_input_tp1, output_num, conv_only=True)
                self._target_bootstrap_outputs = create_bootstraps(self.num_bootstraps, self._t_target_network_output, output_num)
            # else:
            #     target_value_output, target_advantage_output = self._network_generator(self._t_x_input_tp1, output_num)
            #     self._t_target_network_output = tf_util.dueling_to_q_vals(target_value_output, target_advantage_output)

            # get trainables for target network, used in assign op for the update target network step
            target_network_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target-network')

        # update target network with network variables
        with tf.name_scope('update-target-network'):
            self._tf_update_target_network_ops = [target_v.assign(v) for v, target_v in zip(self._tf_network_trainables, target_network_trainables)]

        # if double convience function to get target values for online action
        if 'double' in self.algorithm_type or 'dueling' in self.algorithm_type:
            with tf.name_scope('network_estimated_action_tp1'):
                self._target_bootstrap_value_online_actions = []
                for bootstrap in range(self.num_bootstraps):
                    argmax_tp1 = tf.argmax(self._target_bootstrap_outputs[bootstrap], axis=1)
                    # Target = target_Q(s_tp1, argmax(online_Q(s_tp1)))
                    target_value_online_action = tf_util.one_hot(self._target_bootstrap_outputs[bootstrap], argmax_tp1, output_num)
                    self._target_bootstrap_value_online_actions.append(target_value_online_action)

        # caclulate QLoss
        with tf.name_scope('loss'):
            bootstrap_errors = []
            for bootstrap in range(self.num_bootstraps):
                # nstep rewards are calculated outside the gpu/graph because it requires a loop
                if 'nstep' not in self.algorithm_type:
                    with tf.name_scope('estimated-reward-tp1'):
                        if self.algorithm_type == 'double' or self.algorithm_type == 'dueling':
                            # Target = target_Q(s_tp1, argmax(online_Q(s_tp1)))
                            target = self._target_bootstrap_value_online_actions[bootstrap]
                        elif self.algorithm_type == 'dqn':
                            # Target = max(target_Q(s_tp1))
                            target = tf.reduce_max(self._target_bootstrap_outputs[bootstrap], axis=1)

                        # compute a mask that returns gamma (discount factor) or 0 if terminal
                        terminal_discount_mask = tf.multiply(1.0 - tf.cast(self._t_x_terminals, tf.float32), self._t_x_discount)
                        est_rew_tp1 = tf.multiply(terminal_discount_mask, target)

                    y = self._t_x_rewards + tf.stop_gradient(est_rew_tp1)
                # else nstep
                # else:
                #     y = self._t_x_rewards

                with tf.name_scope('estimated-reward'):
                    est_rew = tf_util.one_hot(self._bootstrap_outputs[bootstrap], self._t_x_actions, output_num)

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
                summarizer.summarize(error, 'scalar', 'bootstrap-loss={}'.format(bootstrap))
                bootstrap_errors.append(error)

        # TODO: this assumes mask P = 1
        with tf.name_scope('total-loss'):
            error = tf.reduce_sum(bootstrap_errors)

        # optimizer
        with tf.name_scope('shared-optimizer'):
            self._tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = self._optimizer_fn(learning_rate=self._tf_learning_rate)
            # only train the network vars not the target network
            gradients = optimizer.compute_gradients(error, var_list=self._tf_network_trainables)
            # BOOTSTRAP DQN: rescale conv grads by 1 / num _bootstraps
            rescaled_gradients = []
            for gradient, tensor in gradients:
                if 'conv' in gradient.name.lower():
                    print('Rescaling {}'.format(gradient.name))
                    with ops.colocate_with(gradient):
                        gradient = gradient * 1 / self.num_bootstraps
                rescaled_gradients.append((gradient, tensor))

            # gradients are stored as a tuple, (gradient, tensor the gradient corresponds to)
            # kinda lame that clip by global norm doesn't accept the list of tuples returned from compute_gradients
            # so we unzip then zip
            tensors = [tensor for gradient, tensor in rescaled_gradients]
            grads = [gradient for gradient, tensor in rescaled_gradients]
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
        if 'nstep' not in self.algorithm_type:
            feed_dict = {self._t_x_input_channel_firstdim: states, self._t_x_input_tp1_channel_firstdim: states_tp1,
                         self._t_x_actions: actions, self._t_x_rewards: rewards, self._t_x_terminals: terminals,
                         self._tf_learning_rate: self.current_learning_rate}
        # else nstep calculate TD reward
        # else:
        #     if sum(terminals) > 1:
        #         raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

        #     # last state not terminal need to query target network
        #     curr_reward = 0
        #     if not terminals[-1]:
        #         # make a list to add back the first dim (needs to be 4 dims)
        #         target_feed_dict = {self._t_x_input_tp1_channel_firstdim: [states_tp1[-1]]}
        #         if self.algorithm_type == 'nstep':
        #             curr_reward = max(sess.run(self._t_target_network_output, feed_dict=target_feed_dict)[0])
        #         elif self.algorithm_type == 'doublenstep' or self.algorithm_type == 'duelingnstep':
        #             curr_reward = sess.run(self._t_target_value_online_action, feed_dict=target_feed_dict)[0]

        #     # get bootstrap estimate of last state_tp1
        #     td_rewards = []
        #     for reward in reversed(rewards):
        #         curr_reward = reward + self._q_discount * curr_reward
        #         td_rewards.append(curr_reward)
        #     # td rewards is computed backward but other lists are stored forward so need to reverse
        #     td_rewards = list(reversed(td_rewards))
        #     feed_dict = {self._t_x_input_channel_firstdim: states, self._t_x_actions: actions, self._t_x_rewards: td_rewards,
        #                  self._tf_learning_rate: self.current_learning_rate}

        if summaries:
            return sess.run([self._tf_train_step, self._tf_summaries], feed_dict=feed_dict)
        else:
            return sess.run([self._tf_train_step], feed_dict=feed_dict)

    def update_target_net(self, sess):
        return sess.run([self._tf_update_target_network_ops])

    def get_output(self, state, bootstrap_number=None):
        # TODO: if bootstrap_number is None then vote
        if bootstrap_number is not None:
            feed_dict = {self._t_x_input_channel_firstdim: state}
            return np.argmax(self.tf_session.run(self._bootstrap_outputs[bootstrap_number], feed_dict=feed_dict))
        else:
            feed_dict = {self._t_x_input_channel_firstdim: state}
            bootstrap_outputs = self.tf_session.run(self._bootstrap_outputs, feed_dict=feed_dict)
            bootstrap_actions = np.argmax(bootstrap_outputs, axis=0)
            # returns a tuple of arrays, return the first element of the first array
            return stats.mode(bootstrap_actions)[0][0]

    def possible_update_target_network(self, global_step):
        if global_step > self._target_network_next_update_step:
            self._target_network_next_update_step += self.target_network_update_steps
            logger = logging.getLogger(__name__)
            logger.info('Updating target network')
            self.update_target_net(self.tf_session)

    def anneal_learning_rate(self, global_step):
        self.learning_rate_annealer.anneal_to(global_step)

    @property
    def current_learning_rate(self):
        return self.learning_rate_annealer.curr_val
