from functools import partial
import tensorflow as tf
import tflearn
import tflearn.helpers.summarizer as summarizer


class NStepA3C:
    def __init__(self, input_shape, output_num, optimizer=None, network_generator=None, q_discount=0.99, entropy_regularization=0.01):
        self.tf_session = None
        self.tf_graph = None
        self.saver = None
        self.q_discount = q_discount

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
            network_generator = create_a3c_network
        elif not callable(network_generator):
            raise AttributeError('Network Generator must be callable. EX: create_a3c_network(input_tensor, output_num)')

        with tf.Graph().as_default() as graph:
            self.tf_graph = graph
            self.create_network_graph(input_shape, output_num, network_generator, q_discount, optimizer, entropy_regularization=entropy_regularization)
            self.init_tf_session()
            self.update_target_network()

    def init_tf_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=config)
        self.tf_session.run(tf.initialize_all_variables())

    def create_network_graph(self, input_shape, output_num, network_generator, q_discount, optimizer, entropy_regularization):
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

        # Target network does not reuse variables. so we use two different variable scopes
        with tf.variable_scope('network'):
            actor_output, critic_output = network_generator(x_input, output_num)
            # flatten the critic_output NOTE: THIS IS VERY IMPORTANT
            # otherwise critic_output will be (batch_size, 1) and all ops with it and x_rewards will create a
            # tensor of shape (batch_size, batch_size)
            critic_output = tf.reshape(critic_output, [-1])

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

        # caclulate losses
        with tf.name_scope('loss'):
            with tf.name_scope('critic-reward-diff'):
                critic_diff = tf.sub(x_rewards, critic_output)

            with tf.name_scope('log-of-actor-policy'):
                # Because of https://github.com/tensorflow/tensorflow/issues/206
                # we cannot use numpy like indexing so we convert to a one hot
                # multiply then take the max over last dim
                # NumPy/Theano est_rew = network_output[:, x_actions]
                x_actions_one_hot = tf.one_hot(x_actions, depth=output_num, name='one-hot',
                                               on_value=1.0, off_value=0.0, dtype=tf.float32)
                # we reduce sum here because the output could be negative we can't take the max
                # the other indecies will be 0
                actor_output_one_hot = tf.mul(-tf.log(actor_output), x_actions_one_hot)
                log_policy = tf.reduce_sum(actor_output_one_hot, reduction_indices=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = -tf.reduce_sum(tf.mul(actor_output, tf.log(actor_output)))

            with tf.name_scope('actor-loss'):
                # NOTICE: we are summing (accumulating) gradients
                actor_loss_notacc = tf.mul(log_policy, critic_diff)
                actor_loss = tf.reduce_sum(actor_loss_notacc)

            with tf.name_scope('critic-loss'):
                critic_loss = tf.square(critic_diff) * 0.5
                # NOTICE: we are summing gradients
                critic_loss = tf.reduce_sum(critic_loss)

            summarizer.summarize(critic_loss, 'scalar', 'critic-loss')
            summarizer.summarize(actor_loss, 'scalar', 'actor-loss')

        # optimizer
        with tf.name_scope('shared-optimizer'):
            tf_learning_rate = tf.placeholder(tf.float32)
            optimizer = optimizer(learning_rate=tf_learning_rate)
            total_loss = actor_loss + (entropy_regularization * actor_entropy) + critic_loss
            # only train the network vars
            gradients = optimizer.compute_gradients(total_loss, var_list=network_trainables)
            clipped_gradients = [(tf.clip_by_norm(gradient, 0.1), tensor) for gradient, tensor in gradients]
            tf_train_step = optimizer.apply_gradients(clipped_gradients)

            # tf learn auto merges all summaries so we just have to grab the last output
            tf_summaries = summarizer.summarize(tf_learning_rate, 'scalar', 'learning-rate')

        # function to get network output
        def get_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state}
            return sess.run([actor_output], feed_dict=feed_dict)

        # function to get network output
        def get_target_output(sess, state):
            feed_dict = {x_input_channel_firstdim: state}
            return sess.run([critic_output], feed_dict=feed_dict)

        # function to get mse feed dict
        def train_step(sess, current_learning_rate, state, action, reward, summaries=False):
            feed_dict = {x_input_channel_firstdim: state, x_actions: action, x_rewards: reward,
                         tf_learning_rate: current_learning_rate}
            # print(sess.run([actor_loss, critic_loss, actor_entropy], feed_dict=feed_dict))
            if summaries:
                return sess.run([tf_summaries, tf_train_step], feed_dict=feed_dict)[0]
            else:
                return sess.run([tf_train_step], feed_dict=feed_dict)

        def update_target_net(sess):
            pass

        self._get_output = get_output
        self._get_target_output = get_target_output
        self._train_step = train_step
        self._update_target_network = update_target_net
        self.saver = tf.train.Saver(var_list=network_trainables)

    def get_output(self, x):
        return self._get_output(self.tf_session, x)

    def get_target_output(self, x):
        return self._get_target_output(self.tf_session, x)

    def train_step(self, current_learning_rate, state, action, reward, summaries=False):
        return self._train_step(self.tf_session, current_learning_rate, state, action, reward, summaries=summaries)

    def update_target_network(self):
        self._update_target_network(self.tf_session)

    def save(self, *args, **kwargs):
        self.saver.save(self.tf_session, *args, **kwargs)

    def load(self, path):
        self.saver.restore(self.tf_session, path)


def create_a3c_network(input_tensor, output_num):
    l_hid1 = tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='relu', scope='conv1')
    l_hid2 = tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='relu', scope='conv2')
    l_hid3 = tflearn.fully_connected(l_hid2, 256, activation='relu', scope='dense3')
    actor_out = tflearn.fully_connected(l_hid3, output_num, activation='softmax', scope='actorout')
    critic_out = tflearn.fully_connected(l_hid3, 1, activation='linear', scope='criticout')

    return actor_out, critic_out
