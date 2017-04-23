import tensorflow as tf


class BaseNetwork:
    """
    Base network class, provides some tensorflow boiler plate by managing graph and session.
    Children must implement create_network_graph and set _get_output, _train_step and optionally _save_variables
    """
    def __init__(self, input_shape, output_num, log_dir='/tmp/tensorboard/', save_interval=float('inf'),
                 summary_interval=float('inf'), session=None, worker_id=0, summaries=True, cpu_only=False,
                 finalize_graph=True):
        self._input_shape = input_shape
        self._output_num = output_num
        self.tf_session = session
        self.tf_graph = None
        self.saver = None
        self.log_dir = log_dir if log_dir[-1] == '/' else log_dir + '/'
        self.save_interval = save_interval
        self.last_save_step = 0
        self.summary_interval = summary_interval
        self.last_summary_step = 0
        self.summaries = summaries
        self.worker_id = worker_id

        # these functions are created by create_network_graph
        self._save_variables = None

        if cpu_only:
            with tf.device('cpu'):
                self.create_network_graph()
        else:
            self.create_network_graph()
        self.saver = self._create_tf_saver()
        if self.tf_session is None:
            self.init_tf_session()

        self.tf_graph = self.tf_session.graph
        if self.summaries:
            # create a summary writer
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.tf_graph)

            with self.tf_graph.as_default():
                # summaries for end of episode
                self._tf_reward = tf.placeholder(tf.int32)
                self._tf_reward_summary = tf.summary.scalar('reward', self._tf_reward)

        variable_initializer = tf.global_variables_initializer()
        if finalize_graph:
            self.tf_graph.finalize()

        self.tf_session.run(variable_initializer)

    def init_tf_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=config)

    def create_network_graph(self):
        raise NotImplementedError('Children of BaseNetwork must implement create_network_graph')

    def _create_tf_saver(self):
        return tf.train.Saver(var_list=self._save_variables, max_to_keep=None)

    def get_output(self, x):
        return self._get_output(self.tf_session, x)

    def train_step(self, *args, global_step, force_summaries=False, **kwargs):
        if self.should_save(global_step):
            self.save(global_step)
        write_summaries = self.should_write_summaries(global_step) or force_summaries
        train_data = self._train_step(self.tf_session, *args, global_step=global_step, summaries=write_summaries, **kwargs)
        summaries = train_data[-1]
        if write_summaries:
            self.write_summary(summaries, global_step)
        return train_data

    def write_episode_reward_summary(self, reward, global_step):
        summary = self.tf_session.run(self._tf_reward_summary, feed_dict={self._tf_reward: reward})
        self.write_summary(summary, global_step)

    def write_summary(self, summary, global_step):
        self.last_summary_step = global_step
        self.summary_writer.add_summary(summary, global_step=global_step)

    def should_write_summaries(self, global_step):
        return (global_step > self.last_summary_step + self.summary_interval) and self.summaries

    def should_save(self, global_step):
        return global_step > self.last_save_step + self.save_interval

    def save(self, global_step, model_name='model'):
        self.last_save_step = global_step
        self.saver.save(self.tf_session, self.log_dir + model_name, global_step=global_step)

    def load(self, path):
        self.saver.restore(self.tf_session, path)
