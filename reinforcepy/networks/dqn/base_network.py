import tensorflow as tf


class BaseNetwork:
    """
    Base network class, provides some tensorflow boiler plate by managing graph and session.
    Children must implement create_network_graph and set _get_output, _train_step and optionally _save_variables
    Recurrent networks should override reset to reset their internal state
    """
    def __init__(self, input_shape, output_num):
        self._input_shape = input_shape
        self._output_num = output_num
        self.tf_session = None
        self.tf_graph = None
        self.saver = None

        # these functions are created by create_network_graph
        self._get_output = None
        self._train_step = None
        self._save_variables = None

        with tf.Graph().as_default() as graph:
            self.tf_graph = graph
            self.create_network_graph()
            self.saver = self._create_tf_saver()
            self.init_tf_session()

    def init_tf_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=config)
        self.tf_session.run(tf.global_variables_initializer())

    def create_network_graph(self):
        raise NotImplementedError('Children of BaseNetwork must implement create_network_graph')

    def _create_tf_saver(self):
        return tf.train.Saver(var_list=self._save_variables)

    def get_output(self, x):
        return self._get_output(self.tf_session, x)

    def train_step(self, state, action, reward, state_tp1, terminal, global_step=None, summaries=False):
        return self._train_step(self.tf_session, state, action, reward, state_tp1, terminal, global_step=global_step, summaries=summaries)

    def save(self, *args, **kwargs):
        self.saver.save(self.tf_session, *args, **kwargs)

    def load(self, path):
        self.saver.restore(self.tf_session, path)

    def reset(self):
        pass
