import tensorflow as tf
from .dqn_inits import dqn_nips_network


class DQN_NIPS:
    def __init__(self, network_parms, training_parms, log_dir=None, log_steps=100, log_metadata=False):
        with tf.Graph().as_default() as graph:
            train, get_output = dqn_nips_network(network_parms, training_parms)

            # create tf session and init vars
            self.tf_session = tf.Session(graph=graph)
            self.tf_session.run(tf.global_variables_initializer())
            self.tf_saver = tf.train.Saver()

            # if we are logging
            if log_dir is not None:
                self.logging = True
                self.summary_writer = tf.summary.FileWriter(log_dir, graph=graph)
                self.log_metadata = log_metadata
                self.log_steps = log_steps
            else:
                self.logging = False

        self._train = train
        self._get_output = get_output

        self.learning_rate = training_parms.get('learning_rate')
        self.train_steps = 0

    def train(self, states, actions, rewards, state_tp1s, terminal):
        if self.logging and self.train_steps % self.log_steps == 0:
            if self.log_metadata:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                loss, summaries, _ = self._train(self.tf_session, self.learning_rate, states, actions, rewards,
                                                 state_tp1s, terminal,
                                                 run_summaries=True,
                                                 options=run_options,
                                                 run_metadata=run_metadata)
                self.summary_writer.add_run_metadata(run_metadata,
                                                     'step{0:.2f}'.format(self.train_steps),
                                                     global_step=self.train_steps)
            else:
                loss, summaries, _ = self._train(self.tf_session, self.learning_rate,
                                                 states, actions, rewards,
                                                 state_tp1s, terminal,
                                                 run_summaries=True)

            self.summary_writer.add_summary(summaries, global_step=self.train_steps)
        else:
            loss, _ = self._train(self.tf_session, self.learning_rate, states, actions, rewards, state_tp1s, terminal)

        self.train_steps += 1
        return loss

    def get_output(self, state):
        return self._get_output(self.tf_session, state)[0]

    def load(self, filename):
        self.tf_saver.restore(self.tf_session, filename)

    def save(self, filename):
        # Save the variables to disk.
        self.tf_saver.save(self.tf_session, filename, global_step=self.train_steps)
