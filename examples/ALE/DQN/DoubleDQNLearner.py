from learningALE.learners.DQN import DQNLearner
import numpy as np


class DoubleDQNLearner(DQNLearner):
    def __init__(self, skip_frame, num_actions, load=None):
        super().__init__(skip_frame, num_actions, load)
        self.target_cnn = self.cnn.copy()

    def copy_new_target(self):
        self.target_cnn = self.cnn.copy()

    def get_est_reward(self, state_tp1s, terminal):
        action_selection = np.argmax(self.cnn.get_output(state_tp1s), axis=1)
        value_estimation = self.target_cnn.get_output(state_tp1s)
        r_double_dqn = value_estimation[np.arange(0, 32), action_selection]
        return (1-terminal) * self.discount * r_double_dqn
