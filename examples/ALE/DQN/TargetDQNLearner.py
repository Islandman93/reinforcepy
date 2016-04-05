from learningALE.learners.DQN import DQNLearner
import numpy as np


class TargetDQNLearner(DQNLearner):
    def __init__(self, skip_frame, num_actions, load=None):
        super().__init__(skip_frame, num_actions, load)
        self.target_cnn = self.cnn.copy()

    def copy_new_target(self):
        self.target_cnn = self.cnn.copy()

    def get_est_reward(self, state_tp1s, terminal):
        r_tp1 = self.target_cnn.get_output(state_tp1s)
        return (1-terminal) * self.discount * np.max(r_tp1, axis=1)
