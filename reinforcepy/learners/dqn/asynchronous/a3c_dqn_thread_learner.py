import numpy as np
from .nstep_dqn_thread_learner import NStepDQNThreadLearner


class A3CThreadLearner(NStepDQNThreadLearner):
    def get_action(self, state):
        """
        Get action according to policy probabilities
        REF: https://github.com/coreylynch/async-rl/blob/master/a3c.py#L52
        https://github.com/muupan/async-rl/blob/master/policy_output.py#L26
        """
        cnn_action_probabilities = self.network.get_output(self.frame_buffer.get_buffer_with(state))

        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        cnn_action_probabilities = cnn_action_probabilities - np.finfo(np.float32).epsneg
        # Useful numpy function ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
        sample = np.random.multinomial(1, cnn_action_probabilities)
        # since we only sample once, sample will look like a one hot array
        action_index = int(np.nonzero(sample)[0])  # numpy where returns an array of length 1, we just want the first
        return action_index
