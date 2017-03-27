import numpy as np
from reinforcepy.learners.dqn.asynchronous.q_thread_learner import QThreadLearner


class BootstrapQThreadLearner(QThreadLearner):
    def __init__(self, num_bootstraps, *args, **kwargs):
        self.steps_since_train = 0
        self.num_bootstraps = num_bootstraps
        self.current_bootstrap = np.random.randint(0, self.num_bootstraps)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.current_bootstrap = np.random.randint(0, self.num_bootstraps)
        self.steps_since_train = 0

    def get_action(self, state):
        """
        Gets an action for the current state. First queries action_handler to see
        if we should execute a random action. If random action, then don't send to gpu
        """
        # check if doing random action
        random, action = self.action_handler.get_random()
        if not random:
            return self.network.get_output(self.frame_buffer.get_buffer_with(state), self.current_bootstrap)
        return action

    def print_episode_end_status(self, reward):
        curr_rand_val = ''
        if self.random_policy:
            curr_rand_val = 'Rand Val: {0}'.format(self.action_handler.curr_rand_val)
        print(self, 'Using Bootstrap: {}'.format(self.current_bootstrap), 'Episode reward:', reward,
              'Steps:', self.environment.curr_step_count,
              'Step count:', self.step_count, curr_rand_val)
