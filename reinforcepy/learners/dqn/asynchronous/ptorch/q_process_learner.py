import numpy as np
from reinforcepy.learners import BaseQLearner
from .base_process_learner import BaseProcessLearner


class QProcessLearner(BaseProcessLearner, BaseQLearner):
    def update(self, state, action, reward, state_tp1, terminal):
        self.frame_buffer.add_state_to_buffer(state)

        # quit update if testing
        if self.testing:
            return

        # clip reward
        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        # accumulate minibatch_vars
        self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                  reward, self.frame_buffer.get_buffer_with(state_tp1), terminal)

        # increment counters
        self.step_count += 1
        self.global_dict['counter'] += 1

        # check perform gradient step
        if self.step_count % self.async_update_step == 0 or terminal:
            self.network.load_state_dict(self.host_network.state_dict())
            self.network.train_step(*self.get_minibatch_vars(), 0.1, self.host_network.optimizer)
            self.reset_minibatch()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals']]
