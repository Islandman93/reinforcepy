from .BaseDQNViewer import BaseDQNViewer
from reinforcepy.learners.base_q_learner import BaseQLearner
import numpy as np


class DQNViewer(BaseDQNViewer, BaseQLearner):
    def reset(self):
        pass

    def get_action(self, state):
        # convert state to 0-1
        state = np.asarray(state / 255.0, dtype=np.float32)
        # add to buffer
        self.frame_buffer.add_state_to_buffer(state)

        # return game action with current buffer
        return self.get_game_action(self.frame_buffer.get_buffer())

    def step(self, environment_step_fn, action):
        reward = 0
        for skipped_frame in range(self.skip_frame):
            env_reward = environment_step_fn(action)
            reward += np.clip(env_reward, 0, 1)
        return reward

    def update(self, state, action, reward, state_tp1, terminal):
        self.total_reward += reward
        pass

    def episode_end(self):
        print(self, 'ending episode. Score:', self.total_reward, 'Current Rand Val:', self.action_handler.curr_rand_val)
