import numpy as np
from .BaseAsyncTargetLearner import BaseAsyncTargetLearner
from reinforcepy.learners import BaseQLearner


class Async1StepDQNLearner(BaseAsyncTargetLearner, BaseQLearner):
    def reset(self):
        self.total_reward = 0
        self.loss_list = list()

    def get_action(self, state):
        # convert game state to float between 0 - 1 and add to buffer
        self.frame_buffer.add_state_to_buffer(np.asarray(state / 255.0, dtype=np.float32))
        return self.get_game_action(self.frame_buffer.get_buffer())

    def step(self, environment_step_fn, action):
        reward = 0
        for skipped_frame in range(self.skip_frame):
            env_reward = environment_step_fn(action)
            reward += np.clip(env_reward, 0, 1)
        return reward

    def update(self, state, action, reward, state_tp1, terminal):
        # we don't need state because it's stored in get action

        # accumulate gradients
        loss = self.cnn.accumulate_gradients(self.frame_buffer.get_buffer(),
                                             self.action_handler.game_action_to_action_ind(action),
                                             reward, self.frame_buffer.get_buffer_with(state_tp1), terminal)
        self.loss_list.append(float(loss))
        self.thread_steps += 1
        self.total_reward += reward

        if self.thread_steps % self.async_update_step == 0 or terminal:
            global_vars = self.async_update()

            self.action_handler.anneal_to(global_vars['counter'])

    def episode_end(self):
        self.async_send_stats()
        print(self, 'ending episode. Step counter:', self.thread_steps,
              'Score:', self.total_reward, 'Current Rand Val:', self.action_handler.curr_rand_val)