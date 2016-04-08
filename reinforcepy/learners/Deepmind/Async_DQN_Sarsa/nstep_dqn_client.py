import numpy as np
from .BaseAsyncTargetLearner import BaseAsyncTargetLearner
from reinforcepy.learners import BaseQLearner


class AsyncNStepDQNLearner(BaseAsyncTargetLearner, BaseQLearner):
    def __init__(self, learner_parms, network_partial, pipe):
        super().__init__(learner_parms, network_partial, pipe)

        learner_parms.required(['discount'])
        self.discount = learner_parms.get('discount')

    def reset(self):
        self.total_reward = 0
        self.loss_list = list()

        # nstep vars, rewards and states are stored in a list of length self.async_update_step
        self.t_start = self.thread_steps
        self.train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
        self.train_actions = list()
        self.rewards = list()

    def get_action(self, state):
        # convert state to 0-1
        state = np.asarray(state / 255.0, dtype=np.float32)
        # add to buffer
        self.frame_buffer.add_state_to_buffer(state)

        # add current buffer to train states
        current_buffer = self.frame_buffer.get_buffer()
        self.train_states[self.thread_steps - self.t_start] = current_buffer

        # return game action with current buffer
        return self.get_game_action(current_buffer)

    def step(self, environment_step_fn, action):
        reward = 0
        for skipped_frame in range(self.skip_frame):
            env_reward = environment_step_fn(action)
            reward += np.clip(env_reward, 0, 1)
        return reward

    def update(self, state, action, reward, state_tp1, terminal):
        # we don't need state because it's stored in get action

        # update nstep vars
        self.rewards.append(reward)
        self.train_actions.append(self.action_handler.game_action_to_action_ind(action))

        # update loop vars
        self.thread_steps += 1
        self.total_reward += reward

        # if update or terminal do a gradient update
        if (self.thread_steps - self.t_start) % self.async_update_step == 0 or terminal:
            # calculate training rewards
            train_rewards = np.zeros(self.thread_steps - self.t_start, dtype=np.float32)

            # estimated reward is 0 if terminal else bootstrap off state_tp1
            train_reward = 0 if terminal else np.max(
                self.cnn.get_target_output(self.frame_buffer.get_buffer_with(state_tp1)))

            # calculate td reward
            for i in range((self.thread_steps - self.t_start) - 1, -1, -1):  # for a reversed range we have to subtract 1
                train_reward = self.rewards[i] + self.discount * train_reward
                train_rewards[i] = train_reward

            # if terminal cuts off the async step then just take the states that correspond to how many steps we did
            if terminal:
                self.train_states = self.train_states[0:self.thread_steps - self.t_start]

            # calculate gradients
            loss = self.cnn.accumulate_gradients(self.train_states, self.train_actions, train_rewards)

            self.loss_list.append(float(loss))

            # async update step
            global_vars = self.async_update()

            self.action_handler.anneal_to(global_vars['counter'])

            # re-initialize nstep vars
            self.t_start = self.thread_steps
            self.train_states = np.zeros((self.async_update_step, self.phi_length, 84, 84), dtype=np.float32)
            self.train_actions = list()
            self.rewards = list()

    def episode_end(self):
        self.async_send_stats()
        print(self, 'ending episode. Step counter:', self.thread_steps,
              'Score:', self.total_reward, 'Current Rand Val:', self.action_handler.curr_rand_val)
