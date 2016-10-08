import numpy as np
from reinforcepy.learners import BaseQLearner
from .onestep_base_thread_learner import OneStepBaseThreadLearner


class NStepDQNThreadLearner(OneStepBaseThreadLearner, BaseQLearner):
    def update(self, state, action, reward, state_tp1, terminal):
        self.frame_buffer.add_state_to_buffer(state)

        # quit update if testing
        if self.testing:
            return

        # clip reward
        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        # accumulate gradients
        self.minibatch_accumulate(self.frame_buffer.get_buffer(), action, reward)

        # increment counters
        self.step_count += 1
        self.global_dict['counter'] += 1

        # check update target
        if self.check_update_target(self.global_dict["counter"]):
            print(self, 'setting target')
            self.network.update_target_network()

        # check perform gradient step
        if self.step_count % self.async_update_step == 0 or terminal:
            td_rewards = self.calculate_td_reward(self.frame_buffer.get_buffer(), terminal)
            summaries = self.global_dict['write_summaries_this_step']
            if summaries:
                self.global_dict['write_summaries_this_step'] = False
                summary = self.network.train_step(self.global_dict['learning_rate'], *self.get_minibatch_vars(), reward=td_rewards, summaries=True)
                self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
            else:
                self.network.train_step(self.global_dict['learning_rate'], *self.get_minibatch_vars(), reward=td_rewards, summaries=False)
            self.reset_minibatch()

        # anneal action handler
        anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
        self.action_handler.anneal_to(anneal_step)

    def calculate_td_reward(self, state, terminal):
        curr_reward = 0 if terminal else np.max(self.network.get_target_output(state))
        td_rewards = [curr_reward]
        # NOTICE: we already created the reward for the last state so we take the rewards only for the states before this one
        for reward in reversed(self.minibatch_vars['rewards'][:-1]):
            curr_reward = reward + self.network.q_discount * curr_reward
            td_rewards.append(curr_reward)
        # td rewards is computed backward but minibatch is stored forward so need to reverse
        return list(reversed(td_rewards))

    def minibatch_accumulate(self, state, action, reward):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions']]
