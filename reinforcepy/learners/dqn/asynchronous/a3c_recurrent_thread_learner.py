import numpy as np
from .a3c_dqn_thread_learner import A3CThreadLearner


class A3CRecurrentThreadLearner(A3CThreadLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_state_for_training = self.network.get_lstm_state()

    def reset(self):
        super().reset()
        self.network.reset_lstm_state()
        self.lstm_state_for_training = self.network.get_lstm_state()

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

        # check perform gradient step
        if self.step_count % self.async_update_step == 0 or terminal:
            td_rewards = self.calculate_td_reward(self.frame_buffer.get_buffer_with(state_tp1), terminal)
            summaries = self.global_dict['write_summaries_this_step']
            if summaries:
                self.global_dict['write_summaries_this_step'] = False
                summary = self.network.train_step(self.global_dict['learning_rate'], *self.get_minibatch_vars(), lstm_state=self.lstm_state_for_training, reward=td_rewards, summaries=True)
                self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
            else:
                self.network.train_step(self.global_dict['learning_rate'], *self.get_minibatch_vars(), lstm_state=self.lstm_state_for_training, reward=td_rewards, summaries=False)
            self.reset_minibatch()

            self.lstm_state_for_training = self.network.get_lstm_state()

        self.anneal_random_policy()
