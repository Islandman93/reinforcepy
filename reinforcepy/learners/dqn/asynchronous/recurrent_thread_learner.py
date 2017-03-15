import numpy as np
from .q_thread_learner import QThreadLearner


class RecurrentThreadLearner(QThreadLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_lstm_state = self.network.blank_lstm_state()
        self.lstm_state_for_training = self.network.blank_lstm_state()

    def reset(self):
        super().reset()
        self.current_lstm_state = self.network.blank_lstm_state()
        self.lstm_state_for_training = self.network.blank_lstm_state()

    def get_action(self, state):
        """
        Gets an action for the current state. First queries action_handler to see
        if we should execute a random action. Even if we execute random we have to send to gpu to update lstm state
        """
        network_action, new_lstm_state = self.network.get_output(self.frame_buffer.get_buffer_with(state), self.current_lstm_state)
        self.current_lstm_state = new_lstm_state
        if self.random_policy:
            # check if doing random action
            random, action = self.action_handler.get_random()
            if random:
                return action
            else:
                return network_action
        else:
            return network_action

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
        self.steps_since_train += 1
        self.global_dict['counter'] += 1

        # check perform gradient step
        if self.steps_since_train % self.async_update_step == 0 or terminal:
            summaries = self.global_dict['write_summaries_this_step']
            if summaries:
                self.global_dict['write_summaries_this_step'] = False
                summary = self.network.train_step(*self.get_minibatch_vars(), lstm_state=self.lstm_state_for_training,
                                                  global_step=self.global_dict['counter'], summaries=True)
                self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
            else:
                self.network.train_step(*self.get_minibatch_vars(), lstm_state=self.lstm_state_for_training,
                                        global_step=self.global_dict['counter'], summaries=False)
            self.reset_minibatch()

            self.lstm_state_for_training = self.current_lstm_state

        # anneal action handler
        self.anneal_random_policy()
