import numpy as np
from .q_thread_learner import QThreadLearner
from reinforcepy.handlers.experience_replay import DataSet


class AuxiliaryThreadLearner(QThreadLearner):
    def __init__(self, environment, network, global_dict, dataset_size, batch_size, **kwargs):
        super().__init__(environment, network, global_dict, **kwargs)
        env_shape = environment.get_state_shape()
        self.batch_size = batch_size
        self.dataset = DataSet(env_shape[0], env_shape[1], max_steps=dataset_size, phi_length=self.phi_length)
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

        # accumulate minibatch_vars
        self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                  reward, self.frame_buffer.get_buffer_with(state_tp1), terminal)

        # add sample to dataset
        self.dataset.add_sample(state, action, reward, False)

        # if terminal add state_tp1 to dataset,
        # terminal stops the loop so we have to add it here
        if terminal:
            self.dataset.add_sample(state_tp1, 0, 0, terminal)

        # increment counters
        self.step_count += 1
        self.global_dict['counter'] += 1

        # check perform gradient step
        if self.step_count % self.async_update_step == 0 or terminal:
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

            self.lstm_state_for_training = self.network.get_lstm_state()

            # auxiliary tasks
            states, actions, rewards, state_tp1s, terminals = self.dataset.random_sequential_batch(self.batch_size)
            # sequential batch can return none if there isn't a sequential batch of requested size
            if states is not None:
                if summaries:
                    summary = self.network.train_auxiliary_value_replay(states, rewards, state_tp1s, terminals, summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                    summary = self.network.train_auxiliary_pixel_control(states, actions, state_tp1s, summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_auxiliary_value_replay(states, rewards, state_tp1s, terminals, summaries=False)
                    summary = self.network.train_auxiliary_pixel_control(states, actions, state_tp1s, summaries=False)

        # anneal action handler
        self.anneal_random_policy()
