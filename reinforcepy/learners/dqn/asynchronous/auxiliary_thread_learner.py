import numpy as np
from .q_thread_learner import QThreadLearner
from reinforcepy.handlers.experience_replay import DataSet


class AuxiliaryThreadLearner(QThreadLearner):
    def __init__(self, environment, network, global_dict, dataset_size, batch_size,
                 reward_pred_batch_size, auxiliary_tasks, **kwargs):
        super().__init__(environment, network, global_dict, **kwargs)
        env_shape = environment.get_state_shape()
        self.batch_size = batch_size
        self.reward_pred_batch_size = reward_pred_batch_size
        self.dataset = DataSet(env_shape[0], env_shape[1], max_steps=dataset_size, phi_length=self.phi_length)
        self.steps_since_train = 0
        self.auxiliary_tasks = auxiliary_tasks

    def reset(self):
        state = self.environment.get_state()
        for _ in range(self.phi_length + 1):
            self.dataset.add_sample(state, 0, 0, False)
        self.steps_since_train = 0

    def get_action(self, state):
        return self.network.get_output(np.expand_dims(self.dataset.phi(state), 0))

    def update(self, state, action, reward, state_tp1, terminal):
        # clip reward
        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        # add sample to dataset
        self.dataset.add_sample(state, action, reward, False)

        # if terminal add state_tp1 to dataset,
        # terminal stops the loop so we have to add it here
        if terminal:
            self.dataset.add_sample(state_tp1, 0, 0, terminal)

        # increment counters
        self.step_count += 1
        self.steps_since_train += 1
        self.global_dict['counter'] += 1

        # check perform gradient step
        if not self.testing and (self.steps_since_train % self.async_update_step == 0 or terminal):
            summaries = self.global_dict['write_summaries_this_step']
            if not terminal:
                last_batch = self.dataset.last_batch(self.steps_since_train, state_tp1)
            else:
                last_batch = self.dataset.last_batch(self.steps_since_train)
            if summaries:
                self.global_dict['write_summaries_this_step'] = False
                summary = self.network.train_step(*last_batch, global_step=self.global_dict['counter'], summaries=True)
                self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
            else:
                self.network.train_step(*last_batch, global_step=self.global_dict['counter'], summaries=False)

            # auxiliary tasks
            # pixel control
            if 'pixel_control' in self.auxiliary_tasks:
                states, actions, rewards, state_tp1s, terminals = self.dataset.random_sequential_batch(self.batch_size)
                # sequential batch can return none if there isn't a sequential batch of requested size
                if states is not None:
                    if summaries:
                        summary = self.network.train_auxiliary_pixel_control(states, actions, state_tp1s, terminals, summaries=True)
                        self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                    else:
                        self.network.train_auxiliary_pixel_control(states, actions, state_tp1s, terminals, summaries=False)

            # value replay
            if 'value_replay' in self.auxiliary_tasks:
                states, actions, rewards, state_tp1s, terminals = self.dataset.random_sequential_batch(self.batch_size)
                # sequential batch can return none if there isn't a sequential batch of requested size
                if states is not None:
                    if summaries:
                        summary = self.network.train_auxiliary_value_replay(states, rewards, state_tp1s, terminals, summaries=True)
                        self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                    else:
                        self.network.train_auxiliary_value_replay(states, rewards, state_tp1s, terminals, summaries=False)

            # reward prediction
            if 'reward_prediction' in self.auxiliary_tasks:
                states, reward_tp1 = self.dataset.reward_prediction_prioritized_sample(self.reward_pred_batch_size)
                # reward_prioritized_sequential_batch can return none if no rewards in dataset or it can't find any
                if states is not None:
                    if summaries:
                        summary = self.network.train_auxiliary_reward_preditiction(states, reward_tp1, summaries=True)
                        self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                    else:
                        self.network.train_auxiliary_reward_preditiction(states, reward_tp1, summaries=False)

            self.steps_since_train = 0
