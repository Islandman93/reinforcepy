import numpy as np
from reinforcepy.handlers.experience_replay import DataSet
from reinforcepy.learners import BaseQLearner
from .base_thread_learner import BaseThreadLearner


class ExpQThreadLearner(BaseThreadLearner, BaseQLearner):
    def __init__(self, environment, network, global_dict, dataset_size, batch_size, **kwargs):
        super().__init__(environment, network, global_dict, **kwargs)
        env_shape = environment.get_state_shape()
        self.batch_size = batch_size
        self.dataset = DataSet(env_shape[0], env_shape[1], max_steps=dataset_size, phi_length=self.phi_length)

    def update(self, state, action, reward, state_tp1, terminal):
        self.frame_buffer.add_state_to_buffer(state)

        # quit update if testing
        if self.testing:
            return

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
        if self.steps_since_train % self.async_update_step == 0 or terminal:
            minibatch_vars = self.dataset.random_batch(self.batch_size)
            self.network.train_step(*minibatch_vars, global_step=self.global_dict['counter'])

        # anneal action handler
        self.anneal_random_policy()
