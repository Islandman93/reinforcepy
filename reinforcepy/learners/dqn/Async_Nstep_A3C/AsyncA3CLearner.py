from reinforcepy.learners.base_async import AsyncProcessClient
from reinforcepy.handlers import ActionHandler
import numpy as np


class AsyncProcessA3CLearner(AsyncProcessClient):
    def __init__(self, num_actions, initial_cnn_values, cnn_partial, pipe,
                 skip_frame=4, phi_length=4, async_update_step=5):
        super().__init__(pipe)

        # A3C doesn't have an EGreedy exploration policy so we set the random values to 0
        self.action_handler = ActionHandler((0, 0, 2))

        # initialize network
        self.cnn = cnn_partial()
        self.cnn.set_parameters(initial_cnn_values)
        self.frame_buffer = np.zeros((1, phi_length, 84, 84), dtype=np.float32)

        self.skip_frame = skip_frame
        self.phi_length = phi_length
        self.loss_list = list()

        self.async_update_step = async_update_step

    def add_state_to_buffer(self, state):
        self.frame_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
        self.frame_buffer[0, self.phi_length-1] = state

    def frame_buffer_with(self, state):
        empty_buffer = np.zeros((1, self.phi_length, 84, 84), dtype=np.float32)
        empty_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
        empty_buffer[0, self.phi_length-1] = state
        return empty_buffer

    def get_action(self, frame_buffer):
        return self.cnn.get_policy_output(frame_buffer)[0]

    def get_game_action(self, frame_buffer):
        action = self.get_action(frame_buffer)
        return self.action_handler.action_vect_to_game_action(action, random=False)

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)
