import numpy as np
from reinforcepy.handlers import FrameBuffer
from reinforcepy.handlers import ActionHandler
from reinforcepy.learners.base_async import AsyncClient


class BaseAsyncTargetLearner(AsyncClient):
    def __init__(self, learner_parms, network_partial, pipe):
        super().__init__(pipe)

        learner_parms.required(['skip_frame', 'phi_length', 'async_update_step', 'target_update_frames'])

        # initialize action handler, ending E-greedy is either 0.1, 0.01, 0.5 with probability 0.4, 0.3, 0.3
        end_rand = np.random.choice([0.1, 0.01, 0.5], p=[0.4, 0.3, 0.3])
        rand_vals = (1, end_rand, 4000000)  # anneal over four million frames
        self.action_handler = ActionHandler(rand_vals)

        # initialize network
        self.cnn = network_partial()
        self.frame_buffer = FrameBuffer((1, learner_parms.get('phi_length'), 84, 84), dtype=np.float32)

        self.skip_frame = learner_parms.get('skip_frame')
        self.phi_length = learner_parms.get('phi_length')
        self.loss_list = list()

        self.async_update_step = learner_parms.get('async_update_step')
        self.target_update_frames = learner_parms.get('target_update_frames')
        self.target_update_count = 0

    def check_update_target(self, total_frames_count):
        if total_frames_count >= self.target_update_count * self.target_update_frames:
            self.target_update_count += 1
            return True
        return False

    def get_action(self, state):
        return self.cnn.get_output(state)[0]

    def get_game_action(self, state):
        # checks to see if we are doing random, if so returns random game action
        rand, action = self.action_handler.get_random()
        if not rand:
            action = self.get_action(state)
            return self.action_handler.action_vect_to_game_action(action, random=False)
        return action

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)
