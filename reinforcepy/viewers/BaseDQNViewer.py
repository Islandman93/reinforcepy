import numpy as np
from reinforcepy.handlers import FrameBuffer
from reinforcepy.handlers import ActionHandler


class BaseDQNViewer:
    def __init__(self, viewer_parms, network):
        viewer_parms.required(['skip_frame', 'phi_length', 'egreedy_val'])

        # setup action handler to randomly select actions with probability egreedy_val
        egreedy = viewer_parms.get('egreedy_val')
        rand_vals = (egreedy, egreedy, 2)
        self.action_handler = ActionHandler(rand_vals)

        # initialize network
        self.cnn = network
        self.frame_buffer = FrameBuffer((1, viewer_parms.get('phi_length'), 84, 84), dtype=np.float32)

        self.skip_frame = viewer_parms.get('skip_frame')
        self.phi_length = viewer_parms.get('phi_length')
        self.total_reward = 0

    def get_game_action(self, state):
        # checks to see if we are doing random, if so returns random game action
        rand, action = self.action_handler.get_random()
        if not rand:
            action = self.cnn.get_output(state)[0]
            return self.action_handler.action_vect_to_game_action(action, random=False)
        return action

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)

