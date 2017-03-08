import os
import time
import numpy as np
import torch.multiprocessing
from reinforcepy.handlers.framebuffer import FrameBuffer


class BaseProcessLearner(torch.multiprocessing.Process):
    def __init__(self, environment, global_dict, phi_length=4,
                 async_update_step=5, reward_clip_vals=[-1, 1], testing=False):
        super().__init__()

        self.step_count = 0
        self._environment_generator = environment
        self.environment = None
        self.reward_clip_vals = reward_clip_vals

        # network stuff
        self.network = None

        self.phi_length = phi_length
        self.frame_buffer = None

        self.async_update_step = async_update_step
        self.global_dict = global_dict

        self.minibatch_vars = {}
        self.reset_minibatch()

        self.testing = testing

    def reset(self):
        self.reset_minibatch()
        self.frame_buffer.reset()

        # initialize the buffer with states
        # TODO: add random starts here
        state = self.environment.get_state()
        for _ in range(self.phi_length):
            self.frame_buffer.add_state_to_buffer(state)
        self.network.load_state_dict(self.global_dict['network'].state_dict())

    def run(self):
        from reinforcepy.networks.dqn.ptorch.nstep_a3c import A3CModel
        try:
            torch.manual_seed(os.getpid())
            self.network = A3CModel()
            for param, shared_param in zip(self.network.parameters(), self.global_dict['network'].parameters()):
                # Use gradients from the local model
                shared_param.grad.data = param.grad.data
            self.network.train()
            self.environment = self._environment_generator(seed=np.random.RandomState(os.getpid()))
            self.frame_buffer = FrameBuffer([1, self.phi_length] + self.environment.get_state_shape())
            st = time.time()
            while True:
                reward = self.run_episode(self.environment)
                self.global_dict['reward_list'].append((reward, self.global_dict['counter']))
                sps = self.step_count / (time.time() - st)
                print(self, 'Episode reward:', reward, 'Steps:', self.environment.curr_step_count,
                      'Step count:', self.step_count, 'SPS: {}'.format(sps))
        except KeyboardInterrupt:
            pass

    def update(self, *args, **kwargs):
        raise NotImplementedError('Base onestep learner does not implement update.')

    def get_action(self, state):
        return self.network.get_output(self.frame_buffer.get_buffer_with(state))

    def reset_minibatch(self):
        pass
