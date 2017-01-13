import threading
import numpy as np
from reinforcepy.handlers import ActionHandler
from reinforcepy.handlers.framebuffer import FrameBuffer


class BaseThreadLearner(threading.Thread):
    def __init__(self, environment, network, global_dict, phi_length=4,
                 async_update_step=5, target_update_steps=10000, reward_clip_vals=[-1, 1], random_policy=True, epsilon_annealing_start=1,
                 epsilon_annealing_choices=[0.1, 0.01, 0.5], epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True,
                 testing=False):
        super().__init__()

        # If doing a random policy (E-greedy)
        self.random_policy = random_policy
        if random_policy:
            # initialize action handler, ending E-greedy is either 0.1, 0.01, 0.5 with probability 0.4, 0.3, 0.3
            end_rand = np.random.choice(epsilon_annealing_choices, p=epsilon_annealing_probabilities)
            rand_vals = (epsilon_annealing_start, end_rand, epsilon_annealing_steps)
            self.action_handler = ActionHandler(environment.get_num_actions(), rand_vals)  # we set num actions later
        self.step_count = 0
        self.environment = environment
        self.reward_clip_vals = reward_clip_vals

        # network stuff
        self.network = network

        self.phi_length = phi_length
        self.frame_buffer = FrameBuffer([1, phi_length] + environment.get_state_shape())

        self.async_update_step = async_update_step
        self.target_update_steps = target_update_steps
        self.global_dict = global_dict
        self.global_epsilon_annealing = global_epsilon_annealing

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

    def run(self):
        while not self.global_dict['done']:
            reward = self.run_episode(self.environment)
            self.global_dict['add_reward'](reward)
            curr_rand_val = ''
            if self.random_policy:
                curr_rand_val = 'Curr Rand Val: {0}'.format(self.action_handler.curr_rand_val)
            print(self, 'Episode reward:', reward, 'Step count:', self.step_count, curr_rand_val)

    def update(self, *args, **kwargs):
        raise NotImplementedError('Base onestep learner does not implement update.')

    def anneal_random_policy(self):
        if self.random_policy:
            # anneal action handler
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.action_handler.anneal_to(anneal_step)

    def get_action(self, state):
        """
        Gets an action for the current state. First queries action_handler to see
        if we should execute a random action. If random action, then don't send to gpu
        """
        if self.random_policy:
            # check if doing random action
            random, action = self.action_handler.get_random()
            if not random:
                return self.network.get_output(self.frame_buffer.get_buffer_with(state))
            return action
        else:
            return self.network.get_output(self.frame_buffer.get_buffer_with(state))
