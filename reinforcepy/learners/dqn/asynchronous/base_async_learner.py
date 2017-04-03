import numpy as np
from reinforcepy.handlers import ActionHandler
from reinforcepy.handlers.framebuffer import FrameBuffer
from reinforcepy.learners.base_learner import BaseLearner


class BaseAsyncLearner(BaseLearner):
    def __init__(self, environment, network, async_handler, phi_length=4,
                 async_update_step=5, reward_clip_vals=[-1, 1], random_policy=True, epsilon_annealing_start=1,
                 epsilon_annealing_choices=[0.1, 0.01, 0.5], epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, seed=np.random.RandomState(),
                 testing=False):
        super().__init__()

        # these can either be generators or instances
        # if they are generators they will be called in the run function
        self.environment = environment
        self.network = network

        # If doing a random policy (E-greedy)
        self.random_policy = random_policy
        if self.random_policy:
            self.epsilon_annealing_choices = epsilon_annealing_choices
            self.epsilon_annealing_probabilities = epsilon_annealing_probabilities
            self.epsilon_annealing_start = epsilon_annealing_start
            self.epsilon_annealing_steps = epsilon_annealing_steps
            self.action_handler = None  # we setup action handler in run
        self.seed = seed
        self.step_count = 0
        self.reward_clip_vals = reward_clip_vals

        self.phi_length = phi_length
        self.frame_buffer = None  # we setup fram_buffer in run

        self.async_update_step = async_update_step
        self.async_handler = async_handler
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
        # initialize some vars that are async
        # generate or set environment and network
        if callable(self.environment):
            self.environment = self.environment()
        if callable(self.network):
            self.network = self.network()

        self.frame_buffer = FrameBuffer([1, self.phi_length] + self.environment.get_state_shape())
        # If doing a random policy (E-greedy)
        if self.random_policy:
            # initialize action handler, ending E-greedy is either 0.1, 0.01, 0.5 with probability 0.4, 0.3, 0.3
            end_rand = self.seed.choice(self.epsilon_annealing_choices, p=self.epsilon_annealing_probabilities)
            rand_vals = (self.epsilon_annealing_start, end_rand, self.epsilon_annealing_steps)
            self.action_handler = ActionHandler(self.environment.get_num_actions(), rand_vals, seed=self.seed)  # we set num actions later

        self.frame_buffer = FrameBuffer([1, self.phi_length] + self.environment.get_state_shape())

        # run games until done
        try:
            while not self.async_handler.done:
                reward = self.run_episode(self.environment)
                self.print_episode_end_status(reward)
                self.async_handler.add_reward(reward)
        except KeyboardInterrupt:
            print(self, 'Exiting')

    def print_episode_end_status(self, reward):
        curr_rand_val = ''
        if self.random_policy:
            curr_rand_val = 'Curr Rand Val: {0}'.format(self.action_handler.curr_rand_val)
        print(self, 'Episode reward:', reward, 'Steps:', self.environment.curr_step_count,
              'Step count:', self.step_count, curr_rand_val)

    def update(self, *args, **kwargs):
        raise NotImplementedError('Base onestep learner does not implement update.')

    def anneal_random_policy(self):
        if self.random_policy:
            # anneal action handler
            anneal_step = self.async_handler.global_step if self.global_epsilon_annealing else self.step_count
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

    def reset_minibatch(self):
        pass
