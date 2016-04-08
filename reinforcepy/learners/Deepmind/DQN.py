import numpy as np
from reinforcepy.handlers import ActionHandler
from reinforcepy.handlers.experience_replay import DataSet
from reinforcepy.learners import BaseLearner


class DQNLearner(BaseLearner):
    def __init__(self, learner_parms, network):
        # set required parameters
        learner_parms.required(['skip_frame', 'anneal_egreedy_steps', 'dataset_shape', 'max_dataset_size', 'phi_length',
                                'minimum_replay_size', 'minibatch_size'])

        # initialize action handler
        rand_vals = (1, 0.1, learner_parms.get('anneal_egreedy_steps'))  # starting at 1 anneal eGreedy policy to 0.1
        self.action_handler = ActionHandler(rand_vals)

        # set cnn to passed in network
        self.cnn = network

        # initialize experience replay
        dataset_shape = learner_parms.get('dataset_shape')
        self.exp_replay = DataSet(dataset_shape['width'], dataset_shape['height'], max_steps=learner_parms.get('max_dataset_size'),
                                   phi_length=learner_parms.get('phi_length'))
        self.minimum_replay_size = learner_parms.get('minimum_replay_size')

        # initialize other vars
        self.skip_frame = learner_parms.get('skip_frame')
        self.cost_list = list()
        self.step_count = 0

    def run_epoch(self, environment, epoch_step_count=50000):
        episode_rewards = list()
        start_step_count = self.step_count
        while (self.step_count - start_step_count) < epoch_step_count:
            episode_rewards.append(self.run_episode(environment))
        return episode_rewards

    def run_episode(self, environment):
        # reset
        total_reward = 0.0
        environment.reset()

        # loop till terminal
        terminal = False
        while not terminal:
            # get action
            state = environment.get_state()

            # check if doing random action
            random, action = self.action_handler.get_random()
            if not random:
                action = self.get_game_action(self.exp_replay.phi(state))

            # step and get new state
            reward = 0
            for frame in range(self.skip_frame):
                env_reward = environment.step(action)
                reward += np.clip(env_reward, 0, 1)

            # check for terminal
            terminal = environment.get_terminal()

            # add experience to memory
            self.exp_replay.add_sample(state, self.action_handler.game_action_to_action_ind(action), reward, terminal)

            # run minibatch
            self.run_minibatch()

            # anneal action handler
            self.step_count += 1
            self.action_handler.anneal_to(self.step_count)

            total_reward += reward

        # end of episode
        return total_reward

    def run_minibatch(self):
        # generate minibatch data
        if self.exp_replay.size > self.minimum_replay_size:
            states, actions, rewards, state_tp1s, terminal = self.exp_replay.random_batch(32)
            cost = self.cnn.train(states / 255.0, actions, rewards, state_tp1s / 255.0, terminal)
            self.cost_list.append(cost)

    def get_game_action(self, state):
        cnn_action = self.get_action(state.reshape((1, self.skip_frame, 84, 84)) / 255.0)
        return self.action_handler.action_vect_to_game_action(cnn_action)

    def get_action(self, processed_screens):
        return self.cnn.get_output(processed_screens)[0]

    def get_status(self):
        return 'Step Count: {0}, Current Rand Val: {1}'.format(self.step_count, self.action_handler.curr_rand_val)

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)

    def save(self, filename):
        self.cnn.save(filename)
