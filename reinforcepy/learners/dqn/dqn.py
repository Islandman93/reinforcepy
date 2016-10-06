import numpy as np
from reinforcepy.handlers import ActionHandler
from reinforcepy.handlers.experience_replay import DataSet
from reinforcepy.learners import BaseQLearner


class DQNLearner(BaseQLearner):
    def __init__(self, learner_parms, network, batch_size=32, testing=False):
        # set required parameters
        learner_parms.required(['skip_frame', 'egreedy_policy', 'dataset_shape', 'max_dataset_size', 'phi_length',
                                'minimum_replay_size', 'minibatch_size'])

        # initialize action handler
        rand_vals = learner_parms.get('egreedy_policy')  # starting at 1 anneal eGreedy policy to 0.1
        self.action_handler = ActionHandler(0, rand_vals)

        # set cnn to passed in network
        self.cnn = network

        # initialize experience replay
        dataset_shape = learner_parms.get('dataset_shape')
        self.exp_replay = DataSet(dataset_shape['width'], dataset_shape['height'],
                                  max_steps=learner_parms.get('max_dataset_size'),
                                  phi_length=learner_parms.get('phi_length'))
        self.minimum_replay_size = learner_parms.get('minimum_replay_size')

        # initialize other vars
        self.batch_size = batch_size
        self.skip_frame = learner_parms.get('skip_frame')
        self.step_count = 0

        self.testing = testing

    def run_epoch(self, environment, epoch_step_count=50000):
        episode_rewards = list()
        start_step_count = self.step_count
        while (self.step_count - start_step_count) < epoch_step_count:
            episode_rewards.append(self.run_episode(environment))
        return episode_rewards

    def get_action(self, state):
        """
        Gets an action for the current state. First queries action_handler to see
        if we should execute a random action. If random action, then don't send to gpu
        """
        # check if doing random action
        random, action = self.action_handler.get_random()
        if not random:
            # NOTICE: we already check random above
            cnn_action_values = self.get_action_values(self.exp_replay.phi(state).reshape((1, self.skip_frame, 84, 84)))
            return self.action_handler.get_action(cnn_action_values, random=False)
        return action

    def update(self, state, action, reward, state_tp1, terminal):
        """
        Adds experience to memory, runs a minibatch and anneals the random policy
        """
        reward = np.clip(reward, -1, 1)
        self.exp_replay.add_sample(state, action, reward, terminal)

        if not self.testing:
            self.run_minibatch()

        # anneal action handler
        self.step_count += 1
        self.action_handler.anneal_to(self.step_count)

    def run_minibatch(self):
        # generate minibatch data
        if self.exp_replay.size > self.minimum_replay_size:
            states, actions, rewards, state_tp1s, terminal = self.exp_replay.random_batch(self.batch_size)
            self.cnn.train(states, actions, rewards, state_tp1s, terminal)

    def get_action_values(self, processed_screens):
        return self.cnn.get_output(processed_screens)

    def get_status(self):
        return 'Step Count: {0}, Current Rand Val: {1}'.format(self.step_count, self.action_handler.curr_rand_val)

    def set_action_num(self, num_actions):
        self.action_handler.num_actions = num_actions

    def save(self, filename):
        self.cnn.save(filename)
