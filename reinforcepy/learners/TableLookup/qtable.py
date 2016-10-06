from ..base_q_learner import BaseQLearner
from ...handlers import ActionHandler
import numpy as np


class QTable(BaseQLearner):
    def __init__(self, action_size, anneal_tuple, learning_rate=0.1, discount=0.99):
        self.q_table = {}
        self.action_size = action_size
        self.action_helper = ActionHandler(anneal_tuple, actions=np.arange(action_size))

        self.learning_rate = learning_rate
        self.discount = discount

        self.episode_count = 0

    def get_action(self, state):
        """
        Gets an action given state. State must be hashable.
        If state is not in the table returns a random action. Else returns action
        selected by current eGreedy policy.
        """
        # try to get action from table
        action_values = self.q_table.get(state)

        # if action is None initialize state in table
        if action_values is None:
            self.q_table[state] = np.zeros(self.action_size)
            # return random action
            return self.action_helper.get_random_action()
        # else either do random or return arg max
        else:
            return self.action_helper.get_action(action_values)

    def update(self, state, action, reward, state_tp1, terminal):
        # https://en.wikipedia.org/wiki/Q-learning
        # QTable[state][action] = old value + lr * (reward + discount * max(Q[state_tp1]) - old value)
        future_value_array = self.q_table.get(state_tp1)

        # state_tp1 doesn't exist add it
        if future_value_array is None:
            self.q_table[state_tp1] = np.zeros(self.action_size)
            future_value = 0
        else:
            future_value = np.max(future_value_array)

        old_value = self.q_table[state][action]
        inside_value = reward + self.discount * future_value - old_value
        self.q_table[state][action] = old_value + self.learning_rate * inside_value

    def episode_end(self):
        self.episode_count += 1
        self.action_helper.anneal_to(self.episode_count)
