from ..base_sarsa_learner import BaseSarsaLearner
from .qtable import QTable
from ...handlers import ActionHandler
import numpy as np


class SARSATable(QTable, BaseSarsaLearner):
    # override run_episode here so we can run the sarsa run_episode
    def run_episode(self, environment):
        return BaseSarsaLearner.run_episode(self, environment)

    def update(self, state, action, reward, state_tp1, action_tp1, terminal):
        # https://en.wikipedia.org/wiki/State-Action-Reward-State-Action
        # Table[state][action] = old value + lr * (reward + discount * Table[state_tp1][action_tp1] - old value)
        future_value_array = self.q_table.get(state_tp1)

        # state_tp1 doesn't exist add it
        if future_value_array is None:
            self.q_table[state_tp1] = np.zeros(self.action_size)
            future_value = 0
        else:
            # sarsa is the reward of the action taken not the max
            future_value = future_value_array[action_tp1]

        old_value = self.q_table[state][action]
        inside_value = reward + self.discount * future_value - old_value
        self.q_table[state][action] = old_value + self.learning_rate * inside_value
