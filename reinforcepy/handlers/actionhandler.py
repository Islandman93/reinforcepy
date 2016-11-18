import numbers
import numpy as np
from enum import Enum


class ActionPolicy(Enum):
    """
    :class:`ActionPolicy` is an Enum used to determine which policy an action handler should use for
    random exploration.

    Currently supported are eGreedy and the addition of random values to the action vector (randVals)

    The idea behind adding random values can be found here:
    https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
    """
    eGreedy = 1
    randVals = 2


class ActionHandler:
    """
    The :class:`ActionHandler` class takes care of two different types of stochastic selection methods.
    :class:`ActionPolicy`-eGreedy where it randomly selects an action with probability e. Or
    :class:`ActionPolicy`-randVals where it adds noise to the action vector before choosing the index of the max action.

    This class supports linear annealing of both the eGreedy probability value and the randVals scalar.

    Parameters
    ----------
    num_actions : int
        Total number of actions

    random_values : tuple or int
        Specifies which values to use for the action policy. If int, no annealing will take place
        format: (Initial random value, ending random value, number of steps to anneal over)

    action_policy : :class:`ActionPolicy`
        Specifies whether using eGreedy or adding randVals to the action value vector
    """
    def __init__(self, num_actions: int, random_values, action_policy: ActionPolicy=ActionPolicy.eGreedy):
        self.action_policy = action_policy
        self.num_actions = num_actions

        # check if random values is just a number
        if isinstance(random_values, numbers.Number):
            self.highest_rand_val = random_values
            self.lowest_rand_val = random_values
            self.curr_rand_val = random_values
            self.diff = 0
        else:
            self.highest_rand_val = random_values[0]
            self.lowest_rand_val = random_values[1]
            lin = np.linspace(random_values[0], random_values[1], random_values[2])
            self.curr_rand_val = self.highest_rand_val
            self.diff = lin[0] - lin[1]
            # if highest_rand_val is less than lowest_rand_val set lowest_rand_val to the lower
            self.lowest_rand_val = self.lowest_rand_val if self.lowest_rand_val < self.highest_rand_val else self.highest_rand_val
        self.rand_count = 0
        self.action_count = 0

    def get_action(self, action_values, random=True):
        """
        Get_Action takes an action_values vector from a learner of length # legal actions and will perform the
        stochastic selection policy on it.

        Parameters
        ----------
        action_values : array of length # legal actions
            Output from a learner of values for each possible action

        random : bool
            Default true. Whether to perform the stochastic action_values selection policy or just return the max value
            index.

        Returns
        -------
        action_ind : int
            Index of max action value.
        """
        action = None
        if random:
            if self.action_policy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action_values
                if np.random.uniform(0, 1) <= self.curr_rand_val:
                    e_greedy = np.random.randint(self.num_actions)
                    action = e_greedy
                    self.rand_count += 1
            elif self.action_policy == ActionPolicy.randVals:
                action_values += np.random.randn(self.num_actions) * self.curr_rand_val

        if action is None:
            action = np.argmax(action_values)
        self.action_count += 1

        return action

    def anneal(self):
        """
        Anneals the random value used in the stochastic action selection policy.
        """
        self.curr_rand_val -= self.diff
        if self.curr_rand_val < self.lowest_rand_val:
            self.curr_rand_val = self.lowest_rand_val

    def anneal_to(self, anneal_count):
        """
        Anneals the random value to a specific step.

        Parameters
        ----------
        anneal_count : int
            Step count to anneal to
        """
        self.curr_rand_val = self.highest_rand_val - self.diff * anneal_count
        if self.curr_rand_val < self.lowest_rand_val:
            self.curr_rand_val = self.lowest_rand_val

    def get_random(self):
        """
        Runs the action policy to see if we are doing a random move. Useful if generating an action from your learner
        takes a long time. No reason to run the learner selection code if a random move is selected.

        Returns
        -------
        If action is random :
            tuple (True, action_index)
        If not random:
            tuple (False, None)
        """
        if self.action_policy == ActionPolicy.eGreedy:
            # egreedy policy to choose random action_values
            if np.random.uniform(0, 1) <= self.curr_rand_val:
                e_greedy = np.random.randint(self.num_actions)
                self.rand_count += 1
                return True, e_greedy
            else:
                return False, None

    def get_random_action(self):
        """
        Returns a random action

        Returns
        -------
        Random action index
        """
        return np.random.randint(self.num_actions)
