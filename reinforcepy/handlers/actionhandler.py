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
    The :class:`ActionHandler` class takes care of the interface between the action indexes returned from an environment
    and a vector of length (num actions). It also allows two different types of stochastic selection methods.
    :class:`ActionPolicy`-eGreedy where it randomly selects an action with probability e. Or
    :class:`ActionPolicy`-randVals where it adds noise to the action vector before choosing the index of the max action.

    This class supports linear annealing of both the eGreedy probability value and the randVals scalar.

    Parameters
    ----------
    action_policy : :class:`ActionPolicy`
       Specifies whether using eGreedy or adding randVals to the action value vector

    random_values : tuple
       Specifies which values to use for the action policy
       format: (Initial random value, ending random value, number of steps to anneal over)

    actions : tuple, list, array
       Default None, should be set by calling set_legal_actions.
    """
    def __init__(self, random_values: tuple, action_policy: ActionPolicy=ActionPolicy.eGreedy, actions: np.ndarray=None):
        self.action_policy = action_policy

        self.highest_rand_val = random_values[0]
        self.lowest_rand_val = random_values[1]
        lin = np.linspace(random_values[0], random_values[1], random_values[2])
        self.curr_rand_val = self.highest_rand_val
        self.diff = lin[0] - lin[1]
        self.rand_count = 0
        self.action_count = 0

        if actions is not None:
            self.numActions = len(actions)
            self.actions = None
            self.set_legal_actions(actions)
        else:
            self.numActions = 0
            self.actions = None

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
        if random:
            if self.action_policy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action_values
                if np.random.uniform(0, 1) <= self.curr_rand_val:
                    e_greedy = np.random.randint(self.numActions)
                    action_values[e_greedy] = np.inf  # set this value as the max action
                    self.rand_count += 1
            elif self.action_policy == ActionPolicy.randVals:
                action_values += np.random.randn(self.numActions) * self.curr_rand_val

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

    def set_legal_actions(self, legal_actions):
        """
        Sets the legal actions for this handler. Sets up values need for conversion from environment action ids to
        the learner output ids.

        Parameters
        ----------
        legal_actions : array/list/tuple
            Legal actions in current environment
        """
        self.actions = np.asarray(legal_actions, dtype=int)
        assert len(self.actions.shape) == 1, "actions must be a vector"
        self.numActions = len(legal_actions)

    def game_action_to_action_ind(self, action):
        """
        Converts an action id returned from environment to the index used in the learner.

        Parameters
        ----------
        action : int
            Environment action index

        Returns
        -------
        action_ind : int
            Action index relative to learner output vector
        """
        return np.where(action == self.actions)[0][0]

    def action_vect_to_game_action(self, action_vect: np.ndarray, random=True):
        """
        Converts action vector output of learner to a environment ready action id.

        Parameters
        ----------
        action_vect : array
            Action vector output from learner
        random : bool
            Default True. Whether or not to use stochastic action policy

        Returns
        -------
        env_action_ind : int
            Environment ready action id
        """
        return self.actions[self.get_action(action_vect, random)]

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
                e_greedy = np.random.randint(self.numActions)
                self.rand_count += 1
                return True, self.actions[e_greedy]
            else:
                return False, None
