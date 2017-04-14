from abc import ABCMeta


class BaseEnvironment(metaclass=ABCMeta):
    """
    Base environment abstract class.
    """
    def step(self, action: int):
        """
        Step should take an environment ready action index. Use it to run the next step of the environment. Then return
        the reward

        Parameters
        ----------
        action : int
            Environment ready action index

        Returns
        -------
        reward : int
            Reward gained for taking action
        """
        pass

    def reset(self):
        """
        Reset should reset the environment back to it's starting position
        """
        pass

    def get_terminal(self):
        """
        Tests whether the environment episode is over

        Returns
        -------
        terminal : bool
            Whether or not the episode is over (terminal state reached)
        """
        pass

    def get_state(self):
        """
        Should return the current state of the environment in whatever format the learner needs
        """
        pass

    def get_state_shape(self):
        """
        Returns the shape of the game state
        """
        pass

    def get_num_actions(self):
        """
        Should return the number of actions that can be performed in the environment. These should never change after
        initialization

        Returns
        -------
        num_actions : int
            Number of actions for this environment
        """
        pass

    def close(self):
        """
        Deletes the environment, not required
        """
        pass

    def get_step_count(self):
        """
        Returns the number of steps the environment has taken
        """
        pass
