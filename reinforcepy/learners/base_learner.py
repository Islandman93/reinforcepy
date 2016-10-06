from abc import ABCMeta


class BaseLearner(metaclass=ABCMeta):
    """
    Base leaner abstract class.
    """
    def run_episode(self, environment):
        """
        Runs an episode. The learner can determine when the episode has stopped
        Should return the total reward
        """
        raise NotImplementedError("Subclasses must implement run_episode")
