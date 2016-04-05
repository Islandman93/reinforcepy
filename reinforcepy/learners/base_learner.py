from abc import ABCMeta


class BaseLearner(metaclass=ABCMeta):
    """
    Base leaner abstract class.
    """
    def run_episode(self, environment):
        raise NotImplementedError("Subclasses must implement run_episode")
