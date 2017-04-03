from abc import ABCMeta


class BaseAsyncHandler(metaclass=ABCMeta):
    def __init__(self, starting_global_step=0, starting_rewards=[]):
        pass

    @property
    def global_step(self):
        pass

    def increment_global_step(self, increment_amount=1):
        pass

    def add_reward(self, reward):
        pass

    @property
    def done(self):
        pass

    @done.setter
    def done(self, new_val):
        pass
