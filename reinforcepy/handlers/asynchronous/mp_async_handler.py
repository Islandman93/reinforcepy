from multiprocessing import Value
from .base_async_handler import BaseAsyncHandler
from .nonclosing_manager import NonclosingManager


class MPAsyncHandler(BaseAsyncHandler):
    def __init__(self, starting_global_step=0, starting_rewards=[]):
        self._global_step = Value('i', starting_global_step, lock=True)
        self._done = Value('b', False, lock=False)
        self.manager = NonclosingManager()
        self.rewards = self.manager.list()

    @property
    def global_step(self):
        return self._global_step.value

    def increment_global_step(self, increment_amount=1):
        with self._global_step.get_lock():
            self._global_step.value += increment_amount

    @property
    def done(self):
        return self._done.value

    @done.setter
    def done(self, new_val):
        self._done.value = new_val

    def add_reward(self, reward):
        self.rewards.append((reward, self._global_step.value))
