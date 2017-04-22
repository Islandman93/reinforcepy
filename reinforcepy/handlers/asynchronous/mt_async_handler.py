from .base_async_handler import BaseAsyncHandler
import numpy as np
import visdom


# NOTE: due to pythons GIL threads are not executed at the same time,
# therefore we don't have to lock anything
class MTAsyncHandler(BaseAsyncHandler):
    def __init__(self, starting_global_step=0, starting_rewards=[]):
        self._global_step = starting_global_step
        self.rewards = starting_rewards
        self._done = False
        self.vis = visdom.Visdom()
        self.win = None

    @property
    def global_step(self):
        return self._global_step

    def increment_global_step(self, increment_amount=1):
        self._global_step += increment_amount

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, new_val):
        self._done = new_val

    def add_reward(self, reward):
        if self.win is None:
            self.win = self.vis.line(np.asarray([reward]), np.asarray([self._global_step]))
        else:
            self.vis.updateTrace(np.asarray([self._global_step]), np.asarray([reward]), self.win)
