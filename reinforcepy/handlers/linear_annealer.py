import numpy as np


class LinnearAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        lin = np.linspace(start, end, steps)
        self.curr_val = start
        self.diff = lin[0] - lin[1]

    def anneal(self):
        """
        Anneals the value by one step
        """
        self.curr_val -= self.diff
        if self.curr_val < self.end:
            self.curr_val = self.end

    def anneal_to(self, anneal_count):
        """
        Anneals the value to a specific step.

        Parameters
        ----------
        anneal_count : int
            Step count to anneal to
        """
        self.curr_val = self.start - self.diff * anneal_count
        if self.curr_val < self.end:
            self.curr_val = self.end
