import threading
from queue import Queue


class ThreadedGameHandler:
    """
    The :class:`ThreadedGameHandler` class is used to be able to run multiple learners on multiple environment instances.
    It uses :class:'GameHandler' to communicate between the ALE and learner

    Parameters
    ----------
    environments: list
        list of already created environments.
    """
    def __init__(self, environments):
        # setup list of gamehandlers and their locks
        self.environments = list()
        for env in environments:
            self.environments.append((env, threading.Lock()))

        # setup thread queue
        self.queue = Queue()

        # lock for unlocking/locking environments
        self.environment_lock = threading.Lock()
        self.current_environment = 0
        self.num_environments = len(self.environments)

    def async_run_environment(self, learner, done_fn):
        # push to queue
        self.queue.put(self._get_next_environment())
        t = threading.Thread(target=self._thread_run_environment, args=(learner, done_fn))
        t.daemon = True
        t.start()

    def _thread_run_environment(self, learner, done_fn):
        # get an environment
        environment, environment_lock = self.queue.get()
        with environment_lock:
            total_reward = learner.run_episode(environment)
        done_fn(total_reward)
        self.queue.task_done()

    def block_until_done(self):
        self.queue.join()

    def _get_next_environment(self):
        with self.environment_lock:
            environment = self.environments[self.current_environment]
            self.current_environment += 1
            self.current_environment %= self.num_environments
        return environment

    def get_legal_actions(self):
        return self.environments[0][0].get_legal_actions()