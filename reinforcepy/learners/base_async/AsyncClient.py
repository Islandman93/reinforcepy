from reinforcepy.learners.base_async.PipeCmds import PipeCmds


class AsyncClient:
    def __init__(self, pipe):
        # client stuff
        self.thread_steps = 0
        self.pipe = pipe
        self.done = False

    def run(self, environment):
        while not self.done:
            try:
                self.run_episode(environment)
            except ConnectionAbortedError:
                print(self, "Successfully stopped")

    def synchronous_update(self, gradients, frames, extra_parms=None):
        # send accumulated grads
        self.pipe.send((PipeCmds.ClientSendingGradientsSteps, (gradients, frames, extra_parms)))

        pipe_cmd, extras = self.pipe.recv()
        if pipe_cmd == PipeCmds.HostSendingGlobalParameters:
            (new_params, global_vars) = extras
            return new_params, global_vars
        elif pipe_cmd == PipeCmds.End:
            self.done = True
            self.pipe.close()
            raise ConnectionAbortedError

    def process_host_cmds(self):
        pipe_recieved = False
        while self.pipe.poll():
            pipe_cmd, extras = self.pipe.recv()
            pipe_recieved = True
            if pipe_cmd == PipeCmds.End:
                self.done = True
                self.pipe.close()
                raise ConnectionAbortedError
            else:
                print('Dropping pipe command', pipe_cmd, 'and continuing')
        return pipe_recieved

    def send_stats(self, stats):
        self.pipe.send((PipeCmds.ClientSendingStats, stats))


from multiprocessing import Process
class AsyncClientProcess(Process):
    """
    A class that runs a given learner and environment in it's own process. Uses a pipe to communicate back to the host.
    Parameters are passed in by using process(args=(pipe connection, learner_partial, environment_partial))

    Parameters
    ----------
        pipe_conn : :class:`Connection`
            Pipe child connection to communicate with host
        learner_partial : partial(Learner, args*)
            Learner partial function to construct the learner. Pipe to host will be passed in as last var
        environment_partial : partial(Emulator, args*)
            Environment partial function to construct the environment. No additional args will be passed
    """
    def run(self):
        """
        Called by process.start(), gets the passed in args. Creates the environment, learner, and calls
        set_legal_actions() on the learner. Waits for the start command from the host then calls the run function on the
        learner.
        """
        # access thread args from http://stackoverflow.com/questions/660961/overriding-python-threading-thread-run
        pipe, learner_partial, environment_partial = self._args

        # create learner and environment
        environment = environment_partial()
        learner = learner_partial(pipe)
        learner.set_legal_actions(environment.get_legal_actions())

        # wait for start command
        pipe.recv()
        learner.run(environment)
