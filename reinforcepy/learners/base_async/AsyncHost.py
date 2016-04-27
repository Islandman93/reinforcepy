from multiprocessing import Pipe
from reinforcepy.learners.base_async.PipeCmds import PipeCmds
import time
import pickle


class AsyncLearnerHost:
    """
    The :class:`AsyncLearnerHost` class is used to be able to run multiple learners on emulator instances, each
    learner will have it's own emulator instance. It should be used for learners that cannot be used outside the thread
    that spawns them (Theano). If your learner can be used outside of the spawning thread (everything else but Theano)
    use :class:`ThreadedGameHandler`.

    Parameters
    ----------
    host_network : NeuralNetwork
        The host acts as the shared parameters for the neural network used in all threads. It should be the
        same class of neural network used in the learner.
    learners : Iterable<tuple(partial(learner_constructor, args), LearnerProcessClass)>
        The learner classes to create with their respective args
    environment_partials : Iterable<:class:`BaseEnvironment`>
        The environments for the learners. should be matched one to one
    """
    def __init__(self, host_network, learners, environment_partials):
        # init host network
        self.network = host_network
        # setup learners and emulators
        self.learner_pipes = list()
        self.learner_processes = list()
        self.learner_frames = list()
        self.learner_stats = list()
        self.logs = list()
        for ind, learner_process in enumerate(learners):
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_partial, process_class = learner_process
            process = process_class(args=(child_conn, learner_partial, environment_partials[ind]), daemon=True)
            process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(process)
            self.learner_frames.append(0)

        self.best_score = 0

    def run(self, epochs=1, save_interval=1, show_status=True):
        ep_count = 0
        for learner in self.learner_pipes:
            learner.send(PipeCmds.Start)

        st = time.time()
        while sum(self.learner_frames) < epochs * 4000000:  # 4000000 frames is defined as an epoch
            for learner_ind, learner in enumerate(self.learner_pipes):
                if learner.poll():
                    self.process_pipe(learner_ind, learner)

            if sum(self.learner_frames) >= ep_count * 4000000 and save_interval is not None:
                ep_count += save_interval

                # save network parms
                with open('async_network_parameters{0}.pkl'.format(sum(self.learner_frames)), 'wb') as out_file:
                    pickle.dump(self.network.get_parameters(), out_file)

                if show_status:
                    self.print_status(st)

    def process_pipe(self, learner_ind, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradientsSteps:
            self.network.gradient_step(extras[0])
            self.learner_frames[learner_ind] = extras[1]
            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters,
                       (self.network.get_parameters(), {'counter': sum(self.learner_frames)})))
        if pipe_cmd == PipeCmds.ClientSendingStats:
            if extras['score'] > self.best_score:
                self.best_score = extras['score']

    def print_status(self, st):
        et = time.time()
        print('==== Status Report ====')
        print('Epoch:', round(float(sum(self.learner_frames)) / 4000000, 2))  # 4000000 frames is defined as an epoch
        print('Time:', et-st)
        print('Frames:', sum(self.learner_frames))
        print('FPS:', sum(self.learner_frames)/(et-st))
        print('Best score:', self.best_score)
        print('=======================')

    def block_until_done(self):
        self.end_processes()
        for learner in self.learner_processes:
            if not learner.join(1):
                print("Can't join learner", learner)

    def end_processes(self):
        for learner in self.learner_pipes:
            # try to recieve any sent message
            if learner.poll(0.1):
                _ = learner.recv()
            learner.send((PipeCmds.End, None))
