from reinforcepy.learners.base_async import AsyncLearnerHost
from reinforcepy.learners.base_async import PipeCmds
import time
from multiprocessing import Manager


class NoveltyHost(AsyncLearnerHost):
    def __init__(self, host_cnn, learners, environment_partials):
        self.novel_states = Manager().dict()
        super().__init__(host_cnn, learners, environment_partials)

    def process_pipe(self, learner_ind, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradientsSteps:
            self.cnn.gradient_step(extras[0])
            self.learner_frames[learner_ind] = extras[1]
            # self.process_novel_frames(extras[2])

            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters,
                       (self.cnn.get_parameters(), {'counter': sum(self.learner_frames)})))
        if pipe_cmd == PipeCmds.ClientSendingStats:
            self.learner_stats[learner_ind].append(extras)
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
        print('Novel frames:', len(self.novel_states))
        print('=======================')