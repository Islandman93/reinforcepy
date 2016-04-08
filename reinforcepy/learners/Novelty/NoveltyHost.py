from reinforcepy.learners.base_async import AsyncLearnerHost
import time


class NoveltyHost(AsyncLearnerHost):
    def __init__(self, host_cnn, novel_states, learners, environment_partials):
        self.novel_states = novel_states
        super().__init__(host_cnn, learners, environment_partials)

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
