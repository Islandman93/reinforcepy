from reinforcepy.learners.base_async import AsyncLearnerHost
import matplotlib.pyplot as plt
import numpy as np
import time


class AsyncA3CLearnerHost(AsyncLearnerHost):
    """
    This class just overrides the print_status function to correctly show policy and value loss
    """
    def print_status(self, st):
        frames = 0
        plt.clf()
        for learner_ind, learner_stat in enumerate(self.learner_stats):
            if len(learner_stat) > 0:
                frames += self.learner_frames[learner_ind]
                scores = list()
                loss = list()
                for learner in learner_stat:
                    scores.append(learner['score'])
                    loss += learner['loss']
                loss_array = np.asarray(loss)

                # score
                plt.subplot(len(self.learner_processes), 3, (learner_ind * 3) + 1)
                plt.plot(scores, '.')
                plt.ylim([0, max(scores)])
                plt.title('Score')

                # policy loss
                plt.subplot(len(self.learner_processes), 3, (learner_ind * 3) + 2)
                plt.plot(loss_array[:, 0], '.')
                plt.ylim([-np.percentile(loss_array[:, 1], 90), np.percentile(loss_array[:, 0], 90)])
                plt.title('Policy Loss')

                # value loss
                plt.subplot(len(self.learner_processes), 3, (learner_ind * 3) + 3)
                plt.plot(loss_array[:, 1], '.')
                plt.ylim([0, np.percentile(loss_array[:, 1], 90)])
                plt.title('Value Loss')
        et = time.time()
        print('==== Status Report ====')
        print('Epoch:', round(float(sum(self.learner_frames)) / 4000000, 1))  # epoch defined as 4 million frames
        print('Time:', et-st)
        print('Frames:', frames)
        print('FPS:', frames/(et-st))
        print('Best score:', self.best_score)
        print('=======================')
        plt.ion()
        plt.show()
        plt.pause(0.01)
        plt.ioff()