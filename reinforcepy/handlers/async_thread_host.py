import time
import numpy as np


class AsyncThreadHost:
    def __init__(self):
        self.reward_list = []

        def output_reward(r):
            self.reward_list.append([r, self.shared_dict['counter']])

        self.shared_dict = {"counter": 0, "done": False, "add_reward": output_reward}

    def run_epochs(self, num_epochs, threaded_learners, EPOCH_DEF=1000000, summary_interval=1):
        """
        Args:
            summary_interval: The interval in epochs to print run stats
                Default 1. Can be set at 0 to output every step (constrained by the sleep time)

        """
        threads = []
        st = time.time()
        for thread in threaded_learners:
            thread.start()
            threads.append(thread)

        last_summary = 0
        try:
            while self.shared_dict['counter'] < num_epochs * EPOCH_DEF:
                current_epoch = self.shared_dict['counter'] / EPOCH_DEF
                if current_epoch > last_summary + summary_interval or last_summary == 0:
                    et = time.time()
                    print('===================')
                    print('Current Step/Epoch: {0}/{1:.2f}'.format(self.shared_dict['counter'], current_epoch))
                    print('SPS:', self.shared_dict['counter'] / (et - st))
                    if len(self.reward_list) > 0:
                        rewards = [x[0] for x in self.reward_list]
                        print('Max Reward:', np.max(rewards))
                        print('Avg Reward:', np.mean(rewards))
                    print('===================')
                    last_summary = current_epoch
                time.sleep(1)
        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')

        self.shared_dict['done'] = True
        for t in threads:
            t.join()
        print('All threads stopped')
        return self.reward_list
