import time
import numpy as np


class AsyncHost:
    def __init__(self, async_handler):
        self.async_handler = async_handler

    def run_epochs(self, num_epochs, threaded_learners, EPOCH_DEF=1000000, summary_interval=1, thread_delay=0):
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
            time.sleep(thread_delay)

        last_summary = 0
        try:
            while self.async_handler.global_step < num_epochs * EPOCH_DEF:
                global_step = self.async_handler.global_step
                current_epoch = global_step / EPOCH_DEF
                if current_epoch > last_summary + summary_interval or last_summary == 0:
                    et = time.time()
                    print('===================')
                    print('Current Step/Epoch: {0}/{1:.2f}'.format(global_step, current_epoch))
                    print('SPS:', global_step / (et - st))
                    reward_list = self.async_handler.rewards
                    if len(reward_list) > 0:
                        rewards = [x[0] for x in reward_list]
                        print('Max Reward:', np.max(rewards))
                        print('Avg Reward:', np.mean(rewards))
                    print('===================')
                    last_summary = current_epoch
                time.sleep(1)
        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')

        self.async_handler.done = True
        for t in threads:
            t.join()
        print('All threads stopped')
        return self.async_handler.rewards
