import time
import numpy as np
from multiprocessing import Manager


class AsyncProcessHost:
    def __init__(self, network, log_dir):
        self.learning_rate_annealer = None
        self.thread_fns = None
        self.log_dir = log_dir

        self.network = network
        self.summary_writer = None
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.shared_dict["counter"] = 0
        self.shared_dict["done"] = False
        self.shared_dict["reward_list"] = self.manager.list()
        self.shared_dict["network"] = network

    def run_epochs(self, num_epochs, threaded_learners, EPOCH_DEF=1000000, save_interval=1):
        """
        Args:
            save_interval: The interval in epochs to save the network, print run stats, and send summaries to tensorboard.
                Default 1. Can be set at 0 to output every step (constrained by the sleep time)

        """
        threads = []
        st = time.time()
        for thread in threaded_learners:
            thread.start()
            threads.append(thread)

        last_save = 0
        try:
            while self.shared_dict['counter'] < num_epochs * EPOCH_DEF:
                current_epoch = self.shared_dict['counter'] / EPOCH_DEF
                if current_epoch > last_save + save_interval or last_save == 0:
                    et = time.time()
                    print('===================')
                    print('Current Step/Epoch: {0}/{1:.2f}'.format(self.shared_dict['counter'], current_epoch))
                    print('SPS:', self.shared_dict['counter'] / (et - st))
                    if len(self.shared_dict['reward_list']) > 0:
                        print('Max Reward:', np.max([x[0] for x in self.shared_dict['reward_list']]))
                        print('Avg Reward:', np.mean([x[0] for x in self.shared_dict['reward_list']]))
                    # print('Learning Rate:', self.network.current_learning_rate)
                    print('===================')

                    # write summaries next step
                    # self.shared_dict['write_summaries_this_step'] = True

                    # self.network.save(self.log_dir + 'model', global_step=self.shared_dict['counter'])
                    last_save = current_epoch
                time.sleep(1)
        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')

        self.shared_dict['done'] = True
        for t in threads:
            t.join()
        print('All threads stopped')
        return self.shared_dict['reward_list']

    def run_no_mp_epochs(self, num_epochs, threaded_learners, EPOCH_DEF=1000000, save_interval=1):
        """
        Args:
            save_interval: The interval in epochs to save the network, print run stats, and send summaries to tensorboard.
                Default 1. Can be set at 0 to output every step (constrained by the sleep time)

        """
        thread = threaded_learners[0]
        thread.run()
        print('All threads stopped')
        return self.shared_dict['reward_list']
