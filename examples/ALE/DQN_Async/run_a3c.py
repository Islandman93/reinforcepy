import os
from shutil import copy2
import json
import datetime
import pickle
import threading
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.nstep_a3c import NStepA3C
from reinforcepy.learners.dqn.asynchronous import AsyncQLearner
from reinforcepy.handlers.asynchronous import AsyncHost, MTAsyncHandler


def main(rom_args, learner_args, network_args, num_threads, epochs, logdir, summary_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()
    network = NStepA3C(input_shape, num_actions, log_dir=logdir, **network_args)

    # create thread host and async handlers
    async_handler = MTAsyncHandler()
    async_host = AsyncHost(async_handler)

    # create threads
    threads = []
    for t in range(num_threads):
        learner = AsyncQLearner(environments[t], network, async_handler, **learner_args)
        thread = threading.Thread(target=learner.run)
        threads.append(thread)

    reward_list = async_host.run_epochs(epochs, threads, summary_interval=summary_interval)

    with open(logdir + 'rewards.pkl', 'wb') as out_file:
        pickle.dump(reward_list, out_file)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    CONFIG = json.load(open('a3c_cfg.json'))
    # add date time to log dir
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    logdir = CONFIG['logdir'] + '_' + run_date + '/'
    CONFIG['logdir'] = logdir
    # make log dir and copy config
    os.makedirs(logdir)
    copy2('a3c_cfg.json', logdir)
    main(**CONFIG)
