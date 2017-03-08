import json
import datetime
from functools import partial
from reinforcepy.environments import ALEEnvironment
from reinforcepy.learners.dqn.asynchronous.ptorch.q_process_learner import QProcessLearner
from reinforcepy.learners.dqn.asynchronous.ptorch.async_process_host import AsyncProcessHost
from reinforcepy.networks.dqn.ptorch.nstep_a3c import A3CModel


def main(rom_args, learner_args, network_args, num_threads, epochs, logdir, save_interval):
    # create envs for each thread
    environments = [partial(ALEEnvironment, **rom_args) for _ in range(num_threads)]

    # create shared network
    network = A3CModel()
    network.share_memory()

    # create thread host
    thread_host = AsyncProcessHost(network, log_dir=logdir)

    # create threads
    threads = [QProcessLearner(environments[t], thread_host.shared_dict, **learner_args) for t in range(num_threads)]

    reward_list = thread_host.run_epochs(epochs, threads, save_interval=save_interval)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])

if __name__ == '__main__':
    CONFIG = json.load(open('a3c_cfg.json'))
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    CONFIG['logdir'] += '_' + run_date + '/'
    main(**CONFIG)
