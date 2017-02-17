import sys
import json
import datetime
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.target_dqn import TargetDQN
from reinforcepy.learners.dqn.asynchronous.exp_replay_q_thread_learner import ExpQThreadLearner
from reinforcepy.learners.dqn.asynchronous.async_thread_host import AsyncThreadHost


def main(rom_args, learner_args, network_args, algorithm_type, num_threads, epochs, logdir, save_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()
    network = TargetDQN(input_shape, num_actions, algorithm_type, **network_args)

    # create thread host
    thread_host = AsyncThreadHost(network, log_dir=logdir)

    # create threads
    threads = [ExpQThreadLearner(environments[t], network, thread_host.shared_dict, dataset_size=100000//num_threads, batch_size=32, **learner_args) for t in range(num_threads)]

    reward_list = thread_host.run_epochs(epochs, threads, save_interval=save_interval)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    run_type = 'dqn'
    if len(sys.argv) >= 2:
        if sys.argv[1] not in ['dqn', 'double', 'nstep', 'doublenstep']:
            raise ValueError('The algorithm type must be passed as a parameter and must be either dqn, double, or nstep')
        run_type = sys.argv[1]

    CONFIG = json.load(open('dqn_cfg.json'))
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    CONFIG['logdir'] += run_type + '_' + run_date + '/'
    main(**CONFIG, algorithm_type=run_type)
