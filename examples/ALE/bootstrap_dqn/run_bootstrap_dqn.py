import sys
import json
import datetime
import matplotlib.pyplot as plt
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.bootstrap_dqn import BootstrapTargetDQN
# import reinforcepy.networks.util.tflow_util as tf_util
from reinforcepy.learners.dqn.bootstrap import BootstrapQThreadLearner
from reinforcepy.handlers.async_thread_host import AsyncThreadHost


def main(rom_args, learner_args, network_args, algorithm_type, num_threads, num_bootstraps, epochs, logdir, summary_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()

    # dueling we have to specify a different network generator for value and advantage output
    # if 'dueling' in algorithm_type:
    #     network = TargetDQN(input_shape, num_actions, algorithm_type, network_generator=tf_util.create_dueling_nips_network,
    #                         log_dir=logdir, **network_args)
    # else:
    network = BootstrapTargetDQN(input_shape, num_actions, algorithm_type, num_bootstraps, log_dir=logdir, **network_args)

    # create thread host
    thread_host = AsyncThreadHost()

    # create threads
    threads = [BootstrapQThreadLearner(num_bootstraps, environments[t], network, thread_host.shared_dict, **learner_args) for t in range(num_threads)]

    reward_list = thread_host.run_epochs(epochs, threads, summary_interval=summary_interval)

    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    # the paper defaults to the double dqn update
    run_type = 'double'
    if len(sys.argv) >= 2:
        algorithm_types = ['dqn', 'double', 'dueling']
        if sys.argv[1] not in algorithm_types:
            raise ValueError('The algorithm type must be passed as a parameter and must be in {}'.format(algorithm_types))
        run_type = sys.argv[1]

    CONFIG = json.load(open('dqn_cfg.json'))
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    CONFIG['logdir'] += 'bootstrap-' + run_type + '_' + run_date + '/'
    main(**CONFIG, algorithm_type=run_type)
