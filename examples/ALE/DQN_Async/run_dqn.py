import sys
import json
import datetime
import pickle
import threading
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.target_dqn import TargetDQN
import reinforcepy.networks.util.tflow_util as tf_util
from reinforcepy.learners.dqn.asynchronous import AsyncQLearner
from reinforcepy.handlers.asynchronous import AsyncHost, MTAsyncHandler


def main(rom_args, learner_args, network_args, algorithm_type, num_threads, epochs, logdir, summary_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()

    # dueling we have to specify a different network generator for value and advantage output
    if 'dueling' in algorithm_type:
        network = TargetDQN(input_shape, num_actions, algorithm_type, network_generator=tf_util.create_dueling_nips_network,
                            log_dir=logdir, **network_args)
    else:
        network = TargetDQN(input_shape, num_actions, algorithm_type, log_dir=logdir, **network_args)

    # create thread host and async handlers
    async_handler = MTAsyncHandler()
    thread_host = AsyncHost(async_handler)

    # create threads
    threads = []
    for t in range(num_threads):
        learner = AsyncQLearner(environments[t], network, async_handler, **learner_args)
        thread = threading.Thread(target=learner.run)
        threads.append(thread)

    reward_list = thread_host.run_epochs(epochs, threads, summary_interval=summary_interval)

    with open(logdir + 'rewards.pkl', 'wb') as out_file:
        pickle.dump(reward_list, out_file)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    run_type = 'dqn'
    if len(sys.argv) >= 2:
        algorithm_types = ['dqn', 'double', 'dueling', 'nstep', 'doublenstep', 'duelingnstep']
        if sys.argv[1] not in algorithm_types:
            raise ValueError('The algorithm type must be passed as a parameter and must be in {}'.format(algorithm_types))
        run_type = sys.argv[1]

    CONFIG = json.load(open('dqn_cfg.json'))
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    CONFIG['logdir'] += run_type + '_' + run_date + '/'
    main(**CONFIG, algorithm_type=run_type)
