import json
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.nstep_a3c import NStepA3C
from reinforcepy.learners.dqn.asynchronous.a3c_thread_learner import A3CThreadLearner
from reinforcepy.learners.dqn.asynchronous.async_target_thread_host import AsyncTargetThreadHost

CONFIG = json.load(open('a3c_cfg.json'))


def main(rom_args, learner_args, network_args, num_threads, initial_learning_rate, epochs, logdir, save_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()
    network = NStepA3C(input_shape, num_actions, **network_args)

    # create thread host
    thread_host = AsyncTargetThreadHost(network, initial_learning_rate, log_dir=logdir)

    # create threads
    threads = [A3CThreadLearner(environments[t], network, thread_host.shared_dict, **learner_args) for t in range(num_threads)]

    reward_list = thread_host.run_epochs(epochs, threads, save_interval=save_interval)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    main(**CONFIG)
