import json
import datetime
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.nstep_a3c_lstm import NStepA3CLSTM
from reinforcepy.learners.dqn.asynchronous.recurrent_thread_learner import RecurrentThreadLearner
from reinforcepy.learners.dqn.asynchronous.async_thread_host import AsyncThreadHost


def main(rom_args, learner_args, network_args, num_threads, epochs, logdir, save_interval):
    # create envs for each thread
    environments = [ALEEnvironment(**rom_args) for _ in range(num_threads)]

    # create shared network
    num_actions = environments[0].get_num_actions()
    input_shape = [learner_args['phi_length']] + environments[0].get_state_shape()
    network = NStepA3CLSTM(input_shape, num_actions, **network_args)

    # create thread host
    thread_host = AsyncThreadHost(network, log_dir=logdir)

    # create threads
    threads = [RecurrentThreadLearner(environments[t], network, thread_host.shared_dict, **learner_args) for t in range(num_threads)]

    reward_list = thread_host.run_epochs(epochs, threads, save_interval=save_interval)

    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
    plt.savefig(logdir + 'rewards.png')
    plt.show()
    return max([x[0] for x in reward_list])


if __name__ == '__main__':
    CONFIG = json.load(open('a3c_cfg.json'))
    run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    CONFIG['logdir'] += 'lstm_' + run_date + '/'
    main(**CONFIG)
