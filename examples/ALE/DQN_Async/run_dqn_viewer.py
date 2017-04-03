import sys
import json
import numpy as np
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.target_dqn import TargetDQN
from reinforcepy.learners.dqn.asynchronous.q_thread_learner import QThreadLearner
import matplotlib.pyplot as plt

CONFIG = json.load(open('dqn_cfg.json'))


def main(model_path, rom_args, learner_args, network_args, num_threads, epochs, logdir, summary_interval):
    # create env
    environment = ALEEnvironment(**rom_args)

    # create network then load
    num_actions = environment.get_num_actions()
    input_shape = [learner_args['phi_length']] + environment.get_state_shape()
    network = TargetDQN(input_shape, num_actions, 'dqn', **network_args)
    network.load(model_path)

    # create threads
    del learner_args['epsilon_annealing_start']
    learner = QThreadLearner(environment, network, {}, **learner_args, epsilon_annealing_start=0.01, testing=True)

    # run 100 episodes
    reward_list = []
    try:
        for _ in range(100):
            reward = learner.run_episode(environment)
            print('Episode: {}. Steps: {}. Reward: {}'.format(_, environment.curr_step_count, reward))
            reward_list.append(reward)
    except KeyboardInterrupt:
        pass

    plt.title('Max: {0}, Mean: {1}, Min: {2}'.format(max(reward_list), np.mean(reward_list), min(reward_list)))
    plt.plot(reward_list)
    plt.show()
    return max(reward_list), np.mean(reward_list), min(reward_list)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise AttributeError('You must specify the models path as an argument')
    main(sys.argv[1], **CONFIG)
