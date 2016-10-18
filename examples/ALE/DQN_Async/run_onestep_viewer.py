import sys
import numpy as np
import json
from reinforcepy.environments import ALEEnvironment
from reinforcepy.networks.dqn.tflow.one_step_target_dqn import OneStepTargetDQN
from reinforcepy.learners.dqn.asynchronous.onestep_dqn_thread_learner import OneStepDQNThreadLearner

CONFIG = json.load(open('onestep_cfg.json'))


def main(model_path, rom_args, learner_args, network_args, num_threads, initial_learning_rate, epochs, logdir, save_interval):
    # create env
    environment = ALEEnvironment(**rom_args)

    # create network then load
    num_actions = environment.get_num_actions()
    input_shape = [learner_args['phi_length']] + environment.get_state_shape()
    network = OneStepTargetDQN(input_shape, num_actions, **network_args)
    network.load(model_path)

    # create threads
    del learner_args['epsilon_annealing_start']
    learner = OneStepDQNThreadLearner(environment, network, {}, **learner_args, epsilon_annealing_start=0.01, testing=True)

    # run 100 episodes
    reward_list = []
    try:
        for _ in range(100):
            reward = learner.run_episode(environment)
            print('Episode: {0}. Reward:'.format(_), reward)
            reward_list.append(reward)
    except KeyboardInterrupt:
        pass

    import matplotlib.pyplot as plt
    plt.title('Max: {0}, Mean: {1}, Min: {2}'.format(max(reward_list), np.mean(reward_list), min(reward_list)))
    plt.plot(reward_list)
    plt.show()
    return max(reward_list), np.mean(reward_list), min(reward_list)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise AttributeError('You must specify the models path as an argument')
    main(sys.argv[1], **CONFIG)
