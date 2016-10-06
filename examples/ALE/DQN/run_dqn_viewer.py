import time
import sys
from reinforcepy.environments.ALE import ALEEnvironment
from reinforcepy.learners.dqn import DQNLearner


def load_config():
    # load config from json
    from reinforcepy.handlers import Parameters
    parameters = Parameters.fromJSON('dqn_cfg.json')
    return [parameters['network_parameters'], parameters['training_parameters'], parameters['learner_parameters'],
            parameters['experiment_parameters']]


def main(model_path, experiment_parameters):
    experiment_parameters.required(['rom'])

    # load parameters
    network_parameters, training_parameters, learner_parameters, _ = load_config()

    # decide which network to import based on backend parameter
    if network_parameters.get('backend') == 'tensorflow':
        from reinforcepy.networks.dqn.tflow.dqn_nips import DQN_NIPS
        network = DQN_NIPS(network_parameters, training_parameters)
        network.load(model_path)
    else:
        from reinforcepy.networks.dqn.theanolasagne.dqn_nips import DQN_NIPS
        network = DQN_NIPS(network_parameters, training_parameters)
        network.load(model_path)

    # initialize environment and network/learner
    environment = ALEEnvironment(experiment_parameters.get('rom'), loss_of_life_termination=False, show_rom=True)
    network = DQN_NIPS(network_parameters, training_parameters)

    # set learner parameters egreedy policy to 0.01 for testing
    learner_parameters.set('egreedy_policy', 0.01)
    learner_parameters.set('max_dataset_size', 4)  # only need enough past frames for the cnn input

    learner = DQNLearner(learner_parameters, network, testing=True)
    learner.set_action_num(environment.get_num_actions())

    # main loop to run episodes
    reward_list = list()
    st = time.time()
    try:
        for episode in range(10):
            reward = learner.run_episode(environment)
            reward_list.append(reward)
            print("Episode finished", "Reward:", reward, "SPS:", learner.step_count/(time.time() - st), learner.get_status())

        print("Done, Total Time:", time.time()-st)
    except KeyboardInterrupt:
        print("KeyboardInterrupt total time:", time.time()-st)

    return reward_list, learner

if __name__ == '__main__':
    if len(sys.argv) == 0:
        raise AttributeError('You must pass the path to the saved network as an argument')
    _, _, _, experiment_parameters = load_config()
    reward_list, learner = main(sys.argv[1], experiment_parameters)

    print('Max score {0}s'.format(max(reward_list)))
    import matplotlib.pyplot as plt
    plt.plot(reward_list)
    plt.show()
