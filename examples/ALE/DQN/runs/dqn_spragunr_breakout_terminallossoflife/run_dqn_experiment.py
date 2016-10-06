from reinforcepy.environments.ALE import ALEEnvironment
from reinforcepy.learners.Deepmind import DQNLearner
from reinforcepy.networks.Deepmind import DQN_NIPS
import time


def load_config():
    # load config from json
    from reinforcepy.handlers import Parameters

    # function to fix rom string to byte string
    def str2byte(string):
        return string.encode()

    rom_fix = {'experiment_parameters': {'rom': str2byte}}

    parameters = Parameters.fromJSON('dqn_cfg.json', rom_fix)
    return [parameters['network_parameters'], parameters['training_parameters'], parameters['learner_parameters'],
            parameters['experiment_parameters']]


def main(experiment_parameters):
    experiment_parameters.required(['epochs', 'save_interval', 'rom'])

    # setup vars
    EPOCH_DEF = 50000  # an epoch is defined as 50,000 steps in the NIPS paper

    # load parameters
    network_parameters, training_parameters, learner_parameters, _ = load_config()

    # initialize environment and network/learner
    environment = ALEEnvironment(experiment_parameters.get('rom'), loss_of_life_termination=True)
    network = DQN_NIPS(network_parameters, training_parameters)
    learner = DQNLearner(learner_parameters, network)
    learner.set_legal_actions(environment.get_legal_actions())

    # main loop to run episodes until enough epochs have been reached
    # saves every save_interval
    ep_count = 0
    reward_list = list()
    st = time.time()
    try:
        while learner.step_count < experiment_parameters.get('epochs') * EPOCH_DEF:
            reward = learner.run_episode(environment)
            reward_list.append(reward)
            print("Episode finished", "Reward:", reward, "SPS:", learner.step_count/(time.time() - st), learner.get_status())
            if experiment_parameters.get('save_interval') is not None:
                if learner.step_count > ep_count * EPOCH_DEF:
                    # save parameters
                    learner.save("dqn_{0:.2f}.pkl".format(ep_count))
                    ep_count += experiment_parameters.get('save_interval')

        print("Done, Total Time:", time.time()-st)
    except KeyboardInterrupt:
        print("KeyboardInterrupt total time:", time.time()-st)

    return reward_list, learner, ep_count

if __name__ == '__main__':
    _, _, _, experiment_parameters = load_config()
    reward_list, learner, epoch_count = main(experiment_parameters)

    # save learner
    learner.save("dqn_{0:.2f}.pkl".format(epoch_count))

    print('Max score {0}. {1:.2f} epochs'.format(max(reward_list), epoch_count))
    import matplotlib.pyplot as plt
    plt.plot(reward_list)
    plt.show()
