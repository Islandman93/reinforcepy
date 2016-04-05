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

    parameters = Parameters.fromJSON('onestep_dqn_cfg.json', rom_fix)
    return [parameters['network_parameters'], parameters['training_parameters'], parameters['learner_parameters']]


def main(epochs, rom):
    # setup vars
    ep_mod = 0.5
    epoch_def = 50000

    # load parameters
    network_parameters, training_parameters, learner_parameters = load_config()

    # initialize environment and network/learner
    environment = ALEEnvironment(rom)
    network = DQN_NIPS(network_parameters, training_parameters)
    learner = DQNLearner(learner_parameters, network)
    learner.set_legal_actions(environment.get_legal_actions())
    
    # main loop
    ep_count = 0
    reward_list = list()
    st = time.time()
    while ep_count < epochs:
        reward = learner.run_episode(environment)
        reward_list.append(reward)
        print("Episode finished", "Reward:", reward, "SPS:", learner.step_count/(time.time() - st), learner.get_status())
        if learner.step_count > epoch_def * ep_count:
            # save parameters
            learner.save("dqn_{0}.pkl".format(ep_count))
            ep_count += ep_mod
    
    print("Done")
