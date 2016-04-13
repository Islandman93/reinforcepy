from reinforcepy.networks.Deepmind import AsyncTargetCNNNstep
from reinforcepy.viewers import DQNViewer
from reinforcepy.environments.ALE import ALEEnvironment
import pickle
from reinforcepy.handlers import Parameters


def main():
    # run vars
    rom = b'D:\\_code\\breakout.bin'
    network_p = {"input_shape": [1, 4, 84, 84], "output_num": 4, "stride": [4, 2], "untie_biases": True}
    train_p = {"learning_rate": 0.0001, "epsilon": 1e-6, "training_size": 5}
    viewer_p = {"skip_frame": 4, "phi_length": 4, "egreedy_val": 0.1}
    cnn_parms_file = 'async_network_parameters0.pkl'

    # setup parameters
    network_parms = Parameters('Network', network_p)
    train_parms = Parameters('Train', train_p)
    viewer_parms = Parameters('Viewer', viewer_p)

    # load cnn parms
    with open(cnn_parms_file, 'rb') as in_file:
        cnn_params = pickle.load(in_file)
    network_parms.set('initial_values', cnn_params)

    # create network
    cnn = AsyncTargetCNNNstep(network_parms, train_parms)

    # create viewer
    dqn_viewer = DQNViewer(viewer_parms, cnn)

    # create environment
    env = ALEEnvironment(rom, show_rom=True)
    dqn_viewer.set_legal_actions(env.get_legal_actions())

    # run episode
    dqn_viewer.run_episode(env)

if __name__ == '__main__':
    main()
