from reinforcepy.networks.dqn.tflow.nstep_a3c import NStepA3C
import visdom
from reinforcepy.handlers.supervised.util import load_data, overwrite_with_discounted_reward, get_minibatch, VisPlotter, evaluate_accuracy
from sklearn.metrics import confusion_matrix
import numpy as np
from pprint import pprint


def main(data_dir, num_actions, num_steps=2000, discount=0.99, phi_length=4, sequential_batch=False, RGB=False,
         batch_size=32, critic_weight=1.0, actor_weight=1.0, actor_cce=True, actor_cce_weight=1.0):
    # dataset loading and wrangling
    dataset = load_data(data_dir)
    overwrite_with_discounted_reward(dataset, discount)

    # create network, get state shape
    s, _, _ = get_minibatch(dataset, 1, phi_length, RGB=RGB)
    state_shape = list(s[0].shape)

    # num outputs
    network = NStepA3C(state_shape, num_actions, initial_learning_rate=0.0003,
                       critic_weight=critic_weight, actor_weight=actor_weight, actor_cce=actor_cce, actor_cce_weight=actor_cce_weight,
                       entropy_regularization=0.01, supervised_training=True, mean_grads=True, log_dir='./supervised-model/')

    # Create visdom plotters
    vis = visdom.Visdom()
    tl, cl, al, ae = VisPlotter(vis, 'Total Loss'), VisPlotter(vis, 'Critic Loss'), VisPlotter(vis, 'Actor Loss'), VisPlotter(vis, 'Actor Entropy')
    ac = VisPlotter(vis, 'Actor CCE')

    try:
        for mb_ind in range(num_steps):
            # get minibatch
            states, actions, rewards = get_minibatch(dataset, batch_size, phi_length, sequential_batch=False, RGB=RGB)
            # get losses
            total_loss, critic_loss, actor_loss, actor_entropy, actor_cce = network.supervised_train_step(states, actions, rewards)

            # plot losses
            tl.line(mb_ind, total_loss)
            cl.line(mb_ind, critic_loss)
            al.line(mb_ind, actor_loss)
            ae.line(mb_ind, actor_entropy)
            ac.line(mb_ind, actor_cce)

    except KeyboardInterrupt:
        pass

    network.save(global_step=mb_ind)
    true_actions, predicted_actions, stats = evaluate_accuracy(dataset, network.get_output, phi_length)
    pprint(stats)
    pprint(confusion_matrix(true_actions, predicted_actions))
    pprint(np.unique(true_actions))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Top level directory where recorded episodes are stored')
    parser.add_argument('num_actions', type=int, help='The number of actions in the dataset')
    parser.add_argument('-num_steps', type=int, help='Number of supervised training steps', default=2000)
    args = parser.parse_args()
    main(args.data_dir, args.num_actions, num_steps=args.num_steps)
