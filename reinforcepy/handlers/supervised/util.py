import os
import pickle
import numpy as np


# load data
def load_data(record_dir):
    file_data = {}
    for root, subdirs, files in os.walk(record_dir):
        for f in files:
            with open(os.path.join(root, f), 'rb') as in_file:
                f_data = pickle.load(in_file)
                file_data[os.path.join(root, f)] = f_data
            print('loaded', os.path.join(root, f))
    return file_data


def overwrite_with_discounted_reward(dataset, discount):
    """
        Mutates dataset to replace rewards with their discounted version
    """
    for episode, ep_data in dataset.items():
        rewards = ep_data['rewards']
        discounted_rewards = []
        # calculate discounted reward
        curr_reward = 0
        for r in reversed(rewards):
            curr_reward = r + curr_reward * discount
            discounted_rewards.append(curr_reward)
        dataset[episode]['rewards'] = discounted_rewards[::-1]


def create_rewards_from_vars(dataset, reward_fn):
    """
        Mutates dataset to replace rewards with rewards from reward_fn
    """
    for episode, ep_data in dataset.items():
        extra_state_data = ep_data['extra_state_data']
        rewards = []

        for i in range(len(extra_state_data) - 1):
            old_vars = extra_state_data[i]
            new_vars = extra_state_data[i + 1]
            rewards.append(reward_fn(new_vars, old_vars))
        dataset[episode]['rewards'] = rewards


def get_minibatch(dataset, batch_size, phi_length, sequential_batch=True, RGB=False, flatten_RGB=True):
    # dataset is list of [{'states', 'actions', 'rewards'}]
    episode_key = np.random.choice(list(dataset.keys()))
    episode = dataset[episode_key]

    if sequential_batch:
        # minibatch upper range
        mb_upper_ind = len(episode['states']) - batch_size - phi_length
        mb_ind = np.random.randint(0, mb_upper_ind)

        states = []
        actions = []
        rewards = []
        for i in range(mb_ind, mb_ind + batch_size):
            if RGB:
                s = np.asarray(episode['states'][i:i + phi_length], dtype=np.uint8)
                if not flatten_RGB:
                    # move RGB to last channel
                    s = np.transpose(s, [0, 2, 3, 1])
                else:
                    # flatten RGB and time
                    s = s.reshape((-1,) + s.shape[2:])
                states.append(s)
            else:
                states_array = np.asarray(episode['states'][i:i + phi_length], dtype=np.uint8)
                # check if we saved RGB data
                if states_array.shape[1] == 3:  # first dimension is channels
                    states.append(states_array[:, 0])
                elif states_array.shape[1] == 3:  # last dimension is channels
                    states.append(states_array[:, :, :, 0])
                # else grayscale image
                else:
                    states.append(states_array)

            actions.append(episode['actions'][i + phi_length - 1])
            rewards.append(episode['rewards'][i + phi_length - 1])

        return states, actions, rewards
    else:
        states = []
        actions = []
        rewards = []
        while len(states) < batch_size:
            # minibatch upper range
            mb_upper_ind = len(episode['states']) - phi_length
            mb_ind = np.random.randint(0, mb_upper_ind)
            if RGB:
                s = np.asarray(episode['states'][mb_ind:mb_ind + phi_length], dtype=np.uint8)
                if not flatten_RGB:
                    # move RGB to last channel
                    s = np.transpose(s, [0, 2, 3, 1])
                else:
                    # flatten RGB and time
                    s = s.reshape((-1,) + s.shape[2:])
                states.append(s)
            else:
                states_array = np.asarray(episode['states'][mb_ind:mb_ind + phi_length], dtype=np.uint8)
                # check if we saved RGB data
                if states_array.shape[1] == 3:  # first dimension is channels
                    states.append(states_array[:, 0])
                elif states_array.shape[1] == 3:  # last dimension is channels
                    states.append(states_array[:, :, :, 0])
                # else grayscale image
                else:
                    states.append(states_array)
            actions.append(episode['actions'][mb_ind + phi_length - 1])
            rewards.append(episode['rewards'][mb_ind + phi_length - 1])

        return states, actions, rewards


def evaluate_accuracy(dataset, predict_action_fn, phi_length, RGB=False, flatten_RGB=False):
    stats = {}
    true_actions = np.empty(0)
    predicted_actions = np.empty(0)
    for episode, ep_data in dataset.items():
        # iterate states into batch
        states = []
        actions = []
        for i in range(len(ep_data['states']) - phi_length):
            if RGB:
                s = np.asarray(ep_data['states'][i:i + phi_length], dtype=np.uint8)
                if not flatten_RGB:
                    # move RGB to last channel
                    s = np.transpose(s, [0, 2, 3, 1])
                else:
                    # flatten RGB and time
                    s = s.reshape((-1,) + s.shape[2:])
                states.append(s)
            else:
                states_array = np.asarray(ep_data['states'][i:i + phi_length], dtype=np.uint8)
                # check if we saved RGB data
                if states_array.shape[1] == 3:  # first dimension is channels
                    states.append(states_array[:, 0])
                elif states_array.shape[1] == 3:  # last dimension is channels
                    states.append(states_array[:, :, :, 0])
                # else grayscale image
                else:
                    states.append(states_array)
            actions.append(ep_data['actions'][i + phi_length - 1])

        actions = np.asarray(actions)
        states = np.asarray(states)
        # get predicted actions
        pred_actions_episode = predict_action_fn(states)

        # calculate number correct
        correct = np.sum(actions == pred_actions_episode)
        # append to over all episodes
        true_actions = np.concatenate((true_actions, actions), axis=0)
        predicted_actions = np.concatenate((predicted_actions, pred_actions_episode), axis=0)

        # update stats with results
        stats[episode] = {
            "correct": correct,
            "accuracy": correct / actions.shape[0]
        }
    return true_actions, predicted_actions, stats


class VisPlotter:
    def __init__(self, vis, name):
        self.vis = vis
        self.win = None
        self.name = name
        self.update_count = 0
        self.xs = []
        self.ys = []

    def line(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.update_count += 1
        if self.update_count % 20 == 0:
            if self.win is None:
                self.win = self.vis.line(np.asarray(self.ys), np.asarray(self.xs), opts={'title': self.name})
            else:
                self.vis.updateTrace(np.asarray(self.xs), np.asarray(self.ys), self.win)
            self.xs = []
            self.ys = []
