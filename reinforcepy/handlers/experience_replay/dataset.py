"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
Original author: https://github.com/spragunr/deep_q_rl
"""

import numpy as np
import time

floatX = 'float32'


class DataSet():
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, width, height, rng=np.random.RandomState(), max_steps=1000, phi_length=4):
        """Construct a DataSet.

        Arguments:
            width, height - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((max_steps, height, width), dtype='uint8')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=0, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    def last_batch(self, batch_size, add_state_tp1_img: np.array=None):
        """Return the most recent batch of data, with img optionally added to states_tp1"""
        # if adding an additional frame at the end, make arange inclusive with self.top + 1
        if add_state_tp1_img is not None:
            indexes = np.arange(self.top - batch_size - self.phi_length + 1, self.top + 1)
        else:
            indexes = np.arange(self.top - batch_size - self.phi_length, self.top + 1)

        # Allocate the response.
        imgs = np.empty((batch_size,
                         self.phi_length + 1,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.empty((batch_size), dtype='int32')
        rewards = np.empty((batch_size), dtype=floatX)
        terminals = np.empty((batch_size), dtype=bool)

        # this code is weird below because we are looking 1 into the past if no add_state_tp1_img (ie last state added was terminal)
        for b in range(batch_size):
            # last one in batch add in image
            if b == batch_size - 1 and add_state_tp1_img is not None:
                tmp_last_sample = np.empty((self.phi_length + 1, self.height, self.width), dtype='uint8')
                tmp_last_sample[0:self.phi_length] = self.imgs.take(indexes[b:b + self.phi_length], axis=0, mode='wrap')
                tmp_last_sample[self.phi_length] = add_state_tp1_img
                imgs[b] = tmp_last_sample
            else:
                imgs[b] = self.imgs.take(indexes[b:b + self.phi_length + 1], axis=0, mode='wrap')

            end_index = b + self.phi_length - 1
            actions[b] = self.actions.take(indexes[end_index], mode='wrap')
            rewards[b] = self.rewards.take(indexes[end_index], mode='wrap')
            if add_state_tp1_img is not None:
                terminals[b] = self.terminal.take(indexes[end_index], mode='wrap')
            else:
                terminals[b] = self.terminal.take(indexes[end_index + 1], mode='wrap')

        return imgs[:, 0:-1], actions, rewards, imgs[:, 1:], terminals

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
           next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        imgs = np.empty((batch_size,
                         self.phi_length + 1,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.empty((batch_size), dtype='int32')
        rewards = np.empty((batch_size), dtype=floatX)
        terminal = np.empty((batch_size), dtype='bool')

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            # Both the before and after states contain phi_length
            # frames, overlapping except for the first and last.
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
                continue

            # Add the state transition to the response.
            imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            count += 1

        return imgs[:, 0:-1], actions, rewards, imgs[:, 1:], terminal

    def random_sequential_batch(self, batch_size, max_tries=100):
        """Return corresponding states, actions, rewards, terminal status, and
           next_states for batch_size a random sequential set of state transitions.

        """
        all_indices, terminals = self._get_random_sequential_indices(batch_size, max_tries)
        if all_indices is None:
            return None, None, None, None, None

        # if requested size is greater than our size
        # Allocate the response.
        imgs = np.empty((batch_size,
                         self.phi_length + 1,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.empty((batch_size), dtype='int32')
        rewards = np.empty((batch_size), dtype=floatX)

        # found a good start ind create sequential batch
        for b in range(batch_size):
            # NOTE: axis 0 is required here
            imgs[b] = self.imgs.take(all_indices[b:b + self.phi_length + 1], axis=0, mode='wrap')
            end_index = b + self.phi_length - 1
            actions[b] = self.actions.take(all_indices[end_index], mode='wrap')
            rewards[b] = self.rewards.take(all_indices[end_index], mode='wrap')

        return imgs[:, 0:-1], actions, rewards, imgs[:, 1:], terminals[-batch_size:]

    def reward_prediction_prioritized_sample(self, sample_size, reward_probability=0.5, max_tries=100):
        # check if we should return a reward
        return_reward = np.random.uniform() < reward_probability
        if return_reward:
            inds = np.where(self.rewards != 0)[0]
            # if no rewards, return none
            if inds.size == 0:
                return None, None
            curr_tries = 0
            while curr_tries < max_tries:
                random_ind = np.random.choice(inds, 1)
                # we make the reward the last thing according to paper, + 1 because arange is exclusive
                all_indices = np.arange(random_ind-sample_size-self.phi_length+1, random_ind+1)
                terminals = self.terminal.take(all_indices, mode='wrap')
                # if no terminals before last one return
                if not np.any(terminals[0:-1]):
                    all_indices + self.bottom
                    break
                else:
                    all_indices = None
                curr_tries += 1
        else:
            all_indices, terminals = self._get_random_sequential_indices(sample_size, max_tries, conditions=lambda x: self.__reward_in_indices(x) == False)
        if all_indices is None:
            return None, None

        # Allocate the response.
        imgs = np.empty((sample_size,
                         self.phi_length,
                         self.height,
                         self.width),
                        dtype='uint8')
        # found a good start ind create sequential batch
        for b in range(sample_size):
            # NOTE: axis 0 is required here
            imgs[b] = self.imgs.take(all_indices[b:b + self.phi_length], axis=0, mode='wrap')

        # we want to make sure rewards is still an array
        return imgs, self.rewards.take(all_indices[np.newaxis, -1], mode='wrap')

    def _get_random_sequential_indices(self, batch_size, max_tries, conditions=lambda x: True):
        """ Finds a set of sequential indices that does not have a terminal except (possibly) at the end
            Returns None if max_tries reached
            Else Returns indices, terminals
        """
        if self.size < batch_size + self.phi_length:
            return None, None, None, None, None
        curr_tries = 0
        while curr_tries < max_tries:
            # Randomly choose a time step from the replay memory.
            # Since this will be the bottom index it must be batch_size*phi_length from the end
            # NOTE: randint high is exclusive so we don't need to subtract 1
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size -
                                     batch_size - self.phi_length)

            all_indices = np.arange(index, index + batch_size + self.phi_length)
            terminals = self.terminal.take(all_indices, mode='wrap')
            # if no terminals before last one return
            if not np.any(terminals[0:-1]) and conditions(all_indices):
                return all_indices, terminals
            curr_tries += 1
        # didn't find anything, max_tries reached
        return None, None

    def __reward_in_indices(self, indices):
        return np.any(self.rewards.take(indices, mode='wrap'))


# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = DataSet(width=2, height=3,
                      rng=np.random.RandomState(42),
                      max_steps=6, phi_length=4)
    for i in range(10):
        img = np.random.randint(0, 256, size=(3, 2))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print('img', img)
        dataset.add_sample(img, action, reward, terminal)
        print("I", dataset.imgs)
        print("A", dataset.actions)
        print("R", dataset.rewards)
        print("T", dataset.terminal)
        print("SIZE", dataset.size)
        print()
    print("LAST PHI", dataset.last_phi())
    print()
    print('BATCH', dataset.random_batch(1))


def speed_tests(case='random'):

    dataset = DataSet(width=80, height=80,
                      rng=np.random.RandomState(42),
                      max_steps=20000, phi_length=4)

    img = np.random.randint(0, 256, size=(80, 80))
    action = np.random.randint(16)
    start = time.time()
    for i in range(100000):
        terminal = False
        reward = 0
        if np.random.random() < .05:
            terminal = True
        if np.random.random() < .05:
            reward = np.random.uniform(-1, 1)
        dataset.add_sample(img, action, reward, terminal)
    print("samples per second: ", 100000 / (time.time() - start))

    start = time.time()
    successful_runs = 0
    for i in range(2000):
        if case == 'sequential':
            batch_stuff = dataset.random_sequential_batch(32)
            if batch_stuff[0] is not None:
                successful_runs += 1
        elif case == 'reward_prioritized':
            batch_stuff = dataset.reward_prediction_prioritized_sample(3)
            if batch_stuff[0] is not None:
                successful_runs += 1
        else:
            dataset.random_batch(32)
            successful_runs += 1
    print("batches per second: {}. Unsuccessful runs: {}".format(successful_runs / (time.time() - start), 2000 - successful_runs))


def trivial_tests():
    dataset = DataSet(width=2, height=1,
                      rng=np.random.RandomState(42),
                      max_steps=3, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 2, 2, True)
    print("last", dataset.last_phi())
    print("random", dataset.random_batch(1))


def max_size_tests():
    dataset1 = DataSet(width=3, height=4,
                       rng=np.random.RandomState(42),
                       max_steps=10, phi_length=4)
    dataset2 = DataSet(width=3, height=4,
                       rng=np.random.RandomState(42),
                       max_steps=1000, phi_length=4)
    for i in range(100):
        img = np.random.randint(0, 256, size=(4, 3))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_phi(),
                                             dataset2.last_phi())
        print("passed")


def test_memory_usage_ok():
    import memory_profiler
    dataset = DataSet(width=80, height=80,
                      rng=np.random.RandomState(42),
                      max_steps=1600000, phi_length=4)
    last = time.time()

    for i in range(1000000000):
        if (i % 100000) == 0:
            print(i)
        dataset.add_sample(np.random.random((80, 80)), 1, 1, False)
        if i > 200000:
            states, actions, rewards, next_states, terminals = \
                                        dataset.random_batch(32)
        if (i % 10007) == 0:
            print(time.time() - last)
            mem_usage = memory_profiler.memory_usage(-1)
            print(len(dataset), mem_usage)
        last = time.time()


def test_random_sequential_batch():
    dataset = DataSet(width=2, height=2,
                      max_steps=20, phi_length=4)
    for i in range(20):
        dataset.add_sample(np.asarray([[i, i], [i*10, i*10]]), i, i, (i+1) % 10 == 0)
    print(dataset.random_sequential_batch(3))


def test_last_batch():
    dataset = DataSet(width=2, height=2,
                      max_steps=10, phi_length=4)
    for i in range(20):
        dataset.add_sample(np.asarray([[i, i], [i*10, i*10]]), i, i, (i+1) % 10 == 0)
    print(dataset.last_batch(5))


def main():
    print('non sequential')
    speed_tests()
    print('sequential')
    speed_tests('sequential')
    print('reward_prioritized')
    speed_tests('reward_prioritized')
    # print('last batch')
    # test_last_batch()
    # test_memory_usage_ok()
    # max_size_tests()
    # simple_tests()
    # test_random_sequential_batch()

if __name__ == "__main__":
    main()
