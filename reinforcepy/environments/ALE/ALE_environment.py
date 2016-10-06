import numpy as np
from scipy.misc import imresize
from reinforcepy.environments import BaseEnvironment
from ale_python_interface import ALEInterface


class ALEEnvironment(BaseEnvironment):
    """
    The :class:`MinimalGameHandler` class takes care of the interface to the ALE and tries to do nothing else. It's
    meant for advanced users who need fine control over every aspect of the process. It has many functions that are simply
    wrappers of the underlying ALE but with pythonic names/usage.

    Parameters
    ----------
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Default False. Whether or not to show the game. True takes longer to run but can be fun to watch
    """
    def __init__(self, rom, resize_shape=(84, 84), skip_frame=1, repeat_action_probability=0.0,
                 loss_of_life_termination=False, loss_of_life_negative_reward=False, show_rom=False):
        # set up emulator
        self.ale = ALEInterface()

        if show_rom:
            self.ale.setBool(b'display_screen', True)

        self.ale.setInt(b'frame_skip', skip_frame)
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)

        self.ale.loadROM(rom.encode())

        # setup gamescreen object. I think this is faster than recreating an empty each time
        width, height = self.ale.getScreenDims()
        self.gamescreen = np.empty((height, width, 1), dtype=np.uint8)

        self.resize_shape = resize_shape

        self.skip_frame = skip_frame

        # setup action converter
        # ALE returns legal action indexes, convert these to just numbers
        self.action_inds = self.ale.getMinimalActionSet()

        # setup lives
        self.loss_of_life_negative_reward = loss_of_life_negative_reward
        self.cur_lives = self.ale.lives()
        self.loss_of_life_termination = loss_of_life_termination
        self.life_lost = False

    def reset(self):
        self.ale.reset_game()
        self.cur_lives = self.ale.lives()
        self.life_lost = False

    def step(self, action):
        ale_action = self.action_inds[action]
        return self._step(ale_action)

    def _step(self, ale_action):
        if not self.loss_of_life_termination and not self.loss_of_life_negative_reward:
            return self.ale.act(ale_action)
        else:
            rew = self.ale.act(ale_action)
            new_lives = self.ale.lives()
            if new_lives < self.cur_lives:
                # if loss of life is negative reward subtract 1 from reward
                if self.loss_of_life_negative_reward:
                    rew -= 1
                self.cur_lives = new_lives
                self.life_lost = True
            return rew

    def get_state(self):
        self.gamescreen = self.ale.getScreenGrayscale(self.gamescreen)
        # convert ALE gamescreen into 84x84 image
        processedImg = imresize(self.gamescreen[:, :, 0], self.resize_shape)
        return processedImg

    def get_state_shape(self):
        return self.resize_shape

    def get_terminal(self):
        if self.loss_of_life_termination and self.life_lost:
            return True
        else:
            return self.ale.game_over()

    def get_num_actions(self):
        return len(self.action_inds)
