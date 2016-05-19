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
    def __init__(self, rom, neg_reward=False, early_termination=False, show_rom=False):
        # set up emulator
        self.ale = ALEInterface()

        if show_rom:
            self.ale.setBool(b'display_screen', True)
        self.ale.loadROM(rom)

        # setup gamescreen object. I think this is faster than recreating an empty each time
        width, height = self.ale.getScreenDims()
        self.gamescreen = np.empty((height, width, 1), dtype=np.uint8)

        # setup lives
        self.neg_reward = neg_reward
        self.cur_lives = self.ale.lives()
        self.early_termination = early_termination
        self.life_lost = False

    def reset(self):
        self.ale.reset_game()
        self.cur_lives = self.ale.lives()
        self.life_lost = False

    def step(self, action):
        if not self.neg_reward:
            return self.ale.act(action)
        else:
            rew = self.ale.act(action)
            new_lives = self.ale.lives()
            if new_lives < self.cur_lives:
                rew -= 1
                self.cur_lives = new_lives
                self.life_lost = True
            return rew

    def get_state(self):
        self.gamescreen = self.ale.getScreenGrayscale(self.gamescreen)
        # convert ALE gamescreen into 84x84 image
        processedImg = imresize(self.gamescreen[33:-16, :, 0], 0.525)
        return processedImg

    def get_state_shape(self):
        return self.ale.getScreenDims()

    def get_terminal(self):
        if self.early_termination and self.life_lost:
            return True
        else:
            return self.ale.game_over()

    def get_legal_actions(self):
        return self.ale.getMinimalActionSet()
