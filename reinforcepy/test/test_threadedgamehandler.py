import numpy as np
from reinforcepy.handlers import ThreadedGameHandler


class randomAgent():
    def __init__(self):
        self.actions = None

    def frames_processed(self, frames, action_performed, reward):
        pass

    def get_game_action(self):
        rand_action = np.random.randint(0, len(self.actions))
        return self.actions[rand_action]

    def game_over(self):
        pass

    def set_legal_actions(self, legal_actions):
        self.actions = legal_actions

# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
skip_frame = 4
threadedGameHandler = ThreadedGameHandler(rom, False, skip_frame, 4)

threads = 4
learners = list()

for thread in range(threads):
    newAgent = randomAgent()
    newAgent.set_legal_actions(threadedGameHandler.get_legal_actions())
    learners.append(newAgent)

import time
st = time.time()
for a in range(4):
    for count, learner in enumerate(learners):
        def callback(reward):
            print('reward', reward)
        threadedGameHandler.async_run_emulator(learner, callback)

threadedGameHandler.block_until_done()
et = time.time()
print('total time', et-st)