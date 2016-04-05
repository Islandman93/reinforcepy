import time

import matplotlib.pyplot as plt
import numpy as np
from DoubleDQNLearner import DoubleDQNLearner
from learningALE.tools.life_ram_inds import BREAKOUT

from learningALE.handlers.ale_specific.gamehandler import GameHandler

# setup vars
rom = b'D:\\_code\\_reinforcementlearning\\breakout.bin'
gamename = 'breakout'
skip_frame = 4
num_actions = 4
learner = DoubleDQNLearner(skip_frame, num_actions)
game_handler = GameHandler(rom, False, learner, skip_frame)
scoreList = list()
validLossList = list()
bestTotReward = -np.inf

plt.ion()
st = time.time()
for episode in range(5000):
    learner.copy_new_target()  # copy a new target learner at the start of each game
    total_reward = game_handler.run_one_game(learner, lives=5, life_ram_ind=BREAKOUT)
    scoreList.append(total_reward)

    learner.game_over()

    # if this is the best score save it as such
    if total_reward >= bestTotReward:
        learner.save('dqnbest{0}.pkl'.format(total_reward))
        bestTotReward = total_reward

    # plot cost and score
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(learner.get_cost_list(), '.')
    plt.subplot(1, 2, 2)
    sl = np.asarray(scoreList)
    plt.plot(sl, '.')
    plt.pause(0.01)

    # save params every 10 games
    if episode % 10 == 0:
        learner.save('dqn{0}.pkl'.format(episode))

    et = time.time()
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    print('Total Time:', et - st, 'Frame Count:', game_handler.total_frame_count, 'FPS:', game_handler.total_frame_count / (et - st))

plt.ioff()
plt.show()

# final save
learner.save('dqn{0}.pkl'.format(episode+1))
