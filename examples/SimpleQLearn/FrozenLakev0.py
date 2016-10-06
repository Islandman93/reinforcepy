from reinforcepy.environments import GymWrapper
from reinforcepy.learners.TableLookup import QTable, SARSATable
import gym
import numpy as np

# FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.
ACTION_NAMES = ['l', 'd', 'u', 'r']
env = GymWrapper(gym.make('FrozenLake-v0'))

anneal_tuple = (1, 0.01, 1000)  # start eGreedy from 1 - 0.01 for 1000 episodes
qtable = QTable(env.get_legal_actions(), anneal_tuple)
sarsatable = SARSATable(env.get_legal_actions(), anneal_tuple)

MAX_EPISODES = 5000


def run_till_convergence(learner, env):
    # generate a reward list to test for convergence
    reward_list = []
    converged = False
    episodes = 0
    while not converged:
        reward = learner.run_episode(env)
        reward_list.append(reward)
        episodes += 1

        # test for convergence, if last 78 of 100 have a reward of 1 it has been solved
        if len(reward_list) > 100:  # happens when less than 100 iterations
            goals_found = reward_list[-100:]
            goal_pct = sum(goals_found) / 100
            # print(goal_pct)
            if goal_pct >= 0.78:
                converged = True
        # don't run forever if we can't converge
        if episodes > MAX_EPISODES:
            converged = True
    return episodes, goal_pct


# prints arrows for each option in the environment
def print_table(learner):
    print_str = ''
    for ind in range(16):
        # if x line finished
        if ind % 4 == 0 and ind > 0:
            print(print_str)
            print_str = ''

        print_str += ACTION_NAMES[np.argmax(learner.q_table[ind])]
    print(print_str)

q_episodes, q_goal_pct = run_till_convergence(qtable, env)
sarsa_episodes, sarsa_goal_pct = run_till_convergence(sarsatable, env)

print('{0} stopped after {1} episodes. Average win percent {2}'.format('QTable', q_episodes, q_goal_pct))
print('{0} stopped after {1} episodes. Average win percent {2}'.format('SARSATable', sarsa_episodes, sarsa_goal_pct))

print('QLearning TABLE')
print_table(qtable)
print('SARSA TABLE')
print_table(sarsatable)
