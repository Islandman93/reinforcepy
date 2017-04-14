import numpy as np
import matplotlib.pyplot as plt


def eval_null_ops(environment_gen_fn, learner, max_no_ops=30):
    # generate a new environment, this creates a different seed
    environment = environment_gen_fn()
    null_ops = np.random.randint(0, max_no_ops)
    for n in range(null_ops):
        environment.step(0)
    # TODO: add average q vals
    return learner.run_episode(environment), null_ops


def plot_save_rewards(reward_dict, output_file_name):
    # sort reward_dict keys
    eval_steps = list(reward_dict.keys()).sort()

    # get mean, min, and max
    min_rewards, max_rewards, mean_rewards = [], [], []
    for eval_step in eval_steps:
        rewards = reward_dict[eval_step]
        min_rewards.append(np.min(rewards))
        max_rewards.append(np.max(rewards))
        mean_rewards.append(np.mean(rewards))

    # plot regions
    plt.figure(figsize=(24, 12))
    plt.plot(eval_steps, mean_rewards, lw=2, label='Mean Reward')
    plt.fill_between(eval_steps, min_rewards, max_rewards, lw=1, label='Min/Max Region', alpha=0.5)
    plt.legend()
    plt.xlabel('# Frames')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig(output_file_name)
    plt.show()
