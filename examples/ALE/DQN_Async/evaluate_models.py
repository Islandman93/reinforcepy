import argparse
import re
import os
import json
from functools import partial
from reinforcepy.environments import ALEEnvironment
from reinforcepy.learners.dqn.asynchronous import AsyncQLearner
from reinforcepy.evaluation import eval_null_ops, plot_save_rewards
# possible models
from reinforcepy.networks.dqn.tflow.nstep_a3c import NStepA3C
from reinforcepy.networks.dqn.tflow.target_dqn import TargetDQN
import reinforcepy.networks.util.tflow_util as tf_util


parser = argparse.ArgumentParser(description='Evaluate an entire directory of trained models')
parser.add_argument('models_path', type=str)
parser.add_argument('cfg_path', type=str)
parser.add_argument('eval_type', type=str, default='null_op', help="One of 'null_op', 'human_starts', 'nothing_special'")
parser.add_argument('model_type', type=str, help="One of 'a3c', 'dqn', 'double', 'dueling', 'nstep', 'doublenstep', 'duelingnstep'")
args = parser.parse_args()

# load and parse config
CONFIG = json.load(open(args.cfg_path))
rom_args = CONFIG['rom_args']
learner_args = CONFIG['learner_args']
network_args = CONFIG['network_args']

# create a dummy env to get num actions and input shape
environment = ALEEnvironment(**rom_args)
num_actions = environment.get_num_actions()
input_shape = [learner_args['phi_length']] + environment.get_state_shape()


# create network then load
def create_load_learner(model_path):
    if args.model_type == 'a3c':
        network = NStepA3C(input_shape, num_actions, **network_args)
    elif 'dueling' in args.model_type:
        # dueling we have to specify a different network generator for value and advantage output
        network = TargetDQN(input_shape, num_actions, args.model_type, network_generator=tf_util.create_dueling_nips_network, **network_args)
    else:
        network = TargetDQN(input_shape, num_actions, args.model_type, **network_args)
    network.load(model_path)
    # create learner
    learner = AsyncQLearner(environment, network, {}, **learner_args, testing=True)
    return learner


# find all models in the directory
stuff_in_dir = os.listdir(args.models_path)
print(stuff_in_dir)
models = []
for name in stuff_in_dir:
    # if like model-1234
    if re.match('^model-\d*$', name):
        models.append(name)

print('Found models {}'.format(models))
# iterate all models
try:
    reward_dict = {}
    for model_name in models:
        learner = create_load_learner(os.path.join(args.models_path, model_name))
        # get the number of steps from the model
        model_num_steps = int(model_name.split('-')[-1])
        model_num_frames = model_num_steps * CONFIG['rom_args']['skip_frame'] if 'skip_frame' in CONFIG['rom_args'] else 1
        # no op evaluation
        if args.eval_type == 'null_op':
            null_op_rewards = []
            # 30 episodes evaluated for null_op start
            for i in range(30):
                null_op_reward, null_ops = eval_null_ops(partial(ALEEnvironment, **rom_args), learner, max_no_ops=30)
                null_op_rewards.append(null_op_reward)
                print('Model: {}, episode: {}, null-ops: {}, reward: {}'.format(model_name, i, null_ops, null_op_reward))
        reward_dict[model_num_frames] = null_op_rewards

        # output reward dict
        with open(os.path.join(args.models_path, 'null_op_eval_results.json'), 'w') as out_file:
            json.dump(reward_dict, out_file)

except KeyboardInterrupt:
    print('Keyboard Interrupt')

# plot pretty picture
plot_save_rewards(reward_dict, os.path.join(args.models_path, 'null_op_eval_results.png'))
