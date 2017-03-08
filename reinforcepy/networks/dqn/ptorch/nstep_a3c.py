import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from reinforcepy.handlers.linear_annealer import LinnearAnnealer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3CModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.hid3 = nn.Linear(9*9*32, 256)
        self.actor = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)
        self.apply(weights_init)
        self.optimizer = optim.RMSprop(self.parameters(), 0.0007, alpha=0.99, eps=0.1)
        self.batch_gradient_vars = {}
        self.reset_batch_gradient_vars()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.hid3(x))
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

    def get_output(self, states):
        states = torch.from_numpy(states).float() / 255.0
        actor_logits, critic = self.forward(Variable(states))
        actor_probs = F.softmax(actor_logits)
        action = actor_probs.multinomial()
        log_policy = F.log_softmax(actor_probs + 1e-6)
        self.batch_gradient_vars['log_policy_action'].append(torch.gather(log_policy, 1, action))
        self.batch_gradient_vars['entropy'].append((log_policy * actor_probs).sum(1))
        self.batch_gradient_vars['critic_output'].append(critic.sum())
        # self.batch_gradient_vars['action_props'].append(actor_probs)
        # cool torch has built in function to return sampled action from softmax
        return action.data[0, 0]

    def train_step(self, states, actions, rewards, states_tp1, terminals, learning_rate):
        # nstep calculate TD reward
        if sum(terminals) > 1:
            raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

        # last state not terminal need to query target network
        curr_reward = 0
        if not terminals[-1]:
            last_state = np.expand_dims(states_tp1[-1], 0)
            torch_last_state = torch.from_numpy(last_state).float() / 255.0
            _, critic = self.forward(Variable(torch_last_state))
            curr_reward = critic.data[0, 0]

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + 0.99 * curr_reward
            td_rewards.append(curr_reward)
        # need to reverse to be forward looking
        td_rewards = list(reversed(td_rewards))

        # get bootstrap estimate of last state_tp1 possible to do in Torch maybe for a speed up?
        # td_rewards = []
        # rewards = Variable(torch.from_numpy(np.asarray(rewards)).float())
        # for reward in reversed(rewards):
        #     curr_reward = reward + 0.99 * curr_reward
        #     td_rewards.append(curr_reward)

        # torch_states = torch.from_numpy(np.asarray(states)).float() / 255.0
        # actor, critic = self.forward(Variable(torch_states))
        # actions = [a.multinomial() for a in actor]

        loss = 0
        for log_action, critic, r in zip(self.batch_gradient_vars['log_policy_action'], self.batch_gradient_vars['critic_output'], td_rewards):
            critic_diff = critic - r
            loss += log_action * critic_diff.detach()
            loss += (critic_diff ** 2) * 0.5
            # value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r])))

        self.optimizer.zero_grad()
        # final_nodes = [value_loss] + actions
        # gradients = [torch.ones(1)] + [None] * len(states)
        # autograd.backward(final_nodes, gradients)
        loss.backward()
        self.optimizer.step()
        self.reset_batch_gradient_vars()

    def reset_batch_gradient_vars(self):
        self.batch_gradient_vars['log_policy_action'] = []
        self.batch_gradient_vars['entropy'] = []
        self.batch_gradient_vars['critic_output'] = []


# class PyTorchNStepA3C():
#     def __init__(self, input_shape, output_num, optimizer=None, network_generator=A3CModel, q_discount=0.99,
#                  entropy_regularization=0.01, global_norm_clipping=40, initial_learning_rate=0.001, learning_rate_decay=None,
#                  deterministic=False):
#         # if optimizer is none use default rms prop
#         self.network = network_generator()
#         if optimizer is None:
#             optimizer = optim.RMSprop(self.network.parameters(), initial_learning_rate, alpha=0.99, eps=0.1)
#         self.network.optimizer = optimizer
#         self.learning_rate_annealer = LinnearAnnealer(initial_learning_rate, 0, learning_rate_decay)
#         self.global_norm_clipping = global_norm_clipping
#         self._q_discount = q_discount
#         self._entropy_regularization = entropy_regularization
#         self.deterministic = deterministic
