import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.autograd as autograd
from torch.autograd import Variable
# from reinforcepy.handlers.linear_annealer import LinnearAnnealer


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
    def __init__(self, is_host=False):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.hid3 = nn.Linear(9*9*32, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.actor = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)
        self.apply(weights_init)
        self.batch_vars = []
        if is_host:
            self.optimizer = optim.RMSprop(self.parameters(), 0.0007, alpha=0.99, eps=0.1)
        self.last_lstm_state = None

    def forward(self, x, lstm_state):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.hid3(x))
        x = self.lstm(x, lstm_state)
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
            _, critic, _ = self.forward(Variable(torch_last_state).cuda(), self.last_lstm_state)
            curr_reward = critic.data[0, 0]

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + 0.99 * curr_reward
            td_rewards.append(curr_reward)
        # need to reverse to be forward looking
        td_rewards = list(reversed(td_rewards))

        # calculate losses
        batch_vars = self.batch_vars
        value_loss = 0
        for (action, value, entropy), r in zip(batch_vars, td_rewards):
            action.reinforce(value.data.squeeze() - r)
            # smooth_l1_loss == huber_loss
            loss = F.smooth_l1_loss(value, Variable(torch.Tensor([r])).cuda())
            value_loss += loss
            value_loss += entropy * 0.01

        optimizer.zero_grad()
        variables = [value_loss] + list(map(lambda p: p.action, batch_vars))
        gradients = [torch.ones(1).cuda()] + [None] * len(batch_vars)
        autograd.backward(variables, gradients)
        optimizer.step()
        # reset state
        self.reset_batch()


    def reset_batch_gradient_vars(self):
        self.batch_gradient_vars['log_policy_action'] = []
        self.batch_gradient_vars['entropy'] = []
        self.batch_gradient_vars['critic_output'] = []
    # def backward(self):
    #     #
    #     # calculate step returns in reverse order
    #     returns = []
    #     step_return = self.outputs[-1].value.data
    #     for reward in self.rewards[::-1]:
    #         step_return.mul_(self.discount).add_(reward.cuda() if USE_CUDA else reward)
    #         returns.insert(0, step_return.clone())
    #     #

    def reset_batch(self):
        self.batch_vars = []

    def reset_lstm_state(self):
        self.last_lstm_state = None
