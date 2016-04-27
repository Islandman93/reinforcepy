import numpy as np
import theano
import theano.tensor as T
import lasagne
from .dqn_inits import create_NIPS_sprag_init
from .dqn_rmsprop import deepmind_rmsprop


class AsyncTargetCNN:
    def __init__(self, network_parms, training_parms):
        network_parms.required(['input_shape'])
        training_parms.required(['discount', 'gradient_clip'])

        # setup shared vars
        inp_shape = network_parms.get('input_shape')
        self.state = theano.shared(np.zeros((1, inp_shape[1], inp_shape[2], inp_shape[3]), dtype=theano.config.floatX))
        self.state_tp1 = theano.shared(np.zeros((1, inp_shape[1], inp_shape[2], inp_shape[3]), dtype=theano.config.floatX))

        network_dic = create_NIPS_sprag_init(network_parms)
        self.l_in = network_dic['l_in']
        self.l_hid1 = network_dic['l_hid1']
        self.l_hid2 = network_dic['l_hid2']
        self.l_hid3 = network_dic['l_hid3']
        self.l_out = network_dic['l_out']
        self.target_l_out = create_NIPS_sprag_init(network_parms)['l_out']

        # if initial values in network parms set values
        if network_parms.has('initial_vals'):
            lasagne.layers.set_all_param_values(self.l_out, network_parms.get('initial_vals'))
        # set target from l_out parms
        lasagne.layers.set_all_param_values(self.target_l_out, lasagne.layers.get_all_param_values(self.l_out))

        # network output vars
        net_output = lasagne.layers.get_output(self.l_out, inputs=self.state)
        target_output = lasagne.layers.get_output(self.target_l_out, inputs=self.state_tp1)

        # setup qlearning values and loss
        loss_shared_vars = self.create_loss_shared_vars()
        loss = self.loss_fn(net_output, target_output, loss_shared_vars, training_parms.get('discount'))

        # get layer parms
        params = lasagne.layers.get_all_params(self.l_out)
        grads = T.grad(loss, params)
        grads = lasagne.updates.total_norm_constraint(grads, training_parms.get('gradient_clip'))
        grads.append(loss)

        # updates
        self.w1_update = theano.shared(np.zeros(self.l_hid1.W.eval().shape, dtype=theano.config.floatX))
        self.w2_update = theano.shared(np.zeros(self.l_hid2.W.eval().shape, dtype=theano.config.floatX))
        self.b1_update = theano.shared(np.zeros(self.l_hid1.b.eval().shape, dtype=theano.config.floatX))
        self.b2_update = theano.shared(np.zeros(self.l_hid2.b.eval().shape, dtype=theano.config.floatX))
        self.w3_update = theano.shared(np.zeros(self.l_hid3.W.eval().shape, dtype=theano.config.floatX))
        self.b3_update = theano.shared(np.zeros(self.l_hid3.b.eval().shape, dtype=theano.config.floatX))
        self.w4_update = theano.shared(np.zeros(self.l_out.W.eval().shape, dtype=theano.config.floatX))
        self.b4_update = theano.shared(np.zeros(self.l_out.b.eval().shape, dtype=theano.config.floatX))

        network_updates = [self.w1_update, self.b1_update, self.w2_update, self.b2_update,
                           self.w3_update, self.b3_update, self.w4_update, self.b4_update]
        theano_updates = deepmind_rmsprop(network_updates, params, training_parms.get('learning_rate'))

        self._get_grads = self.get_grads_fn(loss_shared_vars, grads)
        self._get_output = theano.function([], net_output)
        self._get_target_output = theano.function([], target_output)
        self._gradient_step = theano.function([], updates=theano_updates)

        self.accumulated_grads = None

    def create_loss_shared_vars(self):
        action = T.iscalar()
        reward = T.scalar()
        terminal = T.scalar()
        return [action, reward, terminal]

    def loss_fn(self, net_output, target_output, loss_shared_vars, discount):
        action = loss_shared_vars[0]
        reward = loss_shared_vars[1]
        terminal = loss_shared_vars[2]
        est_rew_tp1 = (1-terminal) * discount * T.max(target_output, axis=1)
        y = reward + est_rew_tp1
        return T.mean((y - net_output[0, action])**2)

    def get_grads_fn(self, loss_shared_vars, grads):
        action = loss_shared_vars[0]
        reward = loss_shared_vars[1]
        terminal = loss_shared_vars[2]
        return theano.function([action, reward, terminal], grads)

    def accumulate_gradients(self, state, action, reward, state_tp1, terminal):
        self.state.set_value(state)
        self.state_tp1.set_value(state_tp1)
        grads = self._get_grads(action, reward, terminal)
        loss = grads.pop()
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            for new_grad, acc_grad in zip(grads, self.accumulated_grads):
                acc_grad += new_grad
        return loss, grads

    def get_output(self, state):
        self.state.set_value(state)
        return self._get_output()

    def get_target_output(self, state):
        self.state_tp1.set_value(state)
        return self._get_target_output()

    def gradient_step(self, gradients):
        self.w1_update.set_value(gradients[0])
        self.b1_update.set_value(gradients[1])
        self.w2_update.set_value(gradients[2])
        self.b2_update.set_value(gradients[3])
        self.w3_update.set_value(gradients[4])
        self.b3_update.set_value(gradients[5])
        self.w4_update.set_value(gradients[6])
        self.b4_update.set_value(gradients[7])
        self._gradient_step()

    def set_parameters(self, new_parms):
        lasagne.layers.set_all_param_values(self.l_out, new_parms)

    def get_parameters(self):
        return lasagne.layers.get_all_param_values(self.l_out)

    def set_target_parameters(self, new_parms):
        lasagne.layers.set_all_param_values(self.target_l_out, new_parms)

    def get_gradients(self):
        return self.accumulated_grads

    def clear_gradients(self):
        self.accumulated_grads = None


class AsyncTargetCNNSarsa(AsyncTargetCNN):
    def create_loss_shared_vars(self):
        action = T.iscalar()
        reward = T.scalar()
        action_tp1 = T.iscalar()  # only used for sarsa
        terminal = T.scalar()
        return [action, reward, action_tp1, terminal]

    def loss_fn(self, net_output, target_output, loss_shared_vars, discount):
        action = loss_shared_vars[0]
        reward = loss_shared_vars[1]
        action_tp1 = loss_shared_vars[2]
        terminal = loss_shared_vars[3]
        est_rew_tp1 = (1-terminal) * discount * target_output[0, action_tp1]
        y = reward + est_rew_tp1
        return T.mean((y - net_output[0, action])**2)

    def get_grads_fn(self, loss_shared_vars, grads):
        action = loss_shared_vars[0]
        reward = loss_shared_vars[1]
        action_tp1 = loss_shared_vars[2]
        terminal = loss_shared_vars[3]
        return theano.function([action, reward, action_tp1, terminal], grads)

    def accumulate_gradients(self, state, action, reward, action_tp1, state_tp1, terminal):
        self.state.set_value(state)
        self.state_tp1.set_value(state_tp1)
        grads = self._get_grads(action, reward, action_tp1, terminal)
        loss = grads.pop()
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            for new_grad, acc_grad in zip(grads, self.accumulated_grads):
                acc_grad += new_grad
        return loss, grads


class AsyncTargetCNNNstep(AsyncTargetCNN):
    def __init__(self, network_parms, training_parms):
        training_parms.required(['training_size'])
        network_parms.required(['input_shape'])
        inp_shape = network_parms.get('input_shape')
        training_size = training_parms.get('training_size')
        self.training_states = theano.shared(np.zeros((training_size, inp_shape[1], inp_shape[2], inp_shape[3]),
                                                      dtype=theano.config.floatX))
        self.training_actions = theano.shared(np.zeros(training_size, dtype=np.int32))
        self.training_rewards = theano.shared(np.zeros(training_size, dtype=theano.config.floatX))
        self.net_training_output = None
        training_parms.set('discount', 0)  # force async base to let us through
        super().__init__(network_parms, training_parms)

    def create_loss_shared_vars(self):
        self.net_training_output = lasagne.layers.get_output(self.l_out, inputs=self.training_states)
        return None

    def loss_fn(self, net_output, target_output, loss_shared_vars, discount):
        return T.sum((self.training_rewards - self.net_training_output[:, self.training_actions])**2)

    def get_grads_fn(self, loss_shared_vars, grads):
        # don't need loss shared vars as they are set in init
        return theano.function([], grads)

    def accumulate_gradients(self, states, actions, rewards):
        self.training_states.set_value(states)
        self.training_actions.set_value(actions)
        self.training_rewards.set_value(rewards)
        grads = self._get_grads()
        loss = grads.pop()
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            raise ValueError("Should not be accumulating gradients for NSTEP, the grad function is sum")
        return loss, grads