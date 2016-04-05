import numpy as np
import theano
import theano.tensor as T
import lasagne
from learningALE.learners.nns import create_A3C

class AsyncA3CCNN:
    def __init__(self, inp_shape, output_num, training_size, stride=(4, 2), untie_biases=False):
        # setup shared vars
        self.state = theano.shared(np.zeros((1, inp_shape[1], inp_shape[2], inp_shape[3]), dtype=theano.config.floatX))
        self.training_states = theano.shared(np.zeros((training_size, inp_shape[1], inp_shape[2], inp_shape[3]),
                                                      dtype=theano.config.floatX))
        self.training_actions = theano.shared(np.zeros(training_size, dtype=np.int32))
        self.training_rewards = theano.shared(np.zeros(training_size, dtype=theano.config.floatX))

        network_dic = create_A3C(inp_shape, output_num, stride=stride, untie_biases=untie_biases)
        self.l_in = network_dic['l_in']
        self.l_hid1 = network_dic['l_hid1']
        self.l_hid2 = network_dic['l_hid2']
        self.l_hid3 = network_dic['l_hid3']
        self.l_policy = network_dic['l_policy']
        self.l_value = network_dic['l_value']

        # network output vars
        policy_output = lasagne.layers.get_output(self.l_policy, inputs=self.state)
        value_output = lasagne.layers.get_output(self.l_value, inputs=self.state)

        # setup training vars and loss
        training_policy_output = lasagne.layers.get_output(self.l_policy, inputs=self.training_states)
        training_value_output = lasagne.layers.get_output(self.l_value, inputs=self.training_states)

        # log(prediction, action taken) * (R - Value(states))
        # one_hot_true = T.zeros_like(training_policy_output)
        # one_hot_true = T.set_subtensor(one_hot_true[T.arange(self.training_actions.shape[0]), self.training_actions], 1)
        # rewrite categorical crossentropy here because the lasagne/theano function sums the result and I need per step
        # categorical_crossentropy = -T.sum(one_hot_true * T.log(training_policy_output), axis=1)
        entropy = 0.01 * -T.sum(training_policy_output * T.log2(training_policy_output), axis=1)
        value_diff_rewards = (self.training_rewards - training_value_output[:, 0] + entropy)

        # sum is to aggregate over the nsteps
        policy_loss = T.sum(T.log(training_policy_output[:, self.training_actions]) * value_diff_rewards)
        value_loss = T.sum((self.training_rewards - training_value_output[:, 0])**2)

        # get layer parms
        policy_params = lasagne.layers.get_all_params(self.l_policy)
        value_params = lasagne.layers.get_all_params(self.l_value)
        params = policy_params + self.l_value.get_params()

        # get grads
        policy_grads = T.grad(policy_loss, policy_params)
        value_grads = T.grad(value_loss, value_params)

        # combine grads for the non-output layers
        combine_grads = policy_grads[0:-2]
        for grad_ind in range(len(value_grads)-2):
            combine_grads[grad_ind] += value_grads[grad_ind]

        # add grads for policy and value layers
        grads = combine_grads + policy_grads[-2:] + value_grads[-2:]

        # add loss to return in grads list
        grads.append(policy_loss)
        grads.append(value_loss)

        # updates
        self.w1_update = theano.shared(np.zeros(self.l_hid1.W.eval().shape, dtype=theano.config.floatX))
        self.w2_update = theano.shared(np.zeros(self.l_hid2.W.eval().shape, dtype=theano.config.floatX))
        if untie_biases:
            self.b1_update = theano.shared(np.zeros(self.l_hid1.b.eval().shape, dtype=theano.config.floatX))
            self.b2_update = theano.shared(np.zeros(self.l_hid2.b.eval().shape, dtype=theano.config.floatX))
        else:
            self.b1_update = theano.shared(np.zeros(self.l_hid1.b.eval().shape, dtype=theano.config.floatX))
            self.b2_update = theano.shared(np.zeros(self.l_hid2.b.eval().shape, dtype=theano.config.floatX))
        self.w3_update = theano.shared(np.zeros(self.l_hid3.W.eval().shape, dtype=theano.config.floatX))
        self.b3_update = theano.shared(np.zeros(self.l_hid3.b.eval().shape, dtype=theano.config.floatX))
        self.l_policy_w_update = theano.shared(np.zeros(self.l_policy.W.eval().shape, dtype=theano.config.floatX))
        self.l_policy_b_update = theano.shared(np.zeros(self.l_policy.b.eval().shape, dtype=theano.config.floatX))
        self.l_value_w_update = theano.shared(np.zeros(self.l_value.W.eval().shape, dtype=theano.config.floatX))
        self.l_value_b_update = theano.shared(np.zeros(self.l_value.b.eval().shape, dtype=theano.config.floatX))

        network_updates = [self.w1_update, self.b1_update, self.w2_update, self.b2_update, self.w3_update,
                           self.b3_update, self.l_policy_w_update, self.l_policy_b_update, self.l_value_w_update,
                           self.l_value_b_update]
        theano_updates = lasagne.updates.rmsprop(network_updates, params, 0.0001)

        self._get_policy_output = theano.function([], policy_output)
        self._get_value_output = theano.function([], value_output)
        self._gradient_step = theano.function([], updates=theano_updates)
        self._get_grads = theano.function([], outputs=grads)

        self.accumulated_grads = None

    def accumulate_gradients(self, states, actions, rewards):
        self.training_states.set_value(states)
        self.training_actions.set_value(actions)
        self.training_rewards.set_value(rewards)
        grads = self._get_grads()
        value_loss = grads.pop()
        policy_loss = grads.pop()
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            raise ValueError("Should not be accumulating gradients for NSTEP, the grad function is sum")
        return (policy_loss, value_loss)

    def get_policy_output(self, state):
        self.state.set_value(state)
        return self._get_policy_output()

    def get_value_output(self, state):
        self.state.set_value(state)
        return self._get_value_output()

    def gradient_step(self, gradients):
        self.w1_update.set_value(gradients[0])
        self.b1_update.set_value(gradients[1])
        self.w2_update.set_value(gradients[2])
        self.b2_update.set_value(gradients[3])
        self.w3_update.set_value(gradients[4])
        self.b3_update.set_value(gradients[5])
        self.l_policy_w_update.set_value(gradients[6])
        self.l_policy_b_update.set_value(gradients[7])
        self.l_value_w_update.set_value(gradients[8])
        self.l_value_b_update.set_value(gradients[9])
        self._gradient_step()

    def set_parameters(self, new_parms):
        lasagne.layers.set_all_param_values(self.l_policy, new_parms[0:-2])

        # value layer set param values
        # code from lasagne.layers.helper.set_all_param_values
        for p, v in zip(self.l_value.get_params(), new_parms[-2:]):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    def get_parameters(self):
        value_parms = [p.get_value() for p in self.l_value.get_params()]
        return lasagne.layers.get_all_param_values(self.l_policy) + value_parms

    def get_gradients(self):
        return self.accumulated_grads

    def clear_gradients(self):
        self.accumulated_grads = None