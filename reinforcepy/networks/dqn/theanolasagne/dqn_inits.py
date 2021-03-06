import lasagne


def create_NIPS(network_parms):
    validate_parms(network_parms)

    conv = get_lasagne_conv_layer()

    # setup network layout
    l_in = lasagne.layers.InputLayer([None] + network_parms.get('input_shape'))
    l_hid1 = conv(l_in, 16, (8, 8), stride=network_parms.get('stride')[0])

    l_hid2 = conv(l_hid1, 32, (4, 4), stride=network_parms.get('stride')[1])

    l_hid3 = lasagne.layers.DenseLayer(l_hid2, 256)

    l_out = lasagne.layers.DenseLayer(l_hid3, network_parms.get('output_num'), nonlinearity=lasagne.nonlinearities.linear)

    return {'l_in': l_in, 'l_hid1': l_hid1, 'l_hid2': l_hid2, 'l_hid3': l_hid3, 'l_out': l_out}


def create_A3C(network_parms):
    validate_parms(network_parms)
    conv = get_lasagne_conv_layer()

    # setup network layout
    l_in = lasagne.layers.InputLayer(network_parms.get('input_shape'))
    l_hid1 = conv(l_in, 16, (8, 8), stride=network_parms.get('stride')[0], untie_biases=network_parms.get('untie_biases'))

    l_hid2 = conv(l_hid1, 32, (4, 4), stride=network_parms.get('stride')[1], untie_biases=network_parms.get('untie_biases'))

    l_hid3 = lasagne.layers.DenseLayer(l_hid2, 256)

    l_value = lasagne.layers.DenseLayer(l_hid3, 1, nonlinearity=lasagne.nonlinearities.linear)
    l_policy = lasagne.layers.DenseLayer(l_hid3, network_parms.get('output_num'), nonlinearity=lasagne.nonlinearities.softmax)

    return {'l_in': l_in, 'l_hid1': l_hid1, 'l_hid2': l_hid2, 'l_hid3': l_hid3, 'l_value': l_value, 'l_policy': l_policy}


def create_NIPS_sprag_init(network_parms):
    validate_parms(network_parms)
    conv = get_lasagne_conv_layer()

    # setup network layout
    l_in = lasagne.layers.InputLayer([None] + network_parms.get('input_shape'))
    l_hid1 = conv(l_in, 16, (8, 8), stride=network_parms.get('stride')[0],
                  W=lasagne.init.Normal(.01),
                  b=lasagne.init.Constant(.1))

    l_hid2 = conv(l_hid1, 32, (4, 4), stride=network_parms.get('stride')[1],
                  W=lasagne.init.Normal(.01),
                  b=lasagne.init.Constant(.1))

    l_hid3 = lasagne.layers.DenseLayer(l_hid2, 256,
                                       W=lasagne.init.Normal(.01),
                                       b=lasagne.init.Constant(.1))

    l_out = lasagne.layers.DenseLayer(l_hid3, network_parms.get('output_num'),
                                      nonlinearity=lasagne.nonlinearities.linear,
                                      W=lasagne.init.Normal(.01),
                                      b=lasagne.init.Constant(.1))

    return {'l_in': l_in, 'l_hid1': l_hid1, 'l_hid2': l_hid2, 'l_hid3': l_hid3, 'l_out': l_out}


def create_async_muupan_init(network_parms):
    validate_parms(network_parms)
    conv = get_lasagne_conv_layer()

    # setup network layout
    input_shape = network_parms.get('input_shape')
    l_in = lasagne.layers.InputLayer(input_shape)
    l_hid1 = conv(l_in, 16, (8, 8), stride=network_parms.get('stride')[0],
                        W=TorchInit((input_shape[1], 8, 8)),
                        b=TorchInit((input_shape[1], 8, 8)))

    l_hid2 = conv(l_hid1, 32, (4, 4), stride=network_parms.get('stride')[1],
                        W=TorchInit((16, 4, 4)),
                        b=TorchInit((16, 4, 4)))

    l_hid3 = lasagne.layers.DenseLayer(l_hid2, 256,
                        W=TorchInit((32, 4, 4)),
                        b=TorchInit((32, 4, 4)))

    l_out = lasagne.layers.DenseLayer(l_hid3, network_parms.get('output_num'), nonlinearity=lasagne.nonlinearities.linear,
                        W=TorchInit(256),
                        b=TorchInit(256))

    return {'l_in': l_in, 'l_hid1': l_hid1, 'l_hid2': l_hid2, 'l_hid3': l_hid3, 'l_out': l_out}


def get_lasagne_conv_layer():
    import theano.tensor.signal.conv
    from theano.sandbox.cuda import dnn
    # if no dnn support use default conv
    if not theano.config.device.startswith("gpu") or not dnn.dnn_available():  # code stolen from lasagne dnn.py
        import lasagne.layers.conv
        conv = lasagne.layers.conv.Conv2DLayer
    else:
        import lasagne.layers.dnn
        conv = lasagne.layers.dnn.Conv2DDNNLayer
    return conv


def validate_parms(network_parms):
    network_parms.required(['input_shape', 'output_num', 'stride'])


# inspired by https://github.com/muupan/async-rl/blob/master/init_like_torch.py
class TorchInit(lasagne.init.Initializer):
    def __init__(self, fan_in):
        import numpy as np
        self.stdv = 1 / np.sqrt(np.prod(fan_in))

    def sample(self, shape):
        return lasagne.utils.floatX(lasagne.random.get_rng().uniform(low=-self.stdv, high=self.stdv, size=shape))
