import lasagne
import numpy as np
import theano.tensor as T
import theano.tensor.signal.downsample as downsample


class SPPLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(SPPLayer, self).__init__(incoming, **kwargs)

        # divide by 4 gives 16 patches
        self.win1 = (int(np.floor(incoming.output_shape[2]/4.0)), int(np.floor(incoming.output_shape[3]/4.0)))
        self.str1 = (int(np.ceil(incoming.output_shape[2]/4.0)), int(np.ceil(incoming.output_shape[3]/4.0)))

        # divide by 2 gives 4 patches
        self.win2 = (int(np.floor(incoming.output_shape[2]/2.0)), int(np.floor(incoming.output_shape[3]/2.0)))
        self.str2 = (int(np.ceil(incoming.output_shape[2]/2.0)), int(np.ceil(incoming.output_shape[3]/2.0)))

        # no divide is one max patch, this is achieved by just doing T.maximum after reshaping

    def get_output_for(self, input, **kwargs):
        p1 = T.reshape(downsample.max_pool_2d(input, ds=self.win1, st=self.str1), (input.shape[0], input.shape[1], 16))
        p2 = T.reshape(downsample.max_pool_2d(input, ds=self.win2, st=self.str2), (input.shape[0], input.shape[1], 4))
        r3 = T.reshape(input, (input.shape[0], input.shape[1], input.shape[2]*input.shape[3]))
        p3 = T.reshape(T.max(r3, axis=2), (input.shape[0], input.shape[1], 1))
        return T.concatenate((p1, p2, p3), axis=2)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], 21
