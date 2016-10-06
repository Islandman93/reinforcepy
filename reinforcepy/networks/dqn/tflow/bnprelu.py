import tflearn


def create_bnprelu_network(input_tensor, output_num):
    l_hid1 = tflearn.batch_normalization(tflearn.conv_2d(input_tensor, 16, 8, strides=4, activation='prelu', scope='conv1'), scope='conv1/bn')
    l_hid2 = tflearn.batch_normalization(tflearn.conv_2d(l_hid1, 32, 4, strides=2, activation='prelu', scope='conv2'), scope='conv2/bn')
    l_hid3 = tflearn.batch_normalization(tflearn.fully_connected(l_hid2, 256, activation='prelu', scope='dense3'), scope='dense3/bn')
    out = tflearn.fully_connected(l_hid3, output_num, scope='denseout')

    return out
