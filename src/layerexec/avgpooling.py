from copy import deepcopy


def feedforward(layer, T_i, examples_multitude, training, lparams):
    del lparams
    '''
    Description:
        It is the feedforward procedure of the Avg Pooling Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples that the input holds.
        <lparams>: [UNUSED VARIABLE]

    Returns:
        <T_o>: ndarray with shape (examples, output_height, output_width, output_channels). Output of the Avg Pooling Layer.
    '''

    T_padded_i = layer.pad.output(T_i, layer.p)

    minibatch_shape_o = (examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o)

    Phi_o = T_o = layer.tm_.tm.nan*layer.tm_.tm.ones(minibatch_shape_o)

    for i in range(examples_multitude):
        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    sum = 0
                    for h0 in range(layer.f):
                        for w0 in range(layer.f):
                            sum += T_padded_i[i, h*layer.s+h0, w*layer.s+w0, c]
                    T_o[i, h, w, c] = deepcopy(sum)

    T_o /= layer.f**2

    Phi_o, T_o = layer.dropout(layer, Phi_o, T_o, examples_multitude, training)

    return Phi_o, T_o

def feedforward_opt1(layer, T_i, examples_multitude, training, lparams):

    T_padded_i = layer.pad.output(T_i, layer.p)

    T_padded_i = layer.as_strided\
    (
        T_padded_i,
        shape=\
        (
            examples_multitude,
            layer.n_h_o,
            layer.n_w_o,
            layer.f,
            layer.f,
            layer.n_c_i
        ),
        strides=\
        (
            T_padded_i.strides[0],
            layer.s*T_padded_i.strides[1],
            layer.s*T_padded_i.strides[2],
            T_padded_i.strides[1],
            T_padded_i.strides[2],
            T_padded_i.strides[3]
        ),
    )

    Phi_o = T_o = layer.tm_.tm.sum(T_padded_i, axis=(3,4))/layer.f**2

    Phi_o, T_o = layer.dropout(layer, Phi_o, T_o, examples_multitude, training)

    return Phi_o, T_o

def backprop(layer, layer_next, delta_next, examples_multitude, K_next, Phi, T, layer_is_convolutional):

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    sum_ = layer.tm_.tm.nan*layer.tm_.tm.ones((examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o))
    for i in range(examples_multitude):
        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    sum = 0
                    for h0 in range(layer_next.n_h_o):
                        for w0 in range(layer_next.n_w_o):
                            if \
                            (
                                ( 0 <= h+layer_next.p-h0*layer_next.s <= layer_next.f-1 )
                                and
                                ( 0 <= w+layer_next.p-w0*layer_next.s <= layer_next.f-1 )
                            ):
                                sum += delta_next[i, h0, w0, c] * A[i, h, w, c]
                    sum_[i, h, w, c] = deepcopy(sum)

    delta = sum_ / layer_next.f**2

    return delta

def backprop_opt1(layer, layer_next, delta_next, examples_multitude, K_next, Phi, T, layer_is_convolutional):

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)/(layer_next.f)**2
    else:
        A = layer.tm_.tm.ones(Phi.shape)/(layer_next.f)**2

    ## Injecting the dilation
    delta_next = layer.tm_.insert(delta_next, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta_next.shape[1]), layer_next.s-1), 0, axis=1)
    delta_next = layer.tm_.insert(delta_next, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta_next.shape[2]), layer_next.s-1), 0, axis=2)

    ## Fill the padding
    delta_next = layer.pad.output(delta_next, layer_next.f-1)

    ## Removing the unused upper left and top part
    delta_next = delta_next[:, layer_next.p:, layer_next.p:, ...]
    # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(layer_next.p), axis=1)
    # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(layer_next.p), axis=2)

    ## This is how many excessively more indices the convolution is going to use, relative to the size of the dilated and paddef delta_next. It's (range of loop)-(size of delta).
    diff_h = layer.n_h_o+layer_next.f-2-delta_next.shape[1]+1
    diff_w = layer.n_w_o+layer_next.f-2-delta_next.shape[2]+1

    if diff_h > 0:

        delta_next = layer.tm_.append(delta_next, layer.tm_.tm.zeros((delta_next.shape[0], diff_h, delta_next.shape[2], delta_next.shape[3])), axis=1)

    elif diff_h < 0:

        delta_next = delta_next[:, delta_next.shape[1]+diff_h:, ...]
        # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(delta_next.shape[1]+diff_h, delta_next.shape[1]), axis=1)

    if diff_w > 0:

        delta_next = layer.tm_.append(delta_next, layer.tm_.tm.zeros((delta_next.shape[0], delta_next.shape[1], diff_w, delta_next.shape[3])), axis=2)

    elif diff_w < 0:

        delta_next = delta_next[:, :, delta_next.shape[2]+diff_w:, ...]
        # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(delta_next.shape[2]+diff_w, delta_next.shape[2]), axis=2)

    delta_next = layer.as_strided\
    (
        delta_next,
        shape=\
        (
            examples_multitude,
            layer.n_h_o,
            layer.n_w_o,
            layer_next.f,
            layer_next.f,
            layer_next.n_c_o
        ),
        strides=\
        (
            delta_next.strides[0],
            delta_next.strides[1],
            delta_next.strides[2],
            delta_next.strides[1],
            delta_next.strides[2],
            delta_next.strides[3]
        ),
    )

    pseudo_kernel_next = layer.tm_.tm.ones((layer_next.f, layer_next.f))

    delta = layer.tm_.tm.einsum('abcdef,de->abcf', delta_next, pseudo_kernel_next) * A

    return delta

def backprop_naive(layer, layer_next, delta_next, texamples_multitude, K_next, Phi, T, layer_is_convolutional):

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    delta = layer.tm_.tm.zeros((texamples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o))

    for i in range(texamples_multitude):
        for h in range(layer_next.n_h_o):
            hi = h * layer_next.s; hf = hi + layer_next.f
            for w in range(layer_next.n_w_o):
                wi = w * layer_next.s; wf = wi + layer_next.f
                for c in range(layer.n_c_o):
                    slice = T[i, hi:hf, wi:wf, c]
                    delta[i, hi:hf, wi:wf, c] += delta_next[i, h, w, c] * A[i, hi:hf, wi:wf, c] / (layer_next.f**2)

    return delta
