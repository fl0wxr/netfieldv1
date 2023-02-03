from copy import deepcopy


def feedforward(layer, T_i, examples_multitude, training, lparams):
    del lparams
    '''
    Description:
        It is the feedforward procedure of the Max Pooling Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples that the input holds.
        <lparams>: [UNUSED VARIABLE]

    Returns:
        <T_o>: ndarray with shape (examples, output_height, output_width, output_channels). Output of the Max Pooling Layer.
    '''

    T_padded_i = layer.pad.output(T_i, layer.p)

    minibatch_shape_o = (examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o)

    Phi_o = T_o = layer.tm_.tm.nan*layer.tm_.tm.ones(minibatch_shape_o)

    for i in range(examples_multitude):
        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    max_ = T_padded_i[i, h*layer.s, w*layer.s, c]
                    for h0 in range(layer.f):
                        for w0 in range(layer.f):
                            ## c output = c input
                            max_ = max(max_, T_padded_i[i, h*layer.s+h0, w*layer.s+w0, c])
                    T_o[i, h, w, c] = max_

    Phi_o, T_o = layer.dropout(layer, Phi_o, T_o, examples_multitude, training)

    return Phi_o, T_o

def feedforward_opt1(layer, T_i, examples_multitude, training, lparams):

    T_padded_i = layer.pad.output(T_i, layer.p)

    T_padded_i_e = layer.as_strided\
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

    Phi_o = T_o = layer.tm_.tm.max(T_padded_i_e, axis=(3,4))

    Phi_o, T_o = layer.dropout(layer, Phi_o, T_o, examples_multitude, training)

    return Phi_o, T_o

def backprop(layer, layer_next, delta_next, texamples_multitude, K_next, Phi, T, layer_is_convolutional):

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    T_padded = layer.pad.output(T, layer_next.p)
    delta = layer.tm_.tm.zeros((texamples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o))

    for i in range(texamples_multitude):
        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    sum = 0

                    for h0 in range(layer_next.n_h_o):
                        for w0 in range(layer_next.n_w_o):
                            (h_amax, w_amax) = \
                                layer.tm_.tm.unravel_index(layer.tm_.tm.argmax(T_padded[i, h0*layer_next.s:h0*layer_next.s+layer_next.f, w0*layer_next.s:w0*layer_next.s+layer_next.f, c]), (layer_next.f, layer_next.f)) \

                            if \
                            (
                                h+layer_next.p == h0*layer_next.s+h_amax
                                and
                                w+layer_next.p == w0*layer_next.s+w_amax
                            ):

                                sum += delta_next[i, h0, w0, c] * A[i, h, w, c]

                    delta[i, h, w, c] = deepcopy(sum)

    return delta

def backprop_opt1(layer, layer_next, delta_next, texamples_multitude, K_next, Phi, T, layer_is_convolutional):

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    T_padded = layer.pad.output(T, layer_next.p)

    B = layer.tm_.tm.tile(delta_next[:,layer.tm_.tm.newaxis,layer.tm_.tm.newaxis,:,:,:], (1,layer.n_h_o,layer.n_w_o,1,1,1))

    C_H = layer.tm_.tm.nan*layer.tm_.tm.ones((texamples_multitude, layer.n_h_o, layer.n_w_o, layer_next.n_h_o, layer_next.n_w_o, layer.n_c_o))
    C_W = deepcopy(C_H)
    D_H = deepcopy(C_H)
    D_W = deepcopy(C_H)

    T_padded_e = layer.as_strided\
    (
        T_padded,
        shape=\
        (
            texamples_multitude,
            layer_next.n_h_o,
            layer_next.n_w_o,
            layer_next.f,
            layer_next.f,
            layer.n_c_o
        ),
        strides=\
        (
            T_padded.strides[0],
            layer_next.s*T_padded.strides[1],
            layer_next.s*T_padded.strides[2],
            T_padded.strides[1],
            T_padded.strides[2],
            T_padded.strides[3]
        ),
    )

    T_padded_e_r = layer.tm_.tm.reshape(T_padded_e, (texamples_multitude, layer_next.n_h_o, layer_next.n_w_o, layer_next.f**2, layer.n_c_o))

    hmax, wmax = layer.tm_.tm.unravel_index(layer.tm_.tm.argmax(T_padded_e_r, axis=3), (layer_next.f, layer_next.f))

    C_H = layer.tm_.tm.tile(hmax[:,layer.tm_.tm.newaxis,layer.tm_.tm.newaxis,:,:,:], (1,layer.n_h_o,layer.n_w_o,1,1,1))

    C_W = layer.tm_.tm.tile(wmax[:,layer.tm_.tm.newaxis,layer.tm_.tm.newaxis,:,:,:], (1,layer.n_h_o,layer.n_w_o,1,1,1))

    ## h rows VS h0 cols, and please consider the i,w,w0,c afterwards to understand the next expression. For w it's basically a case of repeating the same method.
    D_H = layer.tm_.tm.tile(layer.tm_.tm.arange(layer.n_h_o)[:,layer.tm_.tm.newaxis], (1,layer_next.n_h_o)) + layer.tm_.tm.tile(layer.tm_.tm.arange(0, -(layer_next.n_h_o-1)*layer_next.s-1, -layer_next.s), (layer.n_h_o,1))

    D_H = layer.tm_.tm.tile(D_H[layer.tm_.tm.newaxis,:,layer.tm_.tm.newaxis,:,layer.tm_.tm.newaxis,layer.tm_.tm.newaxis], (texamples_multitude, 1, layer.n_w_o, 1, layer_next.n_w_o, layer.n_c_o)) + layer_next.p

    D_W = layer.tm_.tm.tile(layer.tm_.tm.arange(layer.n_w_o)[:,layer.tm_.tm.newaxis], (1,layer_next.n_w_o)) + layer.tm_.tm.tile(layer.tm_.tm.arange(0, -(layer_next.n_w_o-1)*layer_next.s-1, -layer_next.s), (layer.n_w_o,1))

    D_W = layer.tm_.tm.tile(D_W[layer.tm_.tm.newaxis,layer.tm_.tm.newaxis,:,layer.tm_.tm.newaxis,:,layer.tm_.tm.newaxis], (texamples_multitude, layer.n_h_o, 1, layer_next.n_h_o, 1, layer.n_c_o)) + layer_next.p

    E = layer.tm_.tm.logical_and((C_H == D_H), (C_W == D_W))

    delta = layer.tm_.tm.sum(B*E, axis=(3,4)) * A

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
                    delta[i, hi:hf, wi:wf, c] += delta_next[i, h, w, c] * (slice == layer.tm_.tm.max(slice)) * A[i, hi:hf, wi:wf, c]

    return delta
