from copy import deepcopy


def convolution(layer, T_i, examples_multitude, lparams):
    '''
    Description:
        Performs a 2D discrete, multi-channeled strided convolution between two third rank tensors and adds the bias term to that quantity. This is the Convolutional Linear Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <K>: ndarray with shape (output_channels, input_height, input_width, input_channels), it's the convolutions kernel.
            <b>: ndarray with shape (output_channels, output_height, output_width)

    Returns:
        <Phi_o>: ndarray with shape (examples, output_height, output_width, output_channels). Output of the strided convolution summed with the bias term.
    '''

    (K, b) = lparams

    T_padded_i = layer.pad.output(T_i, layer.p)

    minibatch_shape_o = (examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o)

    Phi_o = layer.tm_.tm.zeros(minibatch_shape_o)

    for i in range(examples_multitude):
        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    for h0 in range(layer.f):
                        for w0 in range(layer.f):
                            for c_prev in range(layer.n_c_i):
                                Phi_o[i, h, w, c] += T_padded_i[i, h*layer.s+h0, w*layer.s+w0, c_prev] * K[c, h0, w0, c_prev]

    # for i in range(examples_multitude):
    #     for h in range(layer.n_h_o):
    #         hi = h * layer.s; hf = hi + layer.f
    #         for w in range(layer.n_w_o):
    #             wi = w * layer.s; wf = wi + layer.f
    #             for c in range(layer.n_c_o):
    #                 Phi_o[i, h, w, c] = np.sum(T_padded_i[i, hi:hf, wi:wf, :] * K[])

    ## Depending on the minibatch t.e. set cardinality we tile b across a new axis so that it can be added on each t.e.
    Phi_o = Phi_o + layer.tm_.tm.tile(b, (examples_multitude, 1, 1, 1))

    return Phi_o

def convolution_opt1(layer, T_i, examples_multitude, lparams):
    '''
    Description:
        Performs a 2D discrete, multi-channeled strided convolution between two third rank tensors and adds the bias term to that quantity. This is the Convolutional Linear Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <K>: ndarray with shape (output_channels, input_height, input_width, input_channels), it's the convolutions kernel.
            <b>: ndarray with shape (output_channels, output_height, output_width)

    Returns:
        <Phi_o>: ndarray with shape (examples, output_height, output_width, output_channels). Output of the strided convolution summed with the bias term.
    '''

    K, b = lparams

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

    Phi_o = layer.tm_.tm.einsum('abcdef,gdef->abcg', T_padded_i, K) + layer.tm_.tm.tile(b, (examples_multitude, 1, 1, 1))

    return Phi_o

def feedforward(layer, T_i, examples_multitude, training, lparams):
    '''
    Description:
        This is the feedforward mapping of the Convolutional Layer. It performs a multichanneled 2D strided discrete convolution between two third rank tensors adding a bias term and maps its output using a non-linear activation function.

    Parameters:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <K>: ndarray with shape (output_channels, input_height, input_width, input_channels), it's the convolutions kernel.
            <b>: ndarray with shape (output_channels, output_height, output_width)

    Returns:
        <T_o>: ndarray with shape (examples, output_height, output_width, output_channels). The Convolutional Layers output.
    '''

    Phi_o = convolution(layer, T_i, examples_multitude, lparams)

    T_o = layer.g.output(Phi_o)

    Phi_o, T_o = layer.dropout(layer, Phi_o, T_o, examples_multitude, training)

    return Phi_o, T_o

def feedforward_opt1(layer, T_i, examples_multitude, training, lparams):
    '''
    Description:
        This is the feedforward mapping of the Convolutional Layer. It performs a multichanneled 2D strided discrete convolution between two third rank tensors adding a bias term and maps its output using a non-linear activation function.

    Parameters:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <K>: ndarray with shape (output_channels, input_height, input_width, input_channels), it's the convolutions kernel.
            <b>: ndarray with shape (output_channels, output_height, output_width)

    Returns:
        <T_o>: ndarray with shape (examples, output_height, output_width, output_channels). The Convolutional Layers output.
    '''

    Phi_o = convolution_opt1(layer, T_i, examples_multitude, lparams)

    T_o = layer.g.output(Phi_o)

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
                            for c0 in range(layer_next.n_c_o):
                                if \
                                (
                                    ( 0 <= h+layer_next.p-h0*layer_next.s <= layer_next.f-1 )
                                    and
                                    ( 0 <= w+layer_next.p-w0*layer_next.s <= layer_next.f-1 )
                                ):
                                    sum += delta_next[i, h0, w0, c0] * K_next[c0, h+layer_next.p-h0*layer_next.s, w+layer_next.p-w0*layer_next.s, c] * A[i, h, w, c]

                    sum_[i, h, w, c] = deepcopy(sum)

    delta = sum_

    return delta

def backprop_opt1(layer, layer_next, delta_next, examples_multitude, K_next, Phi, T, layer_is_convolutional):
    '''
        Desciption:
            We utilize the idea behind https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710 with a slight modification, extending the delta tensor to the right and bottom sides.
    '''

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    ## Flipping the Kernel
    K_next_flipped = layer.tm_.tm.swapaxes(K_next, 1, 2)

    ## Injecting the dilation
    delta_next = layer.tm_.insert(delta_next, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta_next.shape[1]), layer_next.s-1), 0, axis=1)
    delta_next = layer.tm_.insert(delta_next, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta_next.shape[2]), layer_next.s-1), 0, axis=2)

    ## Fill the padding
    delta_next = layer.pad.output(delta_next, layer_next.f-1)

    ## Removing the unused upper left and top part (the final convolution will never interact with that area)
    delta_next = delta_next[:, layer_next.p:, layer_next.p:, ...]
    # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(layer_next.p), axis=1)
    # delta_next = layer.tm_.tm.delete(delta_next, layer.tm_.tm.arange(layer_next.p), axis=2)

    ## This is how many excessively more indices the convolution is going to use, relative to the size of the dilated and paddef delta_next. It's (range of loop)-(size of delta).
    diff_h = layer.n_h_o+layer_next.f-2-delta_next.shape[1]+1
    diff_w = layer.n_w_o+layer_next.f-2-delta_next.shape[2]+1

    if diff_h > 0:

        delta_next = layer.tm_.append(delta_next, layer.tm_.tm.zeros((delta_next.shape[0], diff_h, delta_next.shape[2], delta_next.shape[3])), axis=1)

    elif diff_h < 0:

        delta_next = delta_next[:, delta_next.shape[1]+diff_h:, ...] #layer.tm_.delete(delta_next, layer.tm_.tm.arange(delta_next.shape[1]+diff_h, delta_next.shape[1]), axis=1)

    if diff_w > 0:

        delta_next = layer.tm_.append(delta_next, layer.tm_.tm.zeros((delta_next.shape[0], delta_next.shape[1], diff_w, delta_next.shape[3])), axis=2)

    elif diff_w < 0:

        delta_next = delta_next[:, :, delta_next.shape[2]+diff_w:, ...] #layer.tm_.delete(delta_next, layer.tm_.tm.arange(delta_next.shape[2]+diff_w, delta_next.shape[2]), axis=2)

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

    K_next_flipped = layer.tm_.tm.swapaxes(K_next_flipped, 0, 3)

    delta = layer.tm_.tm.einsum('abcdef,gdef->abcg', delta_next, K_next_flipped) * A

    return delta

def backprop_naive(layer, layer_next, delta_next, examples_multitude, K_next, Phi, T, layer_is_convolutional):

    ## WARNING: Lame ass work.. this naive mode does not even work with paddings.

    if layer_is_convolutional:
        A = layer.g.gradient_output(Phi)
    else:
        A = layer.tm_.tm.ones(Phi.shape)

    ## K_next needs to be converted to an array of shape (height, width, input channels, output channels)
    K_next_swapped = K_next.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)

    delta = layer.tm_.tm.zeros((examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o))
    for i in range(examples_multitude):
        for h in range(layer_next.n_h_o):
            hi = h*layer_next.s; hf = hi + layer_next.f
            for w in range(layer_next.n_w_o):
                wi = w*layer_next.s; wf = wi + layer_next.f
                for c in range(layer_next.n_c_o):
                    delta[i, hi:hf, wi:wf, :] += delta_next[i, h, w, c] * K_next_swapped[:, :, :, c] * A[i, hi:hf, wi:wf, :]

    return delta

def gradient(layer, layer_prev, delta, examples_multitude, T_prev):

    T_padded_prev = layer.pad.output(T_prev, layer.p)
    gradC_K = layer.tm_.tm.zeros((examples_multitude, layer.n_c_o, layer.f, layer.f, layer_prev.n_c_o))
    gradC_b = layer.tm_.tm.zeros((examples_multitude, layer.n_h_o, layer.n_w_o, layer.n_c_o))
    for i in range(examples_multitude):
        for h in range(layer.f):
            for w in range(layer.f):
                for c in range(layer.n_c_o):
                    for c_prev in range(layer_prev.n_c_o):
                        for h0 in range(layer.n_h_o):
                            for w0 in range(layer.n_w_o):
                                 gradC_K[i, c, h, w, c_prev] += delta[i, h0, w0, c] * T_padded_prev[i, h0*layer.s+h, w0*layer.s+w, c_prev]

        for h in range(layer.n_h_o):
            for w in range(layer.n_w_o):
                for c in range(layer.n_c_o):
                    gradC_b[i, h, w, c] = delta[i, h, w, c]

    return [gradC_K, gradC_b]

def gradient_opt1(layer, layer_prev, delta, examples_multitude, T_prev):

    T_padded_prev = layer.pad.output(T_prev, layer.p)
    T_padded_prev = layer.as_strided\
    (
        T_padded_prev,
        shape=\
        (
            examples_multitude,
            layer.f,
            layer.f,
            layer.s*(layer.n_h_o-1)+1,
            layer.s*(layer.n_w_o-1)+1,
            layer_prev.n_c_o
        ),
        strides=\
        (
            T_padded_prev.strides[0],
            T_padded_prev.strides[1],
            T_padded_prev.strides[2],
            T_padded_prev.strides[1],
            T_padded_prev.strides[2],
            T_padded_prev.strides[3],
        ),
    )

    ## Injecting the dilation
    delta_ = layer.tm_.insert(delta, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta.shape[1]), layer.s-1), 0, axis=1)
    delta_ = layer.tm_.insert(delta_, layer.tm_.tm.repeat(layer.tm_.tm.arange(1, delta_.shape[2]), layer.s-1), 0, axis=2)

    gradC_K = layer.tm_.tm.einsum('abcdef,adeg->agbcf', T_padded_prev, delta_)

    gradC_b = delta

    return [gradC_K, gradC_b]
