def linear_fully_connected(layer, a_i, examples_multitude, lparams):
    '''
    Description:
        It performs matrix multiplication to its input, and adds a bias term. This is the FC Linear Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <a_i>: ndarray with shape (examples, input_length), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <W>: ndarray with shape (output_channels, input_length, output_length), it's the weights of the fully connected layer.
            <b>: ndarray with shape (output_length)

    Returns:
        <z_o>: ndarray with shape (examples, output_length). The Linear Layers output.
    '''

    W, b = lparams

    sum_ = layer.tm_.tm.zeros((examples_multitude, layer.n_fc_o))
    for i in range(examples_multitude):
        for n in range(layer.n_fc_o):
            sum = 0
            for n0 in range(layer.n_fc_i):
                sum += a_i[i, n0] * W[n0, n]
            sum_[i, n] = sum

    z_o = sum_ + layer.tm_.tm.tile(b, (examples_multitude, 1))

    return z_o

def linear_fully_connected_opt1(layer, a_i, examples_multitude, lparams):
    '''
    Description:
        It performs matrix multiplication to its input, and adds a bias term. This is the FC Linear Layer.

    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <a_i>: ndarray with shape (examples, input_length), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <W>: ndarray with shape (output_channels, input_length, output_length), it's the weights of the fully connected layer.
            <b>: ndarray with shape (output_length)

    Returns:
        <z_o>: ndarray with shape (examples, output_length). The Linear Layers output.
    '''

    W, b = lparams
    z_o = (a_i @ W) + layer.tm_.tm.tile(b, (examples_multitude, 1))

    return z_o

def feedforward(layer, a_i, examples_multitude, training, lparams):
    '''
    Description:
        This is the feedforward mapping of the FC Layer. It performs matrix multiplication to its input, adds a bias term, and maps it using a non-linear activation function.

    Parameters:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <a_i>: ndarray with shape (examples, input_length), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <W>: ndarray with shape (output_channels, input_length, output_length), it's the weights of the fully connected layer.
            <b>: ndarray with shape (output_channels, output_length)

    Returns:
        <z_o>: ndarray with shape (examples, output_length). The FC Layers linear output.
        <a_o>: ndarray with shape (examples, output_length). The FC Layers output.
    '''

    z_o = linear_fully_connected(layer, a_i, examples_multitude, lparams)

    a_o = layer.g.output(z_o)

    z_o, a_o = layer.dropout(layer, z_o, a_o, examples_multitude, training)

    return z_o, a_o

def feedforward_opt1(layer, a_i, examples_multitude, training, lparams):
    '''
    Description:
        This is the feedforward mapping of the FC Layer. It performs matrix multiplication to its input, adds a bias term, and maps it using a non-linear activation function.

    Parameters:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <a_i>: ndarray with shape (examples, input_length), it's the layers input.
        <examples_multitude>: int, the multitude of the training examples the input holds.
        <lparams>: tuple, holding
            <W>: ndarray with shape (output_channels, input_length, output_length), it's the weights of the fully connected layer.
            <b>: ndarray with shape (output_channels, output_length)

    Returns:
        <z_o>: ndarray with shape (examples, output_length). The FC Layers linear output.
        <a_o>: ndarray with shape (examples, output_length). The FC Layers output.
    '''

    z_o = linear_fully_connected_opt1(layer, a_i, examples_multitude, lparams)

    a_o = layer.g.output(z_o)

    z_o, a_o = layer.dropout(layer, z_o, a_o, examples_multitude, training)

    return z_o, a_o

def backprop_output_layer(a_o, y):

    delta = a_o - y

    return delta

def backprop(layer, layer_next, delta_next, W_next, z, examples_multitude):

    sum_ = layer.tm_.tm.nan*layer.tm_.tm.ones((examples_multitude, layer.n_fc_o))
    for i in range(examples_multitude):
        for n in range(layer.n_fc_o):
            sum = 0
            for n0 in range(layer_next.n_fc_o):
                sum += delta_next[i, n0] * W_next[n, n0]
            sum_[i, n] = sum

    delta = sum_ * layer.g.gradient_output(z)

    return delta

def backprop_opt1(layer, layer_next, delta_next, W_next, z, examples_multitude):

    delta = (delta_next @ W_next.T) * layer.g.gradient_output(z)

    return delta

def gradient(layer, layer_prev, delta, examples_multitude, a_prev):

    gradC_W = layer.tm_.tm.nan*layer.tm_.tm.ones((examples_multitude, layer_prev.n_fc_o, layer.n_fc_o))
    gradC_b = layer.tm_.tm.nan*layer.tm_.tm.ones((examples_multitude, layer.n_fc_o))
    for i in range(examples_multitude):
        for n in range(layer.n_fc_o):
            for n_prev in range(layer_prev.n_fc_o):
                gradC_W[i, n_prev, n] = delta[i, n] * a_prev[i, n_prev]
            gradC_b[i, n] = delta[i, n]

    return [gradC_W, gradC_b]

def gradient_opt1(layer, layer_prev, delta, examples_multitude, a_prev):

    gradC_W = layer.tm_.tm.einsum('ab,ac->acb', delta, a_prev)
    gradC_b = delta

    return [gradC_W, gradC_b]
