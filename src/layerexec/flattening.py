def feedforward(layer, T_i, examples_multitude, training, lparams=None):
    '''
    Inputs:
        <layer>: class object, it contains all the required CNN variables (including all the required layer hyperparameters).
        <T_i>: ndarray with shape (examples, input_height, input_width, input_channels), it's the layers input.
        <lparams>: [UNUSED VARIABLE]
        <examples_multitude>: [UNUSED VARIABLE]

    Returns:
        <a_o>: ndarray with shape (examples, output_length). Output of the Flattening Layer.
    '''

    z_o = a_o = T_i.reshape((examples_multitude, layer.n_fc_o))

    z_o, a_o = layer.dropout(layer, z_o, a_o, examples_multitude, training)

    return z_o, a_o

## We go straight to L_f-1, the calculation of the error terms of L_f is captured by this procedure.
## layer: L_f-1, layer_next: L_f, layer_nnext: L_f+1
def backprop_transition(layer, layer_next, layer_nnext, delta_nnext, W_nnext, Phi, examples_multitude):

    sum_ = layer.tm_.tm.nan*layer.tm_.tm.ones((examples_multitude, layer_next.n_fc_o)).astype(layer.tm_.tm.float32)
    for i in range(examples_multitude):
        for n in range(layer_next.n_fc_o):
            sum = 0
            for n0 in range(layer_nnext.n_fc_o):
                sum += delta_nnext[i, n0] * W_nnext[n, n0]
            sum_[i, n] = sum

    sum__shape = layer.merge_tuple((examples_multitude,), layer.shape_o)
    delta = sum_.reshape(sum__shape)

    if layer.name == 'convolutional':
        delta *= layer.g.gradient_output(Phi)

    return delta

def backprop_transition_opt1(layer, layer_next, layer_nnext, delta_nnext, W_nnext, Phi, examples_multitude):

    delta_flat = delta_nnext @ W_nnext.T

    delta_shape = layer.merge_tuple((examples_multitude,), layer.shape_o)
    delta = delta_flat.reshape(delta_shape)

    if layer.name == 'convolutional':
        delta *= layer.g.gradient_output(Phi)

    return delta
