from math import ceil


## rate (input) fraction of neurons are dropped
def dropout(layer, linear_neurons_i, neurons_i, examples_multitude, training):

    if ((training == True) and (layer.drop_rate > 0)):

        final_shape = layer.merge_tuple((examples_multitude,), layer.shape_o)

        disposer_shape0 = layer.product_tuple(final_shape)

        disposer = layer.tm_.tm.ones(disposer_shape0)

        disposer[:ceil(layer.drop_rate*disposer_shape0)] = 0

        layer.tm_.tm.random.shuffle(disposer)

        disposer = layer.tm_.tm.reshape(disposer, final_shape)

        linear_neurons_o = linear_neurons_i * disposer
        neurons_o = neurons_i * disposer

    else:

        linear_neurons_o = linear_neurons_i
        neurons_o = neurons_i

    return linear_neurons_o, neurons_o


## Each neurons has p (input) probability of being fired.
# def dropout(layer, neurons_i, examples_multitude, training):
#
#     if ((training == True) and (drop_perc > 0)):
#
#         tup1 = (examples_multitude,)
#         tup2 = [1 for _ in range(len(layer.shape_i))]
#         disposer_tiling_shape = layer.merge_tuple(tup1, tup2)
#
#         disposer = layer.tm_.tm.tile((layer.tm_.tm.random.uniform(0, 1, layer.shape_i) > drop_perc).astype(layer.tm_.tm.float32)[layer.tm_.tm.newaxis, ...], disposer_tiling_shape)
#
#         neurons_o = neurons_i * disposer
#
#     else:
#
#         neurons_o = neurons_i
#
#     return neurons_o, neurons_o
