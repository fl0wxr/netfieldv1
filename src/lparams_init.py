def lparams_init(convnet):
    ## Across the first index there are the coefficient learnable parameters or sometimes called weights. Across the second index the bias terms are stored.

    # np.random.seed(0)

    lparams = \
    [
        [convnet.tm_.tm.array([]) for l in range(convnet.L)],
        [convnet.tm_.tm.array([]) for l in range(convnet.L)]
    ]

    for l in range(convnet.L):
        if convnet.layers_obj[l].weights_shape != (0,):
            ## Learnable parameter initialization
            in_neur = convnet.layers_obj[l].input_neurons_multitude
            out_neur = convnet.layers_obj[l].output_neurons_multitude

            boundary = convnet.tm_.tm.sqrt( 6/(in_neur+out_neur) )

            lparams[0][l] = convnet.tm_.tm.random.uniform(-boundary, boundary, convnet.layers_obj[l].weights_shape)
            lparams[1][l] = convnet.tm_.tm.zeros(convnet.layers_obj[l].biases_shape)

            ## ! Dummy lparams: Begin

            # idx_prod = 1
            # for _ in range(len(convnet.layers_obj[l].weights_shape)):
            #     idx_prod *= convnet.layers_obj[l].weights_shape[_]

            # lparams[0][l] = convnet.tm_.tm.arange(idx_prod).reshape(convnet.layers_obj[l].weights_shape).astype(convnet.tm_.tm.float32)
            # lparams[1][l] = convnet.tm_.tm.zeros(convnet.layers_obj[l].biases_shape).astype(convnet.tm_.tm.float32)

            ## ! Dummy lparams: End

            # lparams[0][l] = convnet.tm_.tm.around(lparams[0][l], decimals=2)*100
            # lparams[1][l] = convnet.tm_.tm.around(lparams[1][l], decimals=2)*100

    print('New learnable parameters have been generated.')

    return lparams
