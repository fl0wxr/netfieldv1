from copy import deepcopy


def gradient_descent_methods_set(gradient_descent_method_name, tm_, lparam_layers, gradient_function, learning_args):

    if gradient_descent_method_name == 'regular_gd':
        return regular_gd(tm_, lparam_layers, gradient_function, learning_args)

    elif gradient_descent_method_name == 'regular_gd_with_momentum':
        return regular_gd_with_momentum(tm_, lparam_layers, gradient_function, learning_args)

    elif gradient_descent_method_name == 'adagrad_gd':
        return adagrad_gd(tm_, lparam_layers, gradient_function, learning_args)

    elif gradient_descent_method_name == 'rmsprop_gd':
        return rmsprop_gd(tm_, lparam_layers, gradient_function, learning_args)

    elif gradient_descent_method_name == 'adam_gd':
        return adam_gd(tm_, lparam_layers, gradient_function, learning_args)

class regular_gd:

    def __init__(self, tm_, lparam_layers, gradient_function, learning_args):

        self.tm_ = tm_
        self.gradient_function = gradient_function
        self.lparam_layers = lparam_layers
        self.learning_rate = learning_args['learning_rate']

    def update(self, gradients_partial_args, weights, biases, m_k):

        gradients_args = gradients_partial_args+[weights, biases]
        gradients, y_pred = self.gradient_function(*gradients_args)

        for l in self.lparam_layers:

            gradC_W, gradC_b = gradients[l]
            # import pdb; pdb.set_trace()
            weights[l] = weights[l] - self.learning_rate * self.tm_.tm.sum(gradC_W, axis=0)/m_k
            biases[l] = biases[l] - self.learning_rate * self.tm_.tm.sum(gradC_b, axis=0)/m_k

        return weights, biases, y_pred

class regular_gd_with_momentum:

    def __init__(self, tm_, lparam_layers, gradient_function, learning_args):

        self.tm_ = tm_
        self.gradient_function = gradient_function
        self.lparam_layers = lparam_layers
        self.learning_rate = learning_args['learning_rate']
        self.momentum_coeff = learning_args['momentum_coeff']
        self.momentum_W = [0 for l in range(max(self.lparam_layers)+1)]
        self.momentum_b = deepcopy(self.momentum_W)

    def update(self, gradients_partial_args, weights, biases, m_k):

        gradients_args = gradients_partial_args+[weights, biases]
        gradients, y_pred = self.gradient_function(*gradients_args)

        for l in self.lparam_layers:

            gradC_W, gradC_b = gradients[l]
            self.momentum_W[l] = self.momentum_coeff * self.momentum_W[l] - self.learning_rate * self.tm_.tm.sum(gradC_W, axis=0)/m_k
            self.momentum_b[l] = self.momentum_coeff * self.momentum_b[l] - self.learning_rate * self.tm_.tm.sum(gradC_b, axis=0)/m_k

            weights[l] += self.momentum_W[l]
            biases[l] += self.momentum_b[l]

        return weights, biases, y_pred

class adagrad_gd:

    def __init__(self, tm_, lparam_layers, gradient_function, learning_args):

        self.tm_ = tm_
        self.gradient_function = gradient_function
        self.lparam_layers = lparam_layers
        self.epsilon = learning_args['epsilon']
        self.c = learning_args['c']

        max_lparam_layers = max(self.lparam_layers)
        self.r_W = [0 for _ in range(max_lparam_layers+1)]
        self.r_b = [0 for _ in range(max_lparam_layers+1)]

    def update(self, gradients_partial_args, weights, biases, m_k):

        gradients_args = gradients_partial_args+[weights, biases]
        gradients, y_pred = self.gradient_function(*gradients_args)

        for l in self.lparam_layers:

            gradC_W, gradC_b = gradients[l]

            g_W = self.tm_.tm.sum(gradC_W, axis=0)/m_k
            g_b = self.tm_.tm.sum(gradC_b, axis=0)/m_k

            self.r_W[l] += g_W**2
            self.r_b[l] += g_b**2

            weights[l] = weights[l] - (self.epsilon/self.tm_.tm.sqrt(self.c+self.r_W[l])) * g_W
            biases[l] = biases[l] - (self.epsilon/self.tm_.tm.sqrt(self.c+self.r_b[l])) * g_b

        return weights, biases, y_pred

class rmsprop_gd:

    def __init__(self, tm_, lparam_layers, gradient_function, learning_args):

        self.tm_ = tm_
        self.gradient_function = gradient_function
        self.lparam_layers = lparam_layers
        self.epsilon = learning_args['epsilon']
        self.rho = learning_args['rho']
        self.c = learning_args['c']

        max_lparam_layers = max(self.lparam_layers)
        self.r_W = [0 for _ in range(max_lparam_layers+1)]
        self.r_b = [0 for _ in range(max_lparam_layers+1)]

    def update(self, gradients_partial_args, weights, biases, m_k):

        gradients_args = gradients_partial_args+[weights, biases]
        gradients, y_pred = self.gradient_function(*gradients_args)

        for l in self.lparam_layers:

            gradC_W, gradC_b = gradients[l]

            g_W = self.tm_.tm.sum(gradC_W, axis=0)/m_k
            g_b = self.tm_.tm.sum(gradC_b, axis=0)/m_k

            self.r_W[l] = self.rho*self.r_W[l] + (1-self.rho)*g_W**2
            self.r_b[l] = self.rho*self.r_b[l] + (1-self.rho)*g_b**2

            weights[l] = weights[l] - (self.epsilon/self.tm_.tm.sqrt(self.c+self.r_W[l])) * g_W
            biases[l] = biases[l] - (self.epsilon/self.tm_.tm.sqrt(self.c+self.r_b[l])) * g_b

        return weights, biases, y_pred

class adam_gd:

    def __init__(self, tm_, lparam_layers, gradient_function, learning_args):

        self.tm_ = tm_
        self.gradient_function = gradient_function
        self.lparam_layers = lparam_layers
        self.epsilon = learning_args['epsilon'] ## 0.001
        self.rho1 = learning_args['rho1'] ## 0.9
        self.rho2 = learning_args['rho2'] ## 0.999
        self.c = learning_args['c'] ## 10**(-8)

        max_lparam_layers = max(self.lparam_layers)

        self.s_W = [0 for _ in range(max_lparam_layers+1)]
        self.s_b = [0 for _ in range(max_lparam_layers+1)]

        self.r_W = [0 for _ in range(max_lparam_layers+1)]
        self.r_b = [0 for _ in range(max_lparam_layers+1)]

        self.t = 0

    def update(self, gradients_partial_args, weights, biases, m_k):

        gradients_args = gradients_partial_args+[weights, biases]
        gradients, y_pred = self.gradient_function(*gradients_args)

        for l in self.lparam_layers:

            self.t += 1

            gradC_W, gradC_b = gradients[l]

            g_W = self.tm_.tm.sum(gradC_W, axis=0)/m_k
            g_b = self.tm_.tm.sum(gradC_b, axis=0)/m_k

            self.s_W[l] = self.rho1*self.s_W[l] + (1-self.rho1)*g_W
            self.s_b[l] = self.rho1*self.s_b[l] + (1-self.rho1)*g_b

            self.r_W[l] = self.rho2*self.r_W[l] + (1-self.rho2)*(g_W**2)
            self.r_b[l] = self.rho2*self.r_b[l] + (1-self.rho2)*(g_b**2)

            s_W_ = self.s_W[l]/(1-self.rho1**self.t)
            s_b_ = self.s_b[l]/(1-self.rho1**self.t)

            r_W_ = self.r_W[l]/(1-self.rho2**self.t)
            r_b_ = self.r_b[l]/(1-self.rho2**self.t)

            weights[l] += -self.epsilon * s_W_/(self.tm_.tm.sqrt(r_W_)+self.c)
            biases[l] += -self.epsilon * s_b_/(self.tm_.tm.sqrt(r_b_)+self.c)

        return weights, biases, y_pred
