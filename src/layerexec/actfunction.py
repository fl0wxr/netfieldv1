def activation_function_set(tm_, function_name):

    if function_name == 'sigmoid':
        return sigmoid(tm_)

    elif function_name == 'relu':
        return relu(tm_)

    elif function_name in ('', 'identity'):
        return identity(tm_)

    elif function_name == 'softmax':
        return softmax(tm_)

    else:
        assert 0, 'Exception: Invalid activation function name.\nProblematic configuration value:\n' + function_name

class sigmoid:

    def __init__(self, tm_):
        self.tm_ = tm_

    def output(self, x):
        return 1/(1+self.tm_.tm.exp(-x))

    def gradient_output(self, x):
        output_ = self.output(x)
        return output_*(1-output_)

class relu:

    def __init__(self, tm_):
        self.tm_ = tm_

    def output(self, x):
        return self.tm_.tm.maximum(0, x)

    ## Pr(x=0)->0, and if it takes x=0 it doesn't actually hurt that much our CNN.
    def gradient_output(self, x):
        grad = self.tm_.tm.zeros(x.shape)
        grad[x>0] = 1

        return grad

class identity:

    def __init__(self, tm_):
        self.tm_ = tm_

    def output(self, x):
        return x

    def gradient_output(self, x):
        return self.tm_.tm.ones(x.shape)

class softmax:

    def __init__(self, tm_):
        self.tm_ = tm_

    def output(self, x):

        sum_ = self.tm_.tm.sum(self.tm_.tm.exp(x), axis=1, keepdims=True)
        output = self.tm_.tm.exp(x)/sum_

        return output

    def gradient_output(self, x):
        pass
