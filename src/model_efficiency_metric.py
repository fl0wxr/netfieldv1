def cost_function_set(tm_, function_name):

    if function_name == 'categorical_cross_entropy':
        return categorical_cross_entropy(tm_)

    else:
        assert 0, 'Exception: Invalid cost function name.'

class categorical_cross_entropy:

    def __init__(self, tm_):
        '''
        Inputs:
            <tm_>: ndarray, provides tensor modules and offers a set of memory management tools.
        '''

        self.tm_ = tm_

    def output(self, y_pred, y, examples_multitude):
        '''
        Categorical cross entropy output

        Description:
            It returns the output of the cross entropy function

        Inputs:
            <y_pred>: ndarray of shape (examples_multitude, number_of_classes), the CNNs feedforward output.
            <y>: ndarray of shape (examples_multitude, number_of_classes), the ground truth in one hot encoding format.
            <examples_multitude>: int.

        Returns:
            <functions_output>: float, returns the categorical cross entropy functions output.
        '''

        return - self.tm_.tm.sum\
        (
            y * self.tm_.tm.log(y_pred)
        )\
        /examples_multitude

def accuracy(tm_, y_pred, y, examples_multitude):
    '''
    Inputs:
        <tm_>: ndarray, provides tensor modules and offers a set of memory management tools.
        <y_pred>: ndarray of shape (examples_multitude, classes_multitude), the output of the CNNs prediction.
        <y>: ndarray of shape (examples_multitude, classes_multitude), the ground truth for each given example in a one hot encoding format
        <examples_multitude>: int, the size of the provided example set

    Returns:
        <accuracy_>: float, based on
            ACC := (# of right Predictions in the provided example set)/(total # of examples in the provided example set)
    '''

    y_pred = tm_.tm.array(tm_.tm.transpose(tm_.tm.amax(tm_.tm.transpose(y_pred), axis=0) == tm_.tm.transpose(y_pred)), dtype=tm_.tm.int64)
    right_predictions = tm_.tm.sum(tm_.tm.array([bool((y_pred[i] == y[i]).all()) for i in range(examples_multitude)]))
    accuracy_ = right_predictions/examples_multitude

    return accuracy_
