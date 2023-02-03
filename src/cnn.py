from math import floor, ceil
import pandas as pd
import time
import sys
import os
import matplotlib.pyplot as plt
from copy import deepcopy

import model_efficiency_metric
from layerexec import actfunction, convolutional, maxpooling, avgpooling, dropout, flattening, fullyconnected, pad
from lparams_init import lparams_init
import gradient_descent
import cnnprint


def do_nothing(*args):

    pass

def initialize_data(tm_, X, y, normalize=True, shuffle=False):

    examples_multitude = X.shape[0]

    if normalize:
        X = normalize_features(X)
    else:
        print('WARNING: Normalization is turned off.')

    if shuffle:
        shuffled_example_idcs = tm_.tm.random.permutation(examples_multitude)
        X = X[shuffled_example_idcs, ...]
        y = y[shuffled_example_idcs, ...]

    return X, y

def normalize_features(X):

    return X / 255. - .5

def mkdir_(loc_lparams_save_dir):
    '''
    Description:
        Creates a directory with a given name, assuming that a directory with the same name does not exist.
    '''

    if loc_lparams_save_dir != '' and not os.path.isdir(loc_lparams_save_dir):
        try:
            os.mkdir(os.path.abspath(loc_lparams_save_dir))
        except OSError:
            print ("Directory creation %s failed" % loc_lparams_save_dir)
        else:
            print ("Successfully created directory %s " % loc_lparams_save_dir)

def save_lparams(tm_, loc_lparams_save_dir, lparams):

    if loc_lparams_save_dir != '':
        filename = time.strftime(loc_lparams_save_dir+"/lpar_%d-%b-%Y_%H-%M-%S", time.localtime())
        tm_.np.save(filename, lparams)
        print('\nThe learnable parameters are now backed up.\n')

def write_metrics(epoch, metrics_filename, current_metrics):

    df = pd.DataFrame(current_metrics)

    if epoch > 0:
        df.to_csv(metrics_filename, mode='a', header=False)

    elif epoch == 0:
        df.to_csv(metrics_filename, mode='w')

class cnn:

    import numpy as np; np.set_printoptions(precision=20, suppress=True, linewidth=400, threshold=sys.maxsize)

    def __init__(self, tm_, normalize_input, class_names, layers_hparams, loc_lparams_load_file, loc_lparams_save_dir):
        '''
        Description:
            This method is the initialization basis for both the classification and training procedure.

        Input:
            <self>: class object, provides access to important variables about the CNNs state.
            <tm_>: class object, it offers a configurable use of various tensor manipulation frameworks, whilst providing the corresponding module of interest.
            <normalize_input>: bool, depending on this the input features either get normalized or not.
            <class_names>: list, holds the labels in strings corresponding to each of the classes.
            <layers_hparams>: list, holds all the hyperaparameters owned by the CNNs layers.
            <loc_lparams_load_file>: str, holds the relative path of the learnable parameters to be loaded, if it's equal to the empty string then cnn generates a new set of learnable parameters.
            <loc_lparams_save_dir>: str, the directory where the learnable parameters are to be backed up, it acts as the first path of the name.
        '''

        self.tm_ = tm_
        self.normalize_input = normalize_input
        self.class_names = class_names
        self.classes_multitude = len(class_names)
        self.layers_obj = []
        self.loc_lparams_load_file = loc_lparams_load_file
        self.loc_lparams_save_dir = loc_lparams_save_dir

        self.list_of_layers_with_learnable_parameters = \
        [
            'convolutional',
            'fully_connected',
            'softmax_output'
        ]

        ## Total number of learnable parameters in the entire CNN.
        self.all_lparams = 0

        ## Storing layer objects, while printing the CNN table.
        for l in range(len(layers_hparams)):

            if layers_hparams[l][0] != 'input':
                self.layers_obj.append\
                (
                    layer(self.tm_, l, layers_hparams[l][0], layers_hparams[l][1], self.classes_multitude, self.layers_obj[l-1].shape_o)
                )
            else:
                self.layers_obj.append\
                (
                    layer(self.tm_, l, layers_hparams[l][0], layers_hparams[l][1], self.classes_multitude)
                )

            ## Storing the Flattening Layer
            if layers_hparams[l][0] == 'flattening':
                self.L_f = l

            self.all_lparams += self.layers_obj[l].tot_lparams

        ## Total number of features.
        self.all_features = self.layers_obj[0].tot_features

        cnnprint.disp_handlparams(self.all_lparams, self.all_features)

        self.L = len(self.layers_obj)

        ## Create the directory in which the learnable parameters will be saved.
        mkdir_(self.loc_lparams_save_dir)

        if self.loc_lparams_load_file == '':
            ## Generating new learnable parameters.
            [self.weights, self.biases] = lparams_init(self)
            save_lparams(self.tm_, self.loc_lparams_save_dir, [self.weights, self.biases])
            # print('WARNING: DEBUGGING MODE PREVENTS BACKUPS')
        else:
            ## Load an already existing set of learnable parameters.
            self.weights, self.biases = \
                self.tm_.np.load(self.loc_lparams_load_file, allow_pickle=True)

            # self.weights = [self.tm_.tm.array(self.weights[l]) for l in range(self.L)]
            # self.biases = [self.tm_.tm.array(self.biases[l]) for l in range(self.L)]

            print('Learnable parameters loaded.\n')

            self.check_lparams_shapes()

        ## Filename where the metrics are to be saved.
        if self.loc_lparams_save_dir != '':
        #     metrics_filename = self.loc_lparams_save_dir+'/metrics_history.csv'
        # else:
            self.metrics_filename = './lparams/metrics_history.csv'

    def classify_store(self, feature_data_for_classification):
        '''
        Description:
            Acts as an initializer for a classification procedure.

        Inputs:
            <feature_data>: ndarray, holds the the image or set of images which needs to be classified.
        '''

        self.feature_data_for_classification = feature_data_for_classification

        if self.normalize_input == True:
            self.feature_data_for_classification = normalize_features(self.feature_data_for_classification)

    def classify(self):
        '''
        Description:
            This method classifies its input.

        Inputs:
            <self>: object, holds the hyperparameters needed to complete a feedforward run for a given CNN configuration.
                <self.feature_data>: ndarray of shape (# of images, image_height, image_width, image_channels).
        '''

        for i in range(self.feature_data_for_classification.shape[0]):

            ## Feedforward run
            predictions = self.feedforward\
            (
                self.feature_data_for_classification[i:i+1],
                1,
                self.weights,
                self.biases,
                only_output_layer=True
            )

            print('prediction:')

            for cl in range(predictions.shape[1]):
                print('%s: %.3f'%(self.class_names[cl], predictions[0][cl]))

            plt.imshow(self.tm_.tm.asnumpy(self.feature_data_for_classification[i,...,0]))
            plt.show()

            print()

    def train_store(self, training_config, training_set, val_set):
        '''
        Description:
            Responsible for the initialization of the training hyperparameters, minibatches and the dataset segmentation into training - validation sets as a preliminary for a potential training sequence.

        Inputs:
            <self>: object, holds the hyperparameters required to complete a training run, which is also used to store some variables relevant to the training sequence.
            <training_config>: dict, holds all the training hyperparameters.
            <training_set>: tuple of length (2,).
                <X_te>: list, holds the relative paths of each of the training features in str.
                <y_te>: list, holds the relative paths of each of the training target variables in str.
            <val_set>: tuple of length (2,).
                <X_val>: list, holds the relative paths of each of the cross validation features in str.
                <y_val>: list, holds the relative paths of each of the cross validation target variables in str.
        '''

        self.training_config = training_config
        self.shuffle_input = bool(self.training_config['shuffle_input'])

        self.X_te, self.y_te = training_set

        ## Normalize training feature set !
        self.X_te, self.y_te = initialize_data(self.tm_, self.X_te, self.y_te, shuffle=self.shuffle_input, normalize=self.normalize_input)

        self.te_set_size = len(self.X_te)

        self.X_val, self.y_val = val_set

        ## Normalize cross validation feature set !
        self.X_val, self.y_val = initialize_data(self.tm_, self.X_val, self.y_val, shuffle=self.shuffle_input, normalize=self.normalize_input)

        self.val_set_size = len(self.X_val)

        ## ! Hyperparameter initialization: Begin

        ## List which will be filled ONLY with layer indices that include learnable parameters.
        self.lparam_layer = []
        for l in range(self.L):
            if self.layers_obj[l].name in self.list_of_layers_with_learnable_parameters:
                self.lparam_layer.append(l)

        self.epochs = self.training_config['epochs']
        self.gradient_descent = gradient_descent.gradient_descent_methods_set\
        (
            self.training_config['learning_method']['name'],
            self.tm_,
            self.lparam_layer,
            self.backprop_analytical_gradients,
            self.training_config['learning_method']['args']
        )
        self.backup_period = self.training_config['backup_period']
        self.cost_function_name = self.training_config['cost_function']
        self.cost_function = model_efficiency_metric.cost_function_set(self.tm_, self.cost_function_name)

        ## ! Minibatch segmentation: Begin

        wanted_minibatch_size = self.training_config['minibatch_size']
        assert wanted_minibatch_size <= self.te_set_size, 'Exception: Oversized minibatch.'
        self.te_minibatches = ceil(self.te_set_size/wanted_minibatch_size)
        self.te_minibatch_size = self.correct_final_minibatch_size(wanted_minibatch_size, self.te_set_size, self.te_minibatches)
        i_init = self.tm_.tm.concatenate((self.tm_.tm.array([0]), self.tm_.tm.cumsum(self.te_minibatch_size)[:-1]))
        self.te_minibatch_idx_set = \
        [
            list \
            (
                range \
                (
                    int(i_init[te_minibatch]),
                    int(i_init[te_minibatch]+self.te_minibatch_size[te_minibatch])
                )
            )
            for te_minibatch in range(self.te_minibatches)
        ]

        ## The val set minibatch segmentation will be used for a more memory economical approach for the calculation of the error metrics per epoch
        self.val_minibatches = ceil(self.val_set_size/min(wanted_minibatch_size, self.val_set_size))
        self.val_minibatch_size = self.correct_final_minibatch_size(wanted_minibatch_size, self.val_set_size, self.val_minibatches)
        i_init = self.tm_.tm.concatenate((self.tm_.tm.array([0]), self.tm_.tm.cumsum(self.val_minibatch_size)[:-1]))
        self.val_minibatch_idx_set = \
        [
            list \
            (
                range \
                (
                    int(i_init[val_minibatch]),
                    int(i_init[val_minibatch]+self.val_minibatch_size[val_minibatch])
                )
            )
            for val_minibatch in range(self.val_minibatches)
        ]

        self.tm_.memory_deall()

        ## ! Minibatch segmentation: End

        ## ! Hyperparameter initialization: End

    def check_lparams_shapes(self):
        for l in range(0, self.L):

            assert \
                (
                    (self.layers_obj[l].weights_shape == self.weights[l].shape) and (self.layers_obj[l].biases_shape == self.biases[l].shape)
                ), \
                'Exception: Learnable parameter array shape is different from the shape of the learnable parameter input shape on layer {}.'.format(l)

    def correct_final_minibatch_size(self, wanted_minibatch_size, set_size, minibatches):
        final_minibatch_size = \
            int \
            (
                wanted_minibatch_size * \
                (
                    set_size/wanted_minibatch_size - floor(set_size/wanted_minibatch_size)
                )
            )
        minibatch_size = [wanted_minibatch_size for k in range(minibatches)]
        if final_minibatch_size != 0:
            minibatch_size[minibatches-1] = final_minibatch_size

        return minibatch_size

    def train_cnn(self):

        assert self.cost_function_name == 'categorical_cross_entropy', 'Exception: Cost function not available.'
        self.minibatch_learning(self.backprop_analytical_gradients)

    def minibatch_learning(self, gradient_derivation_method):

        acc_val_max = -self.tm_.tm.inf

        t0 = time.time()

        for epoch in range(self.epochs):

            # if epoch == 0:
            #
            #     loss_val, acc_val = self.model_efficiency_metrics\
            #     (
            #         X=self.X_val,
            #         y=self.y_val,
            #         minibatch_idx_set=self.val_minibatch_idx_set,
            #         minibatch_size=self.val_minibatch_size,
            #         minibatches=self.val_minibatches,
            #         set_size=self.val_set_size
            #     )
            #
            #     print('val_loss (before training): ', loss_val)
            #     print('val_acc (before training):  ', acc_val)

            epoch_start_time = time.time()

            for te_minibatch in range(self.te_minibatches):

                ## ! Initializing: Begin

                ## Collecting the initial training examples as the input layers neuron values.
                X_te_minibatch = self.X_te[ self.te_minibatch_idx_set[te_minibatch] ]
                y_te_minibatch = self.y_te[ self.te_minibatch_idx_set[te_minibatch] ]

                ## ! Initializing: End

                gradients_partial_args = [X_te_minibatch, y_te_minibatch, self.te_minibatch_size[te_minibatch]]

                ## It can prove useful to execute the BP algorithm on a GD routine for some occassions like GD with Nesterov momentum
                self.weights, self.biases, y_pred_te_minibatch = self.gradient_descent.update(gradients_partial_args, self.weights, self.biases, self.te_minibatch_size[te_minibatch])

                ## ! Error metrics I: Begin

                ## Training set error metrics.
                loss_te_minibatch = self.cost_function.output(y_pred_te_minibatch, y_te_minibatch, self.te_minibatch_size[te_minibatch])
                acc_te_minibatch = model_efficiency_metric.accuracy(self.tm_, y_pred_te_minibatch, y_te_minibatch, self.te_minibatch_size[te_minibatch])

                ## Displaying error metrics
                cnnprint.disp_training_minibatch_progress(minibatch = te_minibatch, minibatches = self.te_minibatches, metrics = (loss_te_minibatch, acc_te_minibatch))

                ## ! Error metrics I: End

            ## Saving the latest learnable parameters
            if (epoch+1)%self.backup_period == 0:
                save_lparams(self.tm_, self.loc_lparams_save_dir, [self.weights, self.biases])

            ## ! Error metrics II: Begin

            loss_val, acc_val = self.model_efficiency_metrics\
            (
                X=self.X_val,
                y=self.y_val,
                minibatch_idx_set=self.val_minibatch_idx_set,
                minibatch_size=self.val_minibatch_size,
                minibatches=self.val_minibatches,
                set_size=self.val_set_size
            )

            loss_te, acc_te = self.model_efficiency_metrics\
            (
                X=self.X_te,
                y=self.y_te,
                minibatch_idx_set=self.te_minibatch_idx_set,
                minibatch_size=self.te_minibatch_size,
                minibatches=self.te_minibatches,
                set_size=self.te_set_size
            )

            cnnprint.disp_epoch_progress\
            (
                epoch = epoch,
                epochs = self.epochs,
                past_max_acc_val = acc_val_max,
                error_metrics = \
                {
                    'acc': (acc_te, acc_val),
                    'loss': (loss_te, loss_val)
                },
                tot_time = round(float(time.time() - epoch_start_time))
            )

            acc_val_max = max([acc_val, acc_val_max])

            ## Write metrics to disk
            write_metrics\
            (
                epoch = epoch,
                metrics_filename = self.metrics_filename,
                current_metrics = \
                {
                    'training_acc': [.0],
                    'val_acc': [acc_val],
                    'training_loss': [.0],
                    'val_loss': [loss_val]
                }
            )

            ## ! Error metrics II: End

        # print('WEIGHTS: \n', self.weights[-2])#.shape)
        # print('BIASES:  \n', self.biases[-2])#.shape)
        # print('WEIGHTS: \n', self.weights[-1])#.shape)
        # print('BIASES:  \n', self.biases[-1])#.shape)
        # if self.epochs == 0: exit()
            print('val_loss: ', loss_val)
            print('val_acc:  ', acc_val)
            print('Training time: %f s'%(time.time()-t0))

    def model_efficiency_metrics(self, X, y, minibatch_idx_set, minibatch_size, minibatches, set_size):

        y_pred = []

        for minibatch in range(minibatches):

            X_minibatch = X[ minibatch_idx_set[minibatch] ]
            y_pred_minibatch = self.feedforward(X_minibatch, minibatch_size[minibatch], self.weights, self.biases, only_output_layer=True)
            y_pred.append(y_pred_minibatch)

        y_pred = self.tm_.tm.concatenate(y_pred, axis=0)

        ## Training set error metrics.
        loss = self.cost_function.output(y_pred, y, set_size)
        acc = model_efficiency_metric.accuracy(self.tm_, y_pred, y, set_size)

        return loss, acc

    def feedforward(self, features, examples_multitude, weights, biases, return_linear=False, only_output_layer=False, training=False):

        linear_neurons = [None for l in range(self.L)]
        neurons = [None for l in range(self.L)]

        neurons[0] = features

        for l in range(1, self.L):

            [linear_neurons[l], neurons[l]] = self.layers_obj[l].layer_feedforward\
            (
                self.layers_obj[l],
                neurons[l-1],
                examples_multitude,
                training=training,
                lparams=(weights[l], biases[l])
            )

        if only_output_layer:
            neurons_ = neurons[self.L-1]
            linear_neurons_ = linear_neurons[self.L-1]
        else:
            neurons_ = neurons
            linear_neurons_ = linear_neurons

        if return_linear:
            asked_neurons = (linear_neurons_, neurons_)
        else:
            asked_neurons = neurons_

        return asked_neurons

    def backprop_analytical_gradients(self, features, target_variables, examples_multitude, weights, biases):
        """
        Description:
            This method performs the backpropagation algorithm to calculate the final gradients.
        """

        ## ! Initializing: Begin

        training_neurons_minibatch = [None for l in range(self.L)]
        linear_training_neurons_minibatch = [None for l in range(self.L)]
        gradients = [None for l in range(self.L)]
        delta = [None for l in range(self.L)]

        ## ! Initializing: End

        ## ! Feedforward: Begin

        linear_training_neurons_minibatch, training_neurons_minibatch = self.feedforward\
        (
            features,
            examples_multitude,
            weights,
            biases,
            return_linear=True,
            training=True
        )

        ## ! Feedforward: End

        ## ! Errors: Begin

        ## Output layer
        delta[self.L-1] = self.layers_obj[self.L-1].layer_backprop\
        (
            training_neurons_minibatch[self.L-1],
            target_variables
        )

        ## FC Part
        for l in range(self.L-2, self.L_f, -1):

            if l < self.lparam_layer[0]:
                break
            else:
                delta[l] = self.layers_obj[l].layer_backprop\
                (
                    self.layers_obj[l],
                    self.layers_obj[l+1],
                    delta[l+1],
                    weights[l+1],
                    linear_training_neurons_minibatch[l],
                    examples_multitude
                )

        if self.L_f < self.lparam_layer[0]:
            pass
        else:
            ## Transition from the FC to the Conv. Part.
            delta[self.L_f-1] = self.layers_obj[self.L_f].layer_backprop\
            (
                self.layers_obj[self.L_f-1],
                self.layers_obj[self.L_f],
                self.layers_obj[self.L_f+1],
                delta[self.L_f+1],
                weights[self.L_f+1],
                linear_training_neurons_minibatch[self.L_f-1],
                examples_multitude
            )

        ## Conv. Part
        for l in range(self.L_f-2, 0, -1):

            if l < self.lparam_layer[0]:
                break
            else:
                # This executes relative to what the next layers backpropagation function is assigned  to.

                layer_is_convolutional = self.layers_obj[l].name == 'convolutional'

                ## Using self.layers_obj[l+1] instead of self.layers_obj[l] actually helps a lot as most of the backprop function involves operations depending on layer l+1, and it kinda makes sense that the error of l heavily depends on l+1 as its very concept relies on capturing the effect that l has on l+1.

                delta[l] = self.layers_obj[l+1].layer_backprop\
                (
                    self.layers_obj[l],
                    self.layers_obj[l+1],
                    delta[l+1],
                    examples_multitude,
                    weights[l+1],
                    linear_training_neurons_minibatch[l],
                    training_neurons_minibatch[l],
                    layer_is_convolutional
                )

        ## ! Errors: End

        ## ! Gradients: Begin

        for l in range(1, self.L):

            gradients[l] = self.layers_obj[l].layer_gradient\
            (
                self.layers_obj[l],
                self.layers_obj[l-1],
                delta[l],
                examples_multitude,
                training_neurons_minibatch[l-1]
            )

        ## ! Gradients: End

        return gradients, training_neurons_minibatch[-1]

    def slope_of_C_per_lparam(self, features, target_variables, weights, biases):
        '''
        Description:
            This linearizes the cost function in a small enough ball to calculate its slopes across each scalar element of W and b. These slopes can be used to approximate the cost functions gradients with respect to the learnable parameters for debugging purposes. Although this simple method produces a fine approximation of the final gradient, it's unfortunately incredibly slow and/or memory demanding, as we need to calculate the cost functions Jacobians which translates to an astronomical amount of memory usage for any vectorization attempt, and our options are limited in looping through each learnable parameter separately, so there's not much vectorization potential to be invoked. As a result this method performs 4 feedforward runs for each learnable parameter scalar! For that reason any attempt to bring it to a competitive ratio of (execution time)/(memory usage) compared to the backpropagation algorithm is futile.

        Inputs:
            <self>: cnn object.
            <features>: ndarray of shape (input_height, input_width, input_channels).
            <target_variables>: ndarray of shape (1).
            <weights>: list, contains the weights of each layer with learnable parameters.
            <biases>: list, contains the biases of each layer with learnable parameters.

        Returns:
            <slopes>: list of length 2, containing the slopes of the cost function with respect to the learnable parameters slopes of an examples for each layer
                <slopeC_W>: list.
                <slopeC_b>: list.
        '''

        def slopeC(self, plus, minus):

            W1, b1 = plus
            W2, b2 = minus

            y_pred_W_plus = self.feedforward(features, 1, W1, b1, only_output_layer=True)
            y_pred_W_minus = self.feedforward(features, 1, W2, b2, only_output_layer=True)

            C_plus = self.cost_function.output(y_pred_W_plus, self.y_te, 1)
            C_minus = self.cost_function.output(y_pred_W_minus, self.y_te, 1)

            return (C_plus-C_minus)/(2*epsilon)

        ## ! Initialization: Begin

        epsilon = 10**(-7)

        ## Format [[W_l0, b_l0], ..., [W_out, b_out]]
        slopes = [None for l in range(self.L)]

        ## ! Initialization: End

        ## Forming the learnable parameters
        for l in self.lparam_layer:

            slopes[l] = [self.tm_.tm.nan*self.tm_.tm.ones(self.layers_obj[l].weights_shape)[self.tm_.tm.newaxis,...], self.tm_.tm.nan*self.tm_.tm.ones(self.layers_obj[l].biases_shape)[self.tm_.tm.newaxis,...]]

            if self.layers_obj[l].name == 'convolutional':

                for c in range(self.layers_obj[l].n_c_o):
                    for h0 in range(self.layers_obj[l].f):
                        for w0 in range(self.layers_obj[l].f):
                            for c_prev in range(self.layers_obj[l].n_c_i):

                                W_plus = deepcopy(self.weights)
                                W_minus = deepcopy(W_plus)

                                W_plus[l][c, h0, w0, c_prev] += epsilon
                                W_minus[l][c, h0, w0, c_prev] -= epsilon

                                ## slope with respect to W
                                slopes[l][0][0, c, h0, w0, c_prev] = slopeC(self, plus=(W_plus, self.biases), minus=(W_minus, self.biases))

                del W_plus, W_minus

                for h in range(self.layers_obj[l].n_h_o):
                    for w in range(self.layers_obj[l].n_w_o):
                        for c in range(self.layers_obj[l].n_c_o):

                            b_plus = deepcopy(self.biases)
                            b_minus = deepcopy(b_plus)

                            b_plus[l][h, w, c] += epsilon
                            b_minus[l][h, w, c] -= epsilon

                            ## slope with respect to b
                            slopes[l][1][0, h, w, c] = slopeC(self, plus=(self.weights, b_plus), minus=(self.weights, b_minus))

                del b_plus, b_minus

            elif self.layers_obj[l].name in ['fully_connected', 'softmax_output']:

                for n in range(self.layers_obj[l].n_fc_o):
                    for n0 in range(self.layers_obj[l].n_fc_i):

                        W_plus = deepcopy(self.weights)
                        W_minus = deepcopy(W_plus)

                        W_plus[l][n0, n] += epsilon
                        W_minus[l][n0, n] -= epsilon

                        ## slope with respect to W
                        slopes[l][0][0, n0, n] = slopeC(self, plus=(W_plus, self.biases), minus=(W_minus, self.biases))

                for n in range(self.layers_obj[l].n_fc_o):

                    b_plus = deepcopy(self.biases)
                    b_minus = deepcopy(b_plus)

                    b_plus[l][n] += epsilon
                    b_minus[l][n] -= epsilon

                    ## slope with respect to b
                    slopes[l][1][0, n] = slopeC(self, plus=(self.weights, b_plus), minus=(self.weights, b_minus))

        return slopes

    def gradient_checking(self):

        ## Uncertain method
        gradients = self.backprop_analytical_gradients(self.X_te[0:1],  self.y_te[0:1], 1, self.weights, self.biases) # carefull as its first shape axis contain the examples

        ## Approximation using a simpler method
        slopes = self.slope_of_C_per_lparam(self.X_te[0:1],  self.y_te[0:1], self.weights, self.biases)

        gradients_ = self.tm_.tm.array([])
        slopes_ = self.tm_.tm.array([])
        for l in self.lparam_layer:
            gradients_ = self.tm_.tm.append(gradients_, gradients[l][0])
            gradients_ = self.tm_.tm.append(gradients_, gradients[l][1])
            slopes_ = self.tm_.tm.append(slopes_, slopes[l][0])
            slopes_ = self.tm_.tm.append(slopes_, slopes[l][1])

            self.tm_.memory_deall()

        norm2 = self.tm_.tm.linalg.norm(gradients_-slopes_)
        normalization_denominator = self.tm_.tm.linalg.norm(gradients_)+self.tm_.tm.linalg.norm(slopes_)

        normalized_norm2 = norm2/normalization_denominator

        ## According to article: https://towardsdatascience.com/how-to-debug-a-neural-network-with-gradient-checking-41deec0357a9 this has to be on the same or lower scale relative to epsilon
        print('Backprop error: {}'.format(normalized_norm2))


class layer:

    def __init__(self, tm_, layer_index, layer_name, layer_hparams, classes_multitude, *shape_i):

        self.tm_ = tm_
        self.layer_index = layer_index
        self.name = layer_name
        self.hparams = layer_hparams
        self.classes_multitude = classes_multitude
        self.dropout = dropout.dropout

        if 'dropout_perc' in self.hparams.keys():
            self.drop_rate = self.hparams['dropout_rate']
        else:
            self.drop_rate = 0

        ## This results to be a different than (0,) if the layer is consisted by learnable parameters.
        self.weights_shape = (0,)
        self.biases_shape = (0,)

        ## Default number of learnable parameters. This value becomes positive if a layer includes at least one learnable parameters.
        self.tot_lparams = 0

        ## This excludes the input layer
        if shape_i != ():
            self.shape_i = shape_i[0]
            self.input_neurons_multitude = self.product_tuple(self.shape_i)

        self.store_hparams()

    def merge_tuple(self, tup1, tup2):

        list_ = []
        for _ in range(len(tup1)):
            list_.append(tup1[_])

        for _ in range(len(tup2)):
            list_.append(tup2[_])

        tup_ = tuple(list_)

        return tup_

    def product_tuple(self, tup):

        prod = 1
        for _ in range(len(tup)):
            prod *= tup[_]

        return prod

    def store_hparams(self):
        if self.name == 'input':
            self.input_store()
        elif self.name == 'convolutional':
            self.convolutional_store()
        elif self.name == 'maxpooling':
            self.maxpooling_store()
        elif self.name == 'avgpooling':
            self.avgpooling_store()
        elif self.name == 'flattening':
            self.flattening_store()
        elif self.name == 'fully_connected':
            self.fullyconnected_store()
        elif self.name == 'softmax_output':
            self.output_store()

    def scan_output_length(self, input_length):
        return floor((input_length+2*self.p-self.f)/self.s + 1)

    ## ! Storing: Begin

    def input_store(self):
        self.n_h_i = self.n_h_o = self.hparams['input_height']
        self.n_w_i = self.n_w_o = self.hparams['input_width']
        self.n_c_i = self.n_c_o = self.hparams['input_channels']
        self.shape_i = self.shape_o = (self.n_h_o, self.n_w_o, self.n_c_o)

        self.tot_features = self.input_neurons_multitude = self.product_tuple(self.shape_o)

        self.printhparams = cnnprint.inputlayerprint(self.shape_o)

    def convolutional_store(self):
        self.as_strided = self.tm_.as_strided
        self.pad = pad.pad(self.tm_)

        self.f = self.hparams['filter_size']
        self.p = self.hparams['padding']
        self.s = self.hparams['stride']
        self.g = actfunction.activation_function_set(self.tm_, self.hparams['activation_function'])
        self.n_h_i = self.shape_i[0]
        self.n_w_i = self.shape_i[1]
        self.n_c_i = self.shape_i[2]
        self.n_h_o = self.scan_output_length(self.n_h_i)
        self.n_w_o = self.scan_output_length(self.n_w_i)
        self.n_c_o = self.hparams['output_channels']
        self.shape_o = (self.n_h_o, self.n_w_o, self.n_c_o)
        self.output_neurons_multitude = self.product_tuple(self.shape_o)
        self.weights_shape = (self.n_c_o, self.f, self.f, self.n_c_i)
        self.biases_shape = (self.n_h_o, self.n_w_o, self.n_c_o)

        self.tot_lparams = self.product_tuple(self.weights_shape) + self.product_tuple(self.biases_shape)

        cnnprint.convlayerprint((self.n_c_o, self.f, self.p, self.s, self.hparams['activation_function']), self.tot_lparams, self.layer_index, self.shape_i, self.shape_o)

        self.layer_feedforward = convolutional.feedforward_opt1
        self.layer_backprop = convolutional.backprop_opt1# _naive _opt1
        self.layer_gradient = convolutional.gradient_opt1

    def maxpooling_store(self):
        self.as_strided = self.tm_.as_strided
        self.pad = pad.pad(self.tm_)

        self.f = self.hparams['size']
        self.p = self.hparams['padding']
        self.s = self.hparams['stride']
        self.n_h_i = self.shape_i[0]
        self.n_w_i = self.shape_i[1]
        self.n_c_i = self.shape_i[2]
        self.n_h_o = self.scan_output_length(self.n_h_i)
        self.n_w_o = self.scan_output_length(self.n_w_i)
        self.n_c_o = self.n_c_i
        self.shape_o = (self.n_h_o, self.n_w_o, self.n_c_o)
        self.output_neurons_multitude = self.product_tuple(self.shape_o)

        cnnprint.maxplayerprint((self.f, self.p, self.s), self.layer_index, self.shape_i, self.shape_o)

        self.layer_feedforward = maxpooling.feedforward_opt1
        self.layer_backprop = maxpooling.backprop_opt1#_naive
        self.layer_gradient = do_nothing

    def avgpooling_store(self):
        self.as_strided = self.tm_.as_strided
        self.pad = pad.pad(self.tm_)

        self.f = self.hparams['size']
        self.p = self.hparams['padding']
        self.s = self.hparams['stride']
        self.n_h_i = self.shape_i[0]
        self.n_w_i = self.shape_i[1]
        self.n_c_i = self.shape_i[2]
        self.n_h_o = self.scan_output_length(self.n_h_i)
        self.n_w_o = self.scan_output_length(self.n_w_i)
        self.n_c_o = self.n_c_i
        self.shape_o = (self.n_h_o, self.n_w_o, self.n_c_o)
        self.output_neurons_multitude = self.product_tuple(self.shape_o)

        cnnprint.avgplayerprint((self.f, self.p, self.s), self.layer_index, self.shape_i, self.shape_o)

        self.layer_feedforward = avgpooling.feedforward_opt1
        self.layer_backprop = avgpooling.backprop_opt1#_naive#_opt1
        self.layer_gradient = do_nothing

    def flattening_store(self):
        self.n_h_i = self.shape_i[0]
        self.n_w_i = self.shape_i[1]
        self.n_c_i = self.shape_i[2]
        self.n_fc_o = self.n_h_i*self.n_w_i*self.n_c_i
        self.shape_o = (self.n_fc_o,)
        self.output_neurons_multitude = self.shape_o[0]

        cnnprint.flatlayerprint(self.layer_index, self.shape_i)

        self.layer_feedforward = flattening.feedforward
        self.layer_backprop = flattening.backprop_transition_opt1
        self.layer_gradient = do_nothing

    def fullyconnected_store(self):
        self.g = actfunction.activation_function_set(self.tm_, self.hparams['activation_function'])
        self.n_fc_i = self.shape_i[0]
        self.shape_i = (self.n_fc_i,)
        self.n_fc_o = self.hparams['output_size']
        self.shape_o = (self.n_fc_o,)
        self.output_neurons_multitude = self.n_fc_o
        self.weights_shape = (self.n_fc_i, self.n_fc_o)
        self.biases_shape = (self.n_fc_o,)

        self.tot_lparams = self.n_fc_i*self.n_fc_o+self.n_fc_o

        cnnprint.fclayerprint((self.weights_shape, self.hparams['activation_function']), self.tot_lparams, self.layer_index, self.shape_i, self.shape_o)

        self.layer_feedforward = fullyconnected.feedforward_opt1
        self.layer_backprop = fullyconnected.backprop_opt1
        self.layer_gradient = fullyconnected.gradient_opt1

    def output_store(self):
        self.g = actfunction.activation_function_set(self.tm_, 'softmax')
        self.n_fc_i = self.shape_i[0]
        self.shape_i = (self.n_fc_i,)
        self.n_fc_o = self.classes_multitude
        self.shape_o = (self.n_fc_o,)
        self.output_neurons_multitude = self.n_fc_o
        self.weights_shape = (self.n_fc_i, self.n_fc_o)
        self.biases_shape = (self.n_fc_o,)

        self.tot_lparams = self.n_fc_i*self.n_fc_o+self.n_fc_o

        cnnprint.outlayerprint((self.weights_shape, 'softmax'), self.tot_lparams, self.layer_index, self.shape_i, self.shape_o)

        self.layer_feedforward = fullyconnected.feedforward_opt1
        self.layer_backprop = fullyconnected.backprop_output_layer
        self.layer_gradient = fullyconnected.gradient_opt1

    ## ! Storing: End
