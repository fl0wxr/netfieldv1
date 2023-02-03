from math import floor, ceil
import sys


'''
Global variables:
    <len_substr_list>: list, contains in order a fixed length of each column of the displayed info table.
'''

def horizontal_bound_print():
    '''
    Horizontal bounding line build and display.
    '''

    substr_upper_bound_list = \
        [
            '-'*len_substr_list[_]
                for _ in range(len(len_substr_list))
        ]

    horizontal_bound = '+'+'+'.join(substr_upper_bound_list)+'+'

    print(horizontal_bound)

def layerheaders():
    '''
    Headers of info table build and display.

    Description:
        Builds and prints a string line containing the names of each column in the printed table.
    '''

    from numpy import cumsum

    global len_substr_list

    len_substr_list = [9,8,9,14,18,18]

    substr_lheader_list = ['Layer','Lpars','Filters','Shape','Input Shape','Output Shape']

    ## ! Printing section: Begin

    horizontal_bound_print()

    headerstring = combine(substr_lheader_list, len_substr_list, '|')
    print(headerstring)

    horizontal_bound_print()

    ## ! Printing section: End

## ! Layer diplay message build: Begin

'''
Layer information display message.

Description:
    Each of the following routines builds and prints a row, informing the user about the layers properties before the CNN executes.

Common Input: The common variables found on this section are the following.
    <Layer>: str, shows the number of layer.
    <Filters>: str, shows the number of filters in the case of the Convolutional Layer.
    <Size>: str, shows the size of scanned area per iteration, in the set of layers before the flattening occurs.
    <InputSize>: str, shows the size of the input.
    <OutputSize>: str, shows the size of the output.
'''

def inputlayerprint(T_shape):

    layerheaders()

    Layer = '0 in '
    Lpar = '-'
    Filters = '-'
    Size = '-'
    InputSize = '{} x {} x {}'.format(T_shape[0], T_shape[1], T_shape[2])
    OutputSize = '{} x {} x {}'.format(T_shape[0], T_shape[1], T_shape[2])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def convlayerprint(layer_hparams, tot_lparams, l, T_i_shape, T_o_shape):

    Layer = '{} conv'.format(l)
    Lpar = '{}'.format(tot_lparams)
    (n_c_o, f, p, s, g) = layer_hparams
    Filters = '{}'.format(n_c_o)
    Size = '{} x {}'.format(f, f, T_i_shape[2])
    InputSize = '{} x {} x {}'.format(T_i_shape[0], T_i_shape[1], T_i_shape[2])
    OutputSize = '{} x {} x {}'.format(T_o_shape[0], T_o_shape[1], T_o_shape[2])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def avgplayerprint(layer_hparams, l, T_i_shape, T_o_shape):

    Layer = '{} avgp'.format(l)
    Lpar = '-'
    (f, p, s) = layer_hparams
    Filters = '-'
    Size = '{} x {}'.format(f, f, T_i_shape[2])
    InputSize = '{} x {} x {}'.format(T_i_shape[0], T_i_shape[1], T_i_shape[2])
    OutputSize = '{} x {} x {}'.format(T_o_shape[0], T_o_shape[1], T_o_shape[2])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def maxplayerprint(layer_hparams, l, T_i_shape, T_o_shape):

    Layer = '{} maxp'.format(l)
    Lpar = '-'
    (f, p, s) = layer_hparams
    Filters = '-'
    Size = '{} x {}'.format(f, f, T_i_shape[2])
    InputSize = '{} x {} x {}'.format(T_i_shape[0], T_i_shape[1], T_i_shape[2])
    OutputSize = '{} x {} x {}'.format(T_o_shape[0], T_o_shape[1], T_o_shape[2])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def dropoutprint(l, input_shape):

    Layer = '{} drop'.format(l)
    Lpar = '-'
    Filters = '-'
    Size = '-'

    if len(input_shape) == 3:
        InputSize = '{} x {} x {}'.format(input_shape[0] ,input_shape[1], input_shape[2])
        OutputSize = '{} x {} x {}'.format(input_shape[0] ,input_shape[1], input_shape[2])
    else:
        InputSize = '{}'.format(input_shape[0])
        OutputSize = '{}'.format(input_shape[0])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def flatlayerprint(l, T_i_shape):

    Layer = '{} flat'.format(l)
    Lpar = '-'
    Filters = '-'
    Size = '-'
    InputSize = '{} x {} x {}'.format(T_i_shape[0] ,T_i_shape[1], T_i_shape[2])
    OutputSize = '{}'.format(T_i_shape[0]*T_i_shape[1]*T_i_shape[2])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def fclayerprint(layer_hparams, tot_lparams, l, T_i_shape, T_o_shape):

    Layer = '{} fcon'.format(l)
    Lpar = '{}'.format(tot_lparams)
    ending_line = horizontal_bound_print
    (weights_shape, g) = layer_hparams
    Filters = '-'
    Size = '{} x {}'.format(weights_shape[0], weights_shape[1])
    InputSize = '{}'.format(T_i_shape[0])
    OutputSize = '{}'.format(T_o_shape[0])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')

    print(layerstring)

def outlayerprint(layer_hparams, tot_lparams, l, T_i_shape, T_o_shape):

    Layer = '{} out'.format(l)
    Lpar = '{}'.format(tot_lparams)
    ending_line = horizontal_bound_print
    (weights_shape, g) = layer_hparams
    Filters = '-'
    Size = '{} x {}'.format(weights_shape[0], weights_shape[1])
    InputSize = '{}'.format(T_i_shape[0])
    OutputSize = '{}'.format(T_o_shape[0])

    substr_linfo_list = [Layer, Lpar, Filters, Size, InputSize, OutputSize]

    layerstring = combine(substr_linfo_list, len_substr_list, '|')
    print(layerstring)
    horizontal_bound_print()
    print()

## ! Layer display message build: End

def combine(substr_list, len_substr_list, separator):
    '''
    Combines string parts.

    Description:
        Centering string parts separated by a character with the given horizontal lengths.

    Inputs:
        <substr_list>: list, contains substrings refering to actual information (ex. kernel shape) about the layer corresponding to each of the columns
        <len_substr_list>: list, contains the fixed size of each column
        <separator>: str, a character that separates each of the the line substrings

    Returns <line_string>: str, the line string
    '''

    substr_list_fixed_size = \
        [
            substr_list[_].center(len_substr_list[_])
                for _ in range(len(len_substr_list))
        ]

    line_string = separator+separator.join(substr_list_fixed_size)+separator

    return line_string

def disp_result(label_names, T_o):
    '''
    Displays the classification result.

    Inputs:
        <label_names>: list, provides the names of each class in the same order as the final layer output vector.
        <T_o>: ndarray, the vector of the final layer with size n_L.
    '''

    y = list(zip(label_names, T_o))
    y.sort(key=lambda x: x[1])

    print('\nClassification result:')
    [print('{}: {:.1f}%'.format(y[_][0], y[_][1]*100)) for _ in range(len(y))]

## ! Hyperparameter and learnable parameter multitude print
def disp_handlparams(all_lparams, all_features):
    print('Learnable parameters #: %d - Features #: %d\n'%(all_lparams, all_features))

## ! Training/Epoch status: Begin

def seconds2min_sec(tot_time):
    '''
    Description:
        Takes seconds as an integer and returns a string that describes the interval in hour - minute - second format.
    '''

    if tot_time < 60:
        tot_time_h_m_s = '%ds'%(tot_time)
    elif 3600 > tot_time >= 60:
        (m, s) = divmod(tot_time, 60)
        tot_time_h_m_s = '%d:%02d'%(m, s)
    elif tot_time >= 3600:
        (m, s) = divmod(tot_time, 60)
        (h, m) = divmod(m, 60)
        tot_time_h_m_s = '%d:%02d:%02d'%(h, m, s)

    return tot_time_h_m_s

def disp_training_minibatch_progress(minibatch, minibatches, metrics):
    '''
    Description:
        Displays progress of training at each minibatch iteration.

    Inputs:
        <minibatch>: int.
        <minibatches>: int.
        <metrics>: tuple, holds some wanted metrics as a means to evaluate the performance of the current model.
            <C_tot_training>: float, the training sets cost functions output.
            <acc_training>: float, the training set accuracy.
    '''

    ## Progress bar
    def bar(minibatch, minibatches):

        space = 30

        empty_str = '\u2591'
        leader_str = '\u2592'
        filled_str = '\u2588'
        left_str = '|'
        right_str = '|'

        if minibatches == 1:
            return left_str+filled_str*space+right_str

        progress_floor = floor( (space/minibatches)*(minibatch+1) )
        progress_ceil = ceil( (space/minibatches)*(minibatch+1) )

        if minibatch == minibatches-1 or progress_floor == space-1:
            internal_bar = filled_str*(space-2)
        elif progress_ceil == 1:
            internal_bar = leader_str+empty_str*(space-(progress_ceil+2))
        else:
            internal_bar = filled_str*(progress_floor-1)+leader_str+empty_str*(space-(progress_floor+2))

        bar_ = left_str+internal_bar+right_str

        return bar_


    C_tot_training, acc_training = metrics

    progress_bar_frame = bar(minibatch, minibatches)
    message = '%d/%d %s - loss: %.4f - acc: %.4f'%(minibatch+1, minibatches, progress_bar_frame, C_tot_training, acc_training)
    print(message, end='\r')

def disp_epoch_progress(epoch, epochs, past_max_acc_val, error_metrics, tot_time):

    tot_time_proper = seconds2min_sec(tot_time)

    (training_loss_avg_on_specific_epoch, val_loss_avg_on_specific_epoch) = error_metrics['loss']

    (training_acc_avg_on_specific_epoch, val_acc_avg_on_specific_epoch) = error_metrics['acc']

    ## ! First message (mean metrics): Begin

    first_msg = 'Epoch %05d: tot_time: %s - loss: %.4f - acc: %.4f - val_loss: %f - val_acc: %.4f'%(epoch+1, tot_time_proper, training_loss_avg_on_specific_epoch, training_acc_avg_on_specific_epoch, val_loss_avg_on_specific_epoch, val_acc_avg_on_specific_epoch)

    ## ! First message (mean metrics): End

    ## ! Second line message (comparance): Begin

    if epoch == 0:
        improvement_status = ''
    elif past_max_acc_val >= val_acc_avg_on_specific_epoch:
        improvement_status = 'val_acc did not improve from %.4f'%(past_max_acc_val)
    elif past_max_acc_val < val_acc_avg_on_specific_epoch:
        improvement_status = 'val_acc improved from %.4f'%(past_max_acc_val)

    second_msg = ' '*13+improvement_status

    ## ! Second line message (comparance): End

    ## Display messages
    print('\n' + first_msg + '\n' + second_msg + '\n')

## ! Training/Epoch status: End
