import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from functools import partial
import pandas as pd


def cnnsubplots():

    fig = plt.figure()

    ax = [None for i in range(2)]

    ax[0] = fig.add_subplot(211)
    ax[1] = fig.add_subplot(212)

    ## Creates a distance between the subplots to avoid overlapping
    # fig.tight_layout()

    return fig, ax

def error_metrics_subplots(error_metrics_history, frame):

    error_metrics_history = pd.read_csv(error_metrics_history_lc)

    training_acc_history = list(error_metrics_history['training_acc'])
    val_acc_history = list(error_metrics_history['val_acc'])
    training_loss_history = list(error_metrics_history['training_loss'])
    val_loss_history = list(error_metrics_history['val_loss'])

    x_axis_pair = list(range(len(error_metrics_history)))

    ## Clearing previous frame
    ax[0].clear()
    ax[1].clear()

    ## To show only integer numbers on the x-axis
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax[0].set_title('accuracy')
    # ax[1].set_title('loss function output')

    # ax[0].set_xlabel('epoch')
    ax[1].set_xlabel('epoch')

    ax[0].set_ylabel('acc')
    ax[1].set_ylabel('loss')

    ## Add a set of points in the subplot in each iteration
    ax[0].plot(x_axis_pair, training_acc_history, color='red', label='training acc')
    ax[0].plot(x_axis_pair, val_acc_history, color='blue', label='val acc')

    ax[1].plot(x_axis_pair, training_loss_history, color='red', label='training loss')
    ax[1].plot(x_axis_pair, val_loss_history, color='blue', label='val loss')

    ax[0].legend()
    ax[1].legend()

    plt.draw()
    plt.pause(0.0001)


if __name__ == '__main__':

    error_metrics_history_file_format = 'csv'

    error_metrics_history_lc = sys.argv[1]

    assert error_metrics_history_lc.split('.')[-1] == error_metrics_history_file_format, 'Exception: Wrong file path or format.'

    fig, ax = cnnsubplots()

    error_metrics_subplots_ = partial(error_metrics_subplots, error_metrics_history_lc)

    ani = animation.FuncAnimation(fig, error_metrics_subplots_, interval=1000)
    plt.show()
