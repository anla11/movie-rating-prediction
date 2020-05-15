import matplotlib.pyplot as plt
import numpy as np

def 	plotdata(x_range, lines, nfi = 0, show = True, label = ['x' ,'y'], legend = None, early_stop = -1, colors = ['cyan', 'orange', 'green', 'y', 'brown', 'gray']):
    plt.figure(nfi)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    for i in range(len(lines)):
        plt.plot(x_range, lines[i], colors[i])
    if early_stop != -1:
        min_y, max_y = plt.ylim()
        plt.plot([early_stop, early_stop], [min_y, max_y], 'red')
        plt.text(early_stop, (min_y + max_y) / 2, 'STOP')
    if legend != None:
        plt.legend(legend)       
    if show:
        plt.show()

global const_n_epoch, const_layers, const_mnb_size, const_max_patience, const_learning_rate, const_momentum, const_act_func, const_SIGMOID, const_TANH, const_RELU, const_alpha
