import parm
import numpy as np
from get_data import *
from error_function import *
from ANN import *
from plot import *

""" get data """
X, Y = get_data("Data/movies.csv")
print 'Number of input feature: ', X.shape[1]

""" create training set, valdiation set and testing set with ratio: 0.6:0.2:0.2"""
X_train = X[0: int(0.6 * len(X))]
X_val   = X[int(0.6 * len(X)) : int(0.8 * len(X))]
X_test  = X[int(0.8 * len(X)): len(X)]
Y_train = Y[0: int(0.6 * len(Y))]
Y_val   = Y[int(0.6 * len(Y)) : int(0.8 * len(Y))]
Y_test  = Y[int(0.8 * len(Y)) : len(Y)]
print 'Size of train data: %d, Size of validation data: %d, Size of test data: %d' % (len(X_train), len(X_val), len(X_test))


""" normalize data - column 0 1 - DAY-MONTH """
for i in range(2): 
    X_train[:, i], X_val[:, i], X_test[:, i] = scale(X_train[:, i], X_val[:, i], X_test[:, i])

def run_training(plotshow = False):
    print "Training with learning rate = %.4f, momentum = %.4f" % (parm.learning_rate, parm.momentum)
    Train_Errs, Val_Errs, Model = train(X_train, Y_train, X_val, Y_val)
    stop = Model[0]

    R_train = compute_R(Y_train, Train_Errs[stop], Y_train)

    if stop < parm.n_epoch - 1:
        print "Early stopping at epoch %d: " % (stop)

    print "	Adjusted R-squared in training set   = %.4f" % (R_train)
    print "	Error in training set  : MSE = %.4f, RMSE = %.4f" %(Train_Errs[stop], np.sqrt(Train_Errs[stop]))
    print "	Error in validation set: MSE = %.4f, RMSE = %.4f" %(Val_Errs[stop], np.sqrt(Val_Errs[stop]))

    if plotshow == True:
        plotdata(x_range = range(parm.start, parm.n_epoch), lines = [Train_Errs[parm.start:], Val_Errs[parm.start:]], nfi = 0, show = False, label = ['Epoch', 'MSE'], legend = ['Training Error', 'Validation Error'], early_stop = stop)
        plt.show()

    return Train_Errs, Val_Errs, Model

def run_testing(Model):
    stop, Ws, BNs, Caches = Model
    Test_Err = compute_MSE(compute_feedforward(Ws[stop-1], BNs[stop-1], X_test, Caches[stop-1], train_mode = False), Y_test)
    R_test = compute_R(Y_train, Test_Err, Y_test) 
    print "Test:"
    print " Adjusted R-squared: %.4f" %(R_test)
    print " Error in testing set: MSE = %.4f" %(Test_Err) 
    return Test_Err

def write_result(name, errors):
    np.savetxt(name, errors, delimiter = ",",  fmt="%f")

parm.init() 