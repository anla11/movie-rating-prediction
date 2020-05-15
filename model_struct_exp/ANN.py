import parm
import numpy as np
from active_function import *
from error_function import *
from batch_norm import batchnorm_forward, batchnorm_backward
from dropout import dropout
import copy
import math

def activation_function(Z):
    A = Z
    if parm.act_func == parm.SIGMOID:
        A = sigmoid(Z)
    else:
        if parm.act_func == parm.TANH:
            A = tanh(Z)
        else:
            if parm.act_func == parm.RELU:
                A = relu(Z, parm.alpha)   
    return A

def add_bias_unit(A):
    if (parm.act_func == parm.SIGMOID) or (parm.act_func == parm.TANH):
        A = np.hstack((np.ones((A.shape[0], 1)), A))   
    else:
        A = np.hstack((np.ones((A.shape[0], 1)) * 0.1, A))   
    return A

def compute_grad_actfunc(Z):
    if (parm.act_func == parm.SIGMOID) or (parm.act_func == parm.TANH):
        if parm.act_func == parm.SIGMOID:
            return grad_sigmoid(Z)
        else: #TANH
            return grad_tanh(Z)
    else:
        if parm.act_func == parm.RELU:
            return grad_relu(Z, parm.alpha)
        return np.ones(Z.shape)

def compute_feedforward(Ws, BNva, X, cache, train_mode):
    dropout_cache, batchnorm_cache = cache
    N = X.shape[0]
    As = []
    Hs = []
    Zs = []

    A = dropout(X, dropout_cache[0], train_mode = train_mode)
    A = add_bias_unit(A)
    if train_mode:
        As.append(A)  

    for layer in range(len(Ws)):     
        H = A.dot(Ws[layer])
        Z, batchnorm_cache[layer] = batchnorm_forward(H, BNva[layer], batchnorm_cache[layer])

        if layer == len(Ws)-1: #last layer
            A = Z
        else:
            A = activation_function(Z)
            A = dropout(A, dropout_cache[layer+1], train_mode = train_mode)
            A = add_bias_unit(A)           

        if (train_mode == True) or (layer == len(Ws) -1):
            Hs.append(copy.deepcopy(H))
            Zs.append(copy.deepcopy(Z))
            As.append(copy.deepcopy(A)) 
        # if DEBUG:
            # print layer, ': ', A

    cache = dropout_cache, batchnorm_cache
    if train_mode:
        return As, Hs, Zs, cache
    else:
        return As[0]

def compute_feedbackward(W, BNva, As, Hs, Zs, Y, caches):
    dropout_cache, batchnorm_cache = caches
 
    sz = Y.shape[0]
    
    delta = As[-1] - Y
    dz = delta #no activation function at last layer
    dh, bn_grad = batchnorm_backward(Hs[-1], BNva[-1], dz, batchnorm_cache[-1])
    grad = As[-2].T.dot(dh) / sz
    w_grads = [grad]
    bn_grads = [bn_grad]

    for l in range(1, len(W)):
        delta = dh.dot(W[-l][1:].T)
        delta = dropout(delta, dropout_cache[-l], train_mode = True)
        dz = np.multiply(delta, compute_grad_actfunc(Zs[-l]))
        dh, bn_grad = batchnorm_backward(Hs[-l-1], BNva[-l-1], dz, batchnorm_cache[-l-1])
        bn_grads.append(bn_grad)
        w_grad = As[-l-2].T.dot(dh) / sz
        w_grads.append(w_grad)

    return w_grads, bn_grads

def train(X, Y, X_val, Y_val):
    N = X.shape[0]
    layer_sizes = [X.shape[1]] + parm.layers + [1]        

    # Init Weight
    np.random.seed(0) # This will fix the randomization; so, you and me will have the same results    
    W = [np.random.rand(layer_sizes[l]+1, layer_sizes[l+1]) / np.sqrt(layer_sizes[l]+1) for l in range(len(layer_sizes)-1)]    
    if parm.act_func == parm.RELU:
        W = [np.random.rand(layer_sizes[l]+1, layer_sizes[l+1]) * np.sqrt(2.0 / N) for l in range(len(layer_sizes)-1)]    
    mW = [np.zeros((layer_sizes[l]+1, layer_sizes[l+1])) for l in range(len(layer_sizes)-1)]

    # Init Batch Normalize
    #not normalize input layer
    gammas = list(np.ones((1, layer_sizes[l])) for l in range(1, len(layer_sizes)))
    betas = list(np.zeros((1, layer_sizes[l])) for l in range(1, len(layer_sizes)))    
    BNva = [(gammas[i], betas[i]) for i in range(len(layer_sizes)-1)]

    batchnorm_cache = []
    for i in range(len(layer_sizes)-1): 
        if (parm.run_batchnorm == False) or (i == len(layer_sizes) - 2):
            batchnorm_cache.append(None)
        else:
            mu, var = 0.0, 0.0
            cache = (mu, var)
            batchnorm_cache.append(cache)

    dropout_cache = []
    for i in range(len(layer_sizes)-1):
        dropout_cache.append(None)

    # Weight list, Errors list and caches list
    Ws = []
    BNs =[]
    Caches = []    
    errs_val = []
    errs_train = []
   
    #create random input
    rand_idx = np.arange(N)  

    # early stopping
    best_epoch = parm.n_epoch - 1
    best_err = None
    patience = parm.max_patience

    for epoch in range(parm.n_epoch):
        #random position
        np.random.shuffle(rand_idx)

        avg_mnb_cache = batchnorm_cache
        cnt = (N - 1) / parm.mnb_size + 1

        cache = (None, None)

        for idx in range(0, N, parm.mnb_size):
            dropout_cache_idx = dropout_cache
            if parm.run_dropout:
                for i in range(len(layer_sizes)-1): # don't drop output layer
                    if i == 0: #input layer
                        prb = 1
                    else:
                        prb = 0.5
                    dropout_cache_idx[i] = (prb, np.random.binomial(1, prb, size = (1, layer_sizes[i])))

            cache = dropout_cache_idx, batchnorm_cache
            rand_X = X[rand_idx[idx : idx + parm.mnb_size]]
            rand_Y = Y[rand_idx[idx : idx + parm.mnb_size]]
            sz = len(rand_Y)
            
            #feed forward
            As, Hs, Zs, cache = compute_feedforward(W, BNva, rand_X, cache, train_mode = True)
            grads, dparm_bn = compute_feedbackward(W, BNva, As, Hs, Zs, rand_Y, cache)
            #feed backward to update W 
 
            for l in range(1, len(W)+1):            
                mW[-l] = parm.momentum * mW[-l] - parm.learning_rate * (grads[l-1] + 2 * parm.lamda * W[-l])
                W[-l] += mW[-l]
                if (parm.run_batchnorm == True) and (l < len(W)):
                    BNva[-l] = (BNva[-l][0] - dparm_bn[l-1][0], BNva[-l][1] - dparm_bn[l-1][1])
            
            #update avg_batchnorm_cache
            for i in range(len(layer_sizes)-1):
                if cache[1][i] != None:
                    avg_mnb_cache[i] = (avg_mnb_cache[i][0] + cache[1][i][0] / cnt, avg_mnb_cache[i][1] + cache[1][i][1] * cnt / (cnt - 1))
                else:
                    avg_mnb_cache[i] = None

        BNs.append(copy.deepcopy(BNva))
        cache = (cache[0], avg_mnb_cache)
        Caches.append(copy.deepcopy(cache))

        err_train = compute_MSE(compute_feedforward(W, BNva, X, Caches[-1], train_mode = False), Y)
        err_val = compute_MSE(compute_feedforward(W, BNva, X_val, Caches[-1], train_mode = False), Y_val)

        errs_val.append(err_val)
        errs_train.append(err_train)        

        Ws.append(copy.deepcopy(W))
        if (parm.max_patience != None) and (patience != -1):
            if (best_err == None) or (err_val < best_err):
                best_err = err_val
                best_epoch = epoch
                patience = parm.max_patience
            else:
                patience -= 1

    Model = best_epoch, Ws, BNs, Caches
    return errs_train, errs_val, Model

def test(X, Y, Model):
    stop, Ws, BNs, Caches = Model
    errs = []
    for epoch in range(stop):
        err = compute_MSE(compute_feedforward(Ws[epoch], BNs[epoch], X, Caches[epoch], train_mode = False), Y)
        errs.append(err)
    return errs
