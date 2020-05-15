import numpy as np

def sigmoid(S):
    return 1 / (1 + np.exp(-S))

def grad_sigmoid(S):
    A = sigmoid(S)
    return np.multiply(A, 1 - A)

def tanh(S):
    return (np.exp(S) - np.exp(-S))/(np.exp(S) + np.exp(-S))

def grad_tanh(S):
    return 1 - tanh(S) ** 2

def relu(S, alpha):
    zeros = np.zeros(S.shape)
    return np.maximum(zeros, S) + alpha * np.minimum(zeros, S)

def grad_relu(S, alpha):
    return (S > 0) + (S < 0) * alpha    

def softmax(S):
    A = np.exp(S)
    A /= A.sum(axis=1, keepdims=True)
    return A

def nofunc(S):
    return S

def gradNofunc(S):
    return 1
