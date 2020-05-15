import numpy as np

def compute_MSE(A, Y):
    return np.sum((A - Y) ** 2) / Y.shape[0]

def compute_Cost(A, Y):
    return 0.5 * np.sum((A - Y) ** 2)

def compute_R(y_train, y_err, y_res):
	return (1 - y_err / compute_MSE(np.mean(y_train), y_res))