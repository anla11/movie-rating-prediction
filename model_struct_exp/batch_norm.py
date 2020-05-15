import numpy as np

DEBUG = True
eps = 0.000001

def batchnorm_forward(h, w, cache, train_mode = True):
	if cache == None:
		return h, cache
	gamma, beta = w
	n = h.shape[0]
	mu = 1/float(n) * np.sum(h, axis = 0)
	var = 1/float(n) * np.sum((h-mu)**2, axis = 0)
	if train_mode == False:
		mu, var = cache
	h_hat = (h - mu) * ((var + eps) ** (-0.5))
	z = gamma * h_hat + beta
	cache = (mu, var)
	return z, cache

def batchnorm_backward(h, w, dy, cache):
	if cache == None:
		return dy, (0, 0)
	gamma, beta = w
	mu, var = cache 

	h_hat = (h - mu) * ((var + eps) ** (-0.5))
	N = h.shape[0]
	dbeta = np.sum(dy, axis = 0)
	dgamma = np.sum(h_hat * dy, axis = 0) 

	dvar = np.sum(dy * gamma * (h - mu) * (-0.5) * ((var + eps) ** (-1.5)), axis = 0) 
	dmu = np.sum(dy * gamma * (-1 * (var + eps) ** (-0.5)), axis = 0) + dvar * np.sum(-2 * (h - mu), axis = 0) 
	dh = dy * gamma * ((var + eps) ** (-0.5))  + dvar * 2.0 * (h - mu) / N + dmu / N
	grad_bn = dgamma, dbeta
	# print 'check'
	# print np.sum(dy == dh)
	# print 'dgamma, dbeta'
	# print dgamma
	# print dbeta
	# print 'end'
	return dh, grad_bn


'''
def 	check_backward():
	N, D = 4, 5
	x = 5 * np.random.randn(N, D) + 12
	gamma = np.random.randn(D)
	beta = np.random.randn(D)
	dout = np.random.randn(N, D)

	bn_param = {'mode': 'train'}
	fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
	fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
	fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]

	dx_num = eval_numerical_gradient_array(fx, x, dout)
	da_num = eval_numerical_gradient_array(fg, gamma, dout)
	db_num = eval_numerical_gradient_array(fb, beta, dout)

	_, cache = batchnorm_forward(x, gamma, beta, bn_param)
	dx, dgamma, dbeta = batchnorm_backward(dout, cache)
	print 'dx error: ', rel_error(dx_num, dx)
	print 'dgamma error: ', rel_error(da_num, dgamma)
	print 'dbeta error: ', rel_error(db_num, dbeta)

if DEBUG:
	check_backward()
'''