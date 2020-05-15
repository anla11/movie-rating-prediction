import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

def rmse(y, y_hat):
	"""Compute root mean squared error"""
	return torch.sqrt(torch.mean((y - y_hat).pow(2)))

def     main(argv):
	x_test, y_test, model_load = None, None, None
	try:
		opts, args = getopt.getopt(argv, "x:y:m", ["x_test=", "y_test=", "model_load="])
	except getopt.GetoptError:
		sys.exit(2)
	for opt, arg in opts:
		if opt == "--x_test":
			x_test = arg
		if opt == "--y_test":
			y_test = arg
		elif opt == "--model_load":
			model_load = arg

	x_test = pd.read_csv(x_test).iloc[:, 1:]
	y_test = pd.read_csv(y_test).iloc[:, 1:]

	x_test = np.array(x_test, dtype = np.float32)
	y_test = np.array(y_test, dtype = np.float32)			
	model = torch.load(model_load)
	inputs = Variable(torch.from_numpy(x_test))
	targets = Variable(torch.from_numpy(y_test[:,1] * 5), requires_grad = False)
	outputs = Variable(model(inputs).data[:,1] * 5)	
	print rmse(outputs, targets).sum().data[0]

if __name__ == "__main__":
	main(sys.argv[1:])

'''
python test.py --x_test=../data/input_formated/v0_5/x_train.csv --y_test=../data/input_formated/v0_5/y_train.csv --model_load=model/v0_5/checkpoint/1000.pth

'''
