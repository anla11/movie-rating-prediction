import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import sys, getopt
import importlib

model_config, loss_lib = None, None

def train(x_data, y_data, learning_rate, num_epochs, check_point, model_save):
	n = int(0.8 * len(x_data))
	x_train, y_train = x_data[:n,:], y_data[:n, :]
	x_val, y_val = x_data[n:, :], y_data[n:, :]
	
	model, optimizer = model_config.init(x_train.shape[1], y_train.shape[1],learning_rate)
	train_loss_list, val_loss_list = [], []

	#early stopping
	patience = model_config.max_patience
	best_val = None
	
	# Train the Model 
	for epoch in range(num_epochs):
		# Convert numpy array to torch Variable
		inputs = Variable(torch.from_numpy(x_train))
		targets = Variable(torch.from_numpy(y_train), requires_grad = False)

		# Forward + Backward + Optimize
		optimizer.zero_grad()  
		outputs = model(inputs)
		
		train_loss = loss_lib.loss_fnc(outputs, targets).sum()
		train_loss_list.append(train_loss.data[0])
		
		#validate
		inputs = Variable(torch.from_numpy(x_val))
		targets = Variable(torch.from_numpy(y_val), requires_grad = False)
		outputs = model(inputs)   
		val_loss = loss_lib.loss_fnc(outputs, targets).sum().data[0]
		val_loss_list.append(val_loss)
		
		#optimize
		train_loss.backward()
		optimizer.step()
		
		if (epoch == 0) or ((epoch+1) % check_point == 0) or (epoch == num_epochs-1):
			print ('Epoch [%d/%d], Training Loss: %.4f, Validating Loss: %.4f' 
				   %(epoch+1, num_epochs, train_loss.data[0], val_loss))
			torch.save(model, '%s/%d.pth' % (model_save, epoch+1))

		if (best_val is None) or ((best_val is not None) and (val_loss < best_val)) :
			best_val = val_loss
			patience = model_config.max_patience
		else:
			patience -= 1
		if patience == 0:
			print 'Early stopping at %d' % epoch
			break
		

	# Plot the graph
	print 'Plot graph from epoch 10th'
	plt.plot(range(len(train_loss_list))[10:], train_loss_list[10:], label='train')
	plt.plot(range(len(train_loss_list))[10:], val_loss_list[10:], label = 'validate')
	plt.legend()
	plt.show()
	return model

def     main(argv):
	config_path, x_train, y_train, learning_rate, num_epochs, check_point, model_save = None, None, None, None, None, None, None
	try:
		opts, args = getopt.getopt(argv, "c:x:y:lr:ne:cp:ms", ["config=","x_train=", "y_train=", "learning_rate=", "num_epochs=", "check_point=", "model_save="])
	except getopt.GetoptError:
		sys.exit(2)
	for opt, arg in opts:
		if opt == "--config":
			config_path = arg
		if opt == "--x_train":
			x_train = arg
		elif opt == "--y_train":
			y_train = arg
		elif opt == "--learning_rate":
			learning_rate = float(arg)
		elif opt == "--num_epochs":
			num_epochs = int(arg)
		elif opt == "--check_point":
			check_point = int(arg)
		elif opt == "--model_save":
			model_save = arg

	global loss_lib
	loss_lib = importlib.import_module('%s.loss_func' %config_path)
	global model_config
	model_config = importlib.import_module('%s.model_config'% config_path)

	x_train = pd.read_csv(x_train).iloc[:, 1:]
	y_train = pd.read_csv(y_train).iloc[:, 1:]

	x_train = np.array(x_train, dtype = np.float32)
	y_train = np.array(y_train, dtype = np.float32)			
	model = train(x_train, y_train, learning_rate, num_epochs, check_point, model_save)

	
if __name__ == "__main__":
	main(sys.argv[1:])

'''
python train.py --config=model.v0_5 --x_train=../data/input_formated/v0_5/x_train.csv --y_train=../data/input_formated/v0_5/y_train.csv --learning_rate=0.3 --num_epochs=1000 --check_point=100 --model_save=model/v0_5/checkpoint
'''