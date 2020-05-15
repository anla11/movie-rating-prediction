import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def	init(input_shape, output_shape, learning_rate):
	model = LinearRegression(input_shape, output_shape)
    	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
	return model, optimizer

def predict(np_data):
    return model(Variable(torch.from_numpy(x_train))).data.numpy()


max_patience = 100    
