import torch
import torch.nn as nn

criterion = nn.MSELoss()
def loss_fnc(output, target):
	return criterion(output, target)


