import torch 
import torch.nn as nn
def loss_function(a, b):
    return torch.abs((a - b) * (a-b)).mean(dim = 0)
