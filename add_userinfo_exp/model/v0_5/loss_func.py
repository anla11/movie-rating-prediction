import torch 
def loss_fnc(a, b):
    return torch.abs((a - b) * (a-b)).mean(dim = 0)
