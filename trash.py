import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Conv2d(10, 10, 3)
        

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv(x1)
        return x2

def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

model = NeuralNetwork().to(device)
print(model)
print(gimme_params(model))

a = 2
if a == 3 or 1:
    print('T')
else:
    print('F')