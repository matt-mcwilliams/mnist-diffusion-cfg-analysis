import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self, num_hidden=16):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1,4,3),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(4,4,3),
      nn.ReLU(),
      nn.Flatten(-3,-1),
      nn.Linear(484,num_hidden),
      nn.ReLU(),
      nn.Linear(num_hidden,10),
      nn.Softmax()
    )
  
  def forward(self, x):
    return self.layers(x)