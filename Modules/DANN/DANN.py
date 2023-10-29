import torch
import torch.nn as nn
from Modules.GradientReversal import gradient_reversal
import numpy as np

class DANN(nn.Module):
  def __init__(self):
    super().__init__() # initialize all nn.Module base methods

    self.c1 = nn.Conv2d(3, 32, (5,5)) # feature extraction CNN layers
    self.c2 = nn.Conv2d(32, 48, (5,5))
    self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.lb1 = nn.Linear(768, 100) # linear regression in feature space
    self.lb2 = nn.Linear(100, 100) # for label classification
    self.lb3 = nn.Linear(100, 10)

    self.dl1 = nn.Linear(768, 100) # linear regression in feature space
    self.dl2 = nn.Linear(100, 1) # for domain classification

    self.fl = nn.Flatten()
    self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()

  def feature_extraction(self, x):
    x = self.relu(self.c1(x)); x = self.mp(x) # [2x] CONV --> ReLU --> MAX POOL 
    x = self.relu(self.c2(x)); x = self.mp(x)
    return self.fl(x) # N x 768 features

  def label_reg(self, x):
    x = self.relu(self.lb1(x))
    x = self.relu(self.lb2(x))
    return self.lb3(x) # N x 10 logits
  
  def domain_reg(self, x):
    x = self.relu(self.dl1(x))
    return self.sig(self.dl2(x)) # N x 1 domain hypothesis

  def forward(self, x, lambd = torch.tensor([1.])):
    x = x.expand(-1, 3, -1, -1) # populate rgb channels, if not populated
    x = self.feature_extraction(x)
    
    clabel = self.label_reg(x)
    dlabel = self.domain_reg(gradient_reversal.revgrad(x, lambd)) # GRL
    return clabel, dlabel

# Domain adpation paramter, lambda, as defined in https://github.com/fungtion/DANN/blob/master/train/main.py
def DA_parameter(epoch, batch_iter, epoch_num, batch_num):
  p = float(batch_iter + epoch*batch_num) / epoch_num / batch_num
  g = 10

  return torch.tensor([2. /(1+np.exp(-g*p)) - 1]) 