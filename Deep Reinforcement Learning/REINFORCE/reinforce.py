# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.inp_layer = nn.Linear(4, 128)
        self.out_layer = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.inp_layer(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.out_layer(x)
        return F.softmax(x,dim=1)