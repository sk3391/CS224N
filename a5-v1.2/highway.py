#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, embed_size):
        """Initializing Highway Network
        @param embed_size (int): Embedding size (dimensionality)
        """
        super(Highway,self).__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        
    def forward(self, x_conv_out):
        x_proj = F.relu(self.projection(x_conv_out))
        #print("x_proj size = " + str(x_proj.size()))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        #print("x_gate size = " + str(x_gate.size()))
        x_highway = torch.add(torch.mul(x_gate, x_proj), torch.mul((1.0 - x_gate), x_conv_out))
        #print("x_highway size = " + str(x_highway.size()))
        return x_highway
### END YOUR CODE 

