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
        #print(self.projection.weight)
        #print(self.projection.bias)
        #print(self.gate.weight)
        #print(self.gate.bias)
        
    def forward(self, x_conv_out):
        """
        Obtain xhighway by combining the projection with the skip-connection using gate
        @param x_conv_out: Output Tensor of Conv1D of integers of shape (sentence_length * batch_size, 1, embed_size)

        @param x_highway: Tensor of shape (sentence_length * batch_size, 1, embed_size), containing the 
            combination of skip-connection with the projection
        """
        #print("x_conv_out size = " + str(x_conv_out.size()))
        x_proj = F.relu(self.projection(x_conv_out))
        #print("x_proj size = " + str(x_proj.size()))
        #print("x_proj = ")
        #print(x_proj)
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        #print("x_gate size = " + str(x_gate.size()))
        #print("x_gate = ")
        #print(x_gate)
        x_highway = torch.add(torch.mul(x_gate, x_proj), torch.mul((1.0 - x_gate), x_conv_out))
        #print("x_highway size = " + str(x_highway.size()))
        #print("x_highway = ")
        #print(x_highway)
        return x_highway
### END YOUR CODE 

