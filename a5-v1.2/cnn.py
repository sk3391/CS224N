#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char, embed_size, max_word_length, kernel_size=5):
        """Initializing Highway Network
        @param embed_size (int): Embedding size (dimensionality)
        """
        super(CNN,self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length
        self.convnet = nn.Conv1d(e_char, embed_size, kernel_size)
        #print(self.convnet.weight)
        #print(self.convnet.bias)
        
    def forward(self, x_reshaped):
        #print("x_reshaped = " + str(x_reshaped.size()))
        x_conv = self.convnet(x_reshaped)
        #print("x_conv = " + str(x_conv.size()))
        #print("x_conv = ")
        #print(x_conv)
        x_conv_relu = F.relu(x_conv)
        #print("x_conv_relu = " + str(x_conv_relu.size()))
        #print("x_conv_relu = ")
        #print(x_conv_relu)
        x_conv_out = F.max_pool1d(x_conv_relu, self.max_word_length - self.kernel_size + 1).permute(0,2,1)
        #print("x_conv_out = " + str(x_conv_out.size()))
        #print("x_conv_out = ")
        #print(x_conv_out)
        return x_conv_out
### END YOUR CODE

