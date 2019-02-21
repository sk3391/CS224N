#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:42:27 2019

@author: simerjotkaur
"""
from highway import Highway
from cnn import CNN
import torch
embed_size = 300
batch_size = 5
sentence_length = 50
max_word = 21
e_char = 50
k_size = 5
model_higway = Highway(embed_size)
x = torch.randn(batch_size * sentence_length, embed_size)
y_pred_highway = model_higway(x)
print(y_pred_highway.size())
model_cnn = CNN(e_char, embed_size, k_size)
x = torch.randn(batch_size * sentence_length, e_char, max_word)
y_pred_cnn = model_cnn(x)
print(y_pred_cnn.size())