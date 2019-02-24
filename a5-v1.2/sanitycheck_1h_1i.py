#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:42:27 2019

@author: simerjotkaur
"""
from highway import Highway
from cnn import CNN
import torch
embed_size = 3
batch_size = 1
sentence_length = 2
max_word = 5
e_char = 5
k_size = 2


""" Sanity check for highway.py
    (i) input/output shape check
    (ii) intermediate shape checks
    (iii) Verify output value
"""
print ("-"*80)
print("Running Sanity Check for Question 1h: Highway")
print ("-"*80)
model_highway = Highway(embed_size)

weight_size = [embed_size, embed_size]
assert(list(model_highway.projection.weight.size()) == weight_size), "Projection weight shape is incorrect: it should be:\n {} but is:\n{}".format(weight_size, list(model_highway.projection.weight.size()))
assert(list(model_highway.gate.weight.size()) == weight_size), "Gate weight shape is incorrect: it should be:\n {} but is:\n{}".format(weight_size, list(model_highway.gate.weight.size()))

input_highway = torch.ones(batch_size * sentence_length, 1, embed_size)
#print(input_highway)
output = model_highway(input_highway)
output_expected_size = [sentence_length * batch_size, 1, embed_size]
assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

print("Sanity Check Passed for Question 1h: Highway!")
print("-"*80)


""" Sanity check for cnn.py
    (i) input/output shape check
    (ii) intermediate shape checks
    (iii) Verify output value
"""
print ("-"*80)
print("Running Sanity Check for Question 1i: CNN")
print ("-"*80)
model_cnn = CNN(e_char, embed_size, max_word, k_size)

weight_size = [embed_size, e_char, k_size]
assert(list(model_cnn.convnet.weight.size()) == weight_size), "Conv1D weight shape is incorrect: it should be:\n {} but is:\n{}".format(weight_size, list(model_cnn.convnet.weight.size()))

input_cnn = torch.ones(batch_size * sentence_length, e_char, max_word)
#print(input_cnn)
output = model_cnn(input_cnn)
output_expected_size = [sentence_length * batch_size, 1, embed_size]
assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

print("Sanity Check Passed for Question 1i: CNN!")
print("-"*80)