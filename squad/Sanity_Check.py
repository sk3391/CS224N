#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:38:14 2019

@author: simerjotkaur
"""
import layers_qanet
import torch
embed_size = 500
batch_size = 64
sentence_length = 321

print ("-"*80)
print("Running Sanity Check")
print ("-"*80)
c_emb = torch.ones(64, 342, 500)         
q_emb = torch.ones(64, 32, 500)

enc = layers_qanet.EmbeddingEncoder(d_filters=128, drop_prob=0.1, n_conv=4, kernel_size=7, n_blocks=1, embed_size = 500)
cw_idxs = torch.ones(64,342)
qw_idxs = torch.ones(64,32)
c_mask = torch.zeros_like(cw_idxs)!= cw_idxs
q_mask = torch.zeros_like(qw_idxs)!= qw_idxs
print ("-"*80)
print("Starting Context Encoding")
print ("-"*80)
c_enc = enc(c_emb, c_mask)
print ("-"*80)
print("Starting Query Encoding")
print ("-"*80)
q_enc = enc(q_emb, q_mask)
print ("-"*80)
print("Starting CQ Attention")
print ("-"*80)
att = layers_qanet.CQAttention(hidden_size=128, drop_prob=0.1)
att_out = att(c_enc, q_enc, c_mask, q_mask)
print ("-"*80)
print("Starting Model Encoding")
print ("-"*80)
mod = layers_qanet.ModelEncoder(n_conv = 2, kernel_size = 5, d_filters = 4*128, drop_prob = 0.1, n_blocks = 7)
M0, M1, M2 = mod(att_out, c_mask)
print ("-"*80)
print("Starting QANet Output")
print ("-"*80)
out = layers_qanet.QANetOutput(hidden_size=8*128, drop_prob=0.1)
output = out(M0, M1, M2, c_mask)