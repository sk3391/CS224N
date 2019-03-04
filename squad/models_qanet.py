"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers_qanet
import torch
import torch.nn as nn


class QANet(nn.Module):
    """QANet model for SQuAD.

    Based on the paper:
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, Quoc V. Le
    (https://arxiv.org/abs/1804.09541).

    Follows a high-level structure commonly found in SQuAD models:
        - Input Embedding layer: Embed word and char indices to get word and char vectors.
        - Embedding Encoder layer: Encode the embedded sequence.
        - Context-Query Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, device, word_vectors, char_vectors, char_embed_size, hidden_size = 128, drop_prob_word = 0.1, drop_prob_char = 0.05, drop_prob=0.1):
        super(QANet, self).__init__()
        
        self.emb = layers_qanet.Embedding(word_vectors = word_vectors, char_vectors = char_vectors, hidden_size = hidden_size,
                                    char_embed_size = char_embed_size, drop_prob_word = drop_prob_word, 
                                    drop_prob_char = drop_prob_char, drop_prob = drop_prob)

        self.enc = layers_qanet.EmbeddingEncoder(device = device, d_filters=hidden_size, drop_prob=drop_prob, n_conv=4, 
                                                 kernel_size=7, n_blocks=1, embed_size = word_vectors.size(1) + char_embed_size)

        self.att = layers_qanet.CQAttention(hidden_size=hidden_size, drop_prob=drop_prob)

        self.mod = layers_qanet.ModelEncoder(device = device, n_conv = 2, kernel_size = 5, d_filters = hidden_size, drop_prob = drop_prob, n_blocks = 7)

        self.out = layers_qanet.QANetOutput(hidden_size=2*hidden_size, drop_prob=drop_prob)

    #def forward(self, cw_idxs, qw_idxs):
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        #print("cw_idxs = " + str(cw_idxs.size()))
        #print("qw_idxs = " + str(qw_idxs.size()))
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        #c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        #c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        #q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        
        c_emb = self.emb(cc_idxs, cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qc_idxs, qw_idxs)         # (batch_size, q_len, hidden_size)
        #print("context embedding size = " + str(c_emb.size()))
        #print("query embedding size = " + str(c_emb.size()))
        #context embedding size = torch.Size([64, 321, 500])
        #query embedding size = torch.Size([64, 321, 500])
        
        c_enc = self.enc(c_emb, c_mask)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_mask)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        M0, M1, M2 = self.mod(att, c_mask)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(M0, M1, M2, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
