"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from torch.autograd import Variable


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, char_embed_size, drop_prob_word, drop_prob_char, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.drop_prob_word = drop_prob_word
        self.drop_prob_char = drop_prob_char
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = CharEmbeddings(char_embed_size, char_vectors, self.drop_prob_char)
        self.proj = nn.Linear(word_vectors.size(1)+char_embed_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size, self.drop_prob)

    #def forward(self, x_w):
    def forward(self, x_c, x_w):
        emb = F.dropout(self.embed(x_w), self.drop_prob_word, self.training)   # (batch_size, seq_len, embed_size)
        #### Added for char embedding
        char_emb = self.char_embed(x_c) # (batch_size, seq_len, char_embed_size)
        emb = torch.cat((char_emb, emb), 2)
        ##########
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


################################Character Embeddings################################################
class CharEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, char_vectors, drop_prob_char, kernel_size = 7):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CharEmbeddings, self).__init__()
        self.e_char = char_vectors.size(1)
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob_char
        self.embeddings = nn.Embedding.from_pretrained(char_vectors)
        self.model_cnn = CNN(self.e_char, self.embed_size, self.kernel_size)
        self.model_highway = Highway(self.embed_size)

    def forward(self, x):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (batch_size, sentence_length, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (batch_size, sentence_length, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        x_emb = self.embeddings(x)
        (batch_size, sentence_length, max_word_length, e_char) = x_emb.size()
        x_reshaped = (x_emb.view(batch_size*sentence_length, max_word_length, e_char)).permute(0,2,1)
        x_conv_out = self.model_cnn(x_reshaped)
        x_highway = self.model_highway(x_conv_out)
        x_word_emb = F.dropout(x_highway, self.drop_prob, self.training).view(batch_size, sentence_length, self.embed_size)
        return x_word_emb
    
class CNN(nn.Module):
    def __init__(self, e_char, embed_size, kernel_size=7):
        """Initializing CNN Network
        @param embed_size (int): Embedding size (dimensionality)
        """
        super(CNN,self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.convnet = nn.Conv1d(e_char, embed_size, kernel_size)
        
    def forward(self, x):
        x_conv = self.convnet(x)
        x_conv_relu = F.relu(x_conv)
        x_conv_out = F.max_pool1d(x_conv_relu, x.size(2) - self.kernel_size + 1).permute(0,2,1)
        return x_conv_out

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
        """
        Obtain xhighway by combining the projection with the skip-connection using gate
        @param x_conv_out: Output Tensor of Conv1D of integers of shape (sentence_length * batch_size, 1, embed_size)

        @param x_highway: Tensor of shape (sentence_length * batch_size, 1, embed_size), containing the 
            combination of skip-connection with the projection
        """
        x_proj = F.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = torch.add(torch.mul(x_gate, x_proj), torch.mul((1.0 - x_gate), x_conv_out))
        return x_highway
        

##################################End of Character Embeddings##############################################
   
class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size, drop_prob):
        super(HighwayEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
            x = F.dropout(x, self.drop_prob, self.training)

        return x


################################QANet Layers################################################
        
class PositionalEncoding(nn.Module):
    """Initializing Positional Encoding Layer"""
    def __init__(self, d_filters, drop_prob, min_timescale=1.0, max_timescale=1.0e4):
        super(PositionalEncoding, self).__init__()
        self.d_filters = d_filters
        self.drop_prob = drop_prob
        self.frequencies = torch.exp(torch.arange(0, self.d_filters, 2).type(torch.float32) * -(math.log(10000.0) / self.d_filters)).unsqueeze(0)
    def forward(self, x, min_timescale=1.0, max_timescale=1.0e4):
        #print("positional encoding input = " + str(x.size()))
        x = x#.transpose(1, 2)
        length = x.size()[1]
        channels = x.size()[2]
        signal = self.get_timing_signal(length, channels, min_timescale, max_timescale)
        x = (x + signal.to(x))#.transpose(1, 2)
        #print("Final Embedding = " + str(x.size()))
#        print("positional encoding input = " + str(x.size()))
#        batch_size = x.size(0)
#        length = x.size(1)
#        position = torch.arange(0, length).type(torch.float32).unsqueeze(1)
#        print("position = " + str(position.size()))
#        print("frequencies = " + str(self.frequencies.size()))
#        pos_enc = torch.zeros(batch_size, self.d_filters, length)
#        pos_enc[:, :, 0::2] = torch.sin(position * self.frequencies)
#        pos_enc[:, :, 1::2] = torch.cos(position * self.frequencies)
#        ###print("positional encoding = " + str(pos_enc.size()))
#        pos_enc = Variable(pos_enc, requires_grad=False).permute(0,2,1)
#        x = F.dropout(x + pos_enc, self.drop_prob, self.training)
        #print(x[0,0,:])
        ###print("Final Embedding = " + str(x.size()))
        return x
    
    def get_timing_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = torch.arange(length).type(torch.float32)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
        inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
        m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
        signal = m(signal)
        signal = signal.view(1, length, channels)
        return signal

class DepthSepCNN(nn.Module):
    def __init__(self, drop_prob, in_channels, d_filters = 128, kernel_size=7):
        """Initializing Depthwise Separable CNN Network
        Args:
            in_channel (int): Input Channel Size
            d_filters (int): number of filters
            kernel_size (int): Kernel Size
        """
        super(DepthSepCNN, self).__init__()
        self.drop_prob = drop_prob
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.d_filters = d_filters
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size = kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.pointwise = nn.Conv1d(in_channels, d_filters, kernel_size = 1)
        
    def forward(self, x):
        ##print("x in CNN = " + str(x.size()))
        x_conv = self.depthwise(x)
        ##print("x after depthwise CNN = " + str(x_conv.size()))
        x_conv_out = F.dropout(self.pointwise(x_conv), self.drop_prob, self.training)
        ##print("x after pointwise CNN = " + str(x_conv_out.size()))
        return F.relu(x_conv_out)

class SelfAttention(nn.Module):
    """Implementing Multi-head Attention.
    Based on the paper:
    "Attention is all you need."
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/pdf/1706.03762.pdf)"""
    def __init__(self, drop_prob, d_filters = 128, n_heads = 8):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_filters // n_heads
        self.d_k = d_filters // n_heads
        self.drop_prob = drop_prob
        self.W_q = nn.ParameterList()
        self.W_k = nn.ParameterList()
        self.W_v = nn.ParameterList()
        for i in range(n_heads):
            W_q = nn.Parameter(torch.zeros(d_filters, self.d_k))
            nn.init.xavier_uniform_(W_q)
            self.W_q.append(W_q)
            W_k = nn.Parameter(torch.zeros(d_filters, self.d_k))
            nn.init.xavier_uniform_(W_k)
            self.W_k.append(W_k)
            W_v = nn.Parameter(torch.zeros(d_filters, self.d_v))
            nn.init.xavier_uniform_(W_v)
            self.W_v.append(W_v)
        self.W_o = nn.Parameter(torch.zeros(d_filters, self.d_v * n_heads))
        nn.init.xavier_uniform_(self.W_o)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, mask):
        batch_size, sentence_length, _ = x.size()
        att_heads = torch.zeros(batch_size, sentence_length, self.d_v)
        multihead = torch.zeros(batch_size, sentence_length, 0)
        normalize = 1 / math.sqrt(self.d_k)
        for i in range(self.n_heads):
            Q = torch.add(torch.matmul(x, self.W_q[i]), self.bias)
            ##print("Self Attention Q = " + str(Q.size()))
            K = torch.add(torch.matmul(x, self.W_k[i]), self.bias)
            ##print("Self Attention K = " + str(K.size()))
            V = torch.add(torch.matmul(x, self.W_v[i]), self.bias)
            ##print("Self Attention V = " + str(V.size()))
            out = torch.bmm(Q, K.permute(0,2,1))
            ##print("Self Attention QK.T = " + str(out.size()))
            out = torch.mul(out, normalize)
            ##print("Self Attention QK.T/sqrt(d_k) = " + str(out.size()))
            ##print("Self Attention Mask = " + str(mask.size()))
            out = masked_softmax(out, mask.view(batch_size, 1, out.size(1)), dim=2)
            ##print("Self Attention softmax(QK.T/sqrt(d_k)) = " + str(out.size()))
            att_heads = torch.bmm(out, V)
            ##print("Self Attention (softmax(QK.T/sqrt(d_k)))V = " + str(att_heads.size()))
            multihead = torch.cat((att_heads, multihead), dim = 2)
        ##print("Self Attention Concat MultiHeads = " + str(multihead.size()))
        out = F.dropout(torch.add(torch.matmul(multihead, self.W_o.permute(1,0)), self.bias), self.drop_prob, self.training)
        ##print("Self Attention Final Output = " + str(out.size()))
        return out
    
class FeedForward(nn.Module):
    """Initializing Feed Forward Layer"""
    def __init__(self, drop_prob, d_filters = 128):
        super(FeedForward, self).__init__()
        self.d_filters = d_filters
        self.drop_prob = drop_prob
        self.ffn = nn.Linear(self.d_filters, self.d_filters, bias = True)
    def forward(self, x):
        ##print("FFN x = " + str(x.size()))
        out = F.relu(self.ffn(x))
        out = self.ffn(out)
        ##print("FFN x with relu = " + str(out.size()))
        out = F.dropout(out, self.drop_prob, self.training)
        ##print("FFN final = " + str(out.size()))
        return out
        
class LayerNorm(nn.Module):
    """Initializing Layer Normalization
    Based on the paper:
    "Layer Normalization"
    by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    (https://arxiv.org/pdf/1607.06450.pdf)
    """
    def __init__(self, drop_prob, d_filters):
        super(LayerNorm, self).__init__()
        self.drop_prob = drop_prob
        self.d_filters = d_filters
        self.layer_norm = nn.LayerNorm(self.d_filters)
    def forward(self, x):
        ##print("LayerNorm x = " + str(x.size()))
        x = self.layer_norm(x)
        ##print("LayerNorm output = " + str(x.size()))
        return x
        
class ResidualBlock(nn.Module):
    """Initializing Residual Block f(layernorm(x)) + x
    """
    def __init__(self, drop_prob):
        super(ResidualBlock, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x, residual):
        ##print("Residual Block x = " + str(x.size()))
        ##print("Residual Block residual = " + str(residual.size()))
        x = x + residual
        ##print("Residual Block final Output = " + str(x.size()))
        return x
    

class EncoderBlock(nn.Module):
    """"Initializing Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters = 128, drop_prob = 0.1):
        super(EncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.n_conv = n_conv
        self.d_filters = d_filters
        self.pos_enc = PositionalEncoding(self.d_filters, self.drop_prob)
        self.layernorm = nn.ModuleList([LayerNorm(self.drop_prob, self.d_filters) for i in range(self.n_conv+2)])
        self.conv = nn.ModuleList([DepthSepCNN(self.drop_prob, self.d_filters, self.d_filters, kernel_size) for i in range(self.n_conv)])
        self.self_attention = SelfAttention(self.drop_prob, self.d_filters)
        self.ffn = FeedForward(self.drop_prob, self.d_filters)
        self.residual = ResidualBlock(self.drop_prob)
    def forward(self, x, mask):
        # Positional Encoding Block
        out = self.pos_enc(x)
        # Convolutional Block
        for i in range(self.n_conv):
            residual = out
            out = self.layernorm[i](out)
            out = self.conv[i](out.permute(0,2,1)).permute(0,2,1)
            out = self.residual(out, residual)
        # Self Attention Block
        residual = out
        out = self.layernorm[-2](out)
        out= self.self_attention(out, mask)
        out = self.residual(out, residual)
        #print((out.permute(0,2,1))[0,0,:])
        #print((out.permute(0,2,1)).size())
        # Feed Forward Block
        residual = out
        out = self.layernorm[-1](out)
        out = self.ffn(out)
        out = self.residual(out, residual)
        return out
    
class EmbeddingEncoder(nn.Module):
    """Create Embedding Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters, drop_prob, n_blocks=1, embed_size = 500):
        super(EmbeddingEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_conv = n_conv
        #self.conv = DepthSepCNN(drop_prob, embed_size, d_filters, kernel_size)
        self.enc_blocks = EncoderBlock(n_conv, kernel_size, d_filters, drop_prob)
        #self.enc_blocks = nn.ModuleList([EncoderBlock(n_conv, kernel_size, d_filters, drop_prob) for i in range(n_blocks)])
    def forward(self, x, mask):
        ##print("Embedding Encoder Input = " + str(x.size()))
        #x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        ##print("Embedding Encoder after conv = " + str(x.size()))
        for i in range(self.n_blocks):
            x = self.enc_blocks(x, mask)
        out = x
        ##print(out[0,0,:])
        return out

class ModelEncoder(nn.Module):
    """Create Model Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters = 128, drop_prob = 0.1, n_blocks = 7):
        super(ModelEncoder, self).__init__()
        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.drop_prob = drop_prob
        self.conv = DepthSepCNN(drop_prob, d_filters*4, d_filters, kernel_size)
        #self.enc_blocks = EncoderBlock(n_conv, kernel_size, d_filters, drop_prob)
        self.enc_blocks = nn.ModuleList([EncoderBlock(n_conv, kernel_size, d_filters, drop_prob) for i in range(3)])
    def forward(self, x, mask):
        ##print("Model Encoder Input = " + str(x.size()))
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("Model Encoder Output M0 Starting")
        for i in range(self.n_blocks):
            x = self.enc_blocks[0](x, mask)
        M0 = x
        ##print(M0[0,0,:])
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("Model Encoder Output M Starting")
        for i in range(self.n_blocks):
            x = self.enc_blocks[1](x, mask)
        M1 = x
        #print(M1[0,0,:])
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("Model Encoder Output M2 Starting")
        for i in range(self.n_blocks):
            x = self.enc_blocks[2](x, mask)
        M2 = x
        #print(M2[0,0,:])
        ##print("Model Encoder Output M0 = " + str(M0.size()))
        ##print("Model Encoder Output M1 = " + str(M1.size()))
        ##print("Model Encoder Output M2 = " + str(M2.size()))
        return M0, M1, M2
        
        
##################################End of QANet Layers##############################################


class CQAttention(nn.Module):
    """Context-Query attention.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(CQAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        ##print("CQ Attention Context Input = " + str(c.size()))
        ##print("CQ Attention Query Input = " + str(c.size()))
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        ##print("CQ Attention Similarity Matrix = " + str(s.size()))
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("CQ Attention Output = " + str(x.size()))
        ##print(x[0,0,:])
        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(QANetOutput, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.W1 = nn.Linear(self.hidden_size, 1)
        self.W2 = nn.Linear(self.hidden_size, 1)

    def forward(self, M0, M1, M2, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.W1(torch.cat([M0, M1], dim=-1))
        logits_2 = self.W2(torch.cat([M0, M2], dim=-1))
        ##print("QANet Output logits_1 = " + str(logits_1.size()))
        ##print("QANet Output logits_2 = " + str(logits_2.size()))
        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        ##print("QANet Output log_p1 = " + str(log_p1.size()))
        ##print("QANet Output log_p2 = " + str(log_p2.size()))

        return log_p1, log_p2
