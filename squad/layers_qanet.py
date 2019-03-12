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
        #self.model_highway = Highway(self.embed_size)

    def forward(self, x):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (batch_size, sentence_length, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (batch_size, sentence_length, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        x_emb = F.dropout(self.embeddings(x), self.drop_prob, self.training)
        (batch_size, sentence_length, max_word_length, e_char) = x_emb.size()
        x_reshaped = (x_emb.view(batch_size*sentence_length, max_word_length, e_char)).permute(0,2,1)
        x_conv_out = self.model_cnn(x_reshaped)
        #x_highway = self.model_highway(x_conv_out)
        x_word_emb = x_conv_out.view(batch_size, sentence_length, self.embed_size)
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
        x_conv_out = F.relu(self.pointwise(x_conv))
        ##print("x after pointwise CNN = " + str(x_conv_out.size()))
        return x_conv_out

class SelfAttention(nn.Module):
    """Implementing Multi-head Attention.
    Based on the paper:
    "Attention is all you need."
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/pdf/1706.03762.pdf)"""
    def __init__(self, device, drop_prob, d_filters = 128, n_heads = 8):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.device = device
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
        att_heads = []#torch.empty(self.n_heads, batch_size, sentence_length, self.d_v)#, device = self.device)
        #multihead = torch.zeros(batch_size, sentence_length, 0)
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
            att_heads.append(torch.bmm(out, V))
            ##print("Self Attention (softmax(QK.T/sqrt(d_k)))V = " + str(att_heads.size()))
            #multihead = torch.cat((att_heads, multihead), dim = 2)
        multihead = torch.cat(att_heads, dim=2)#att_heads.permute(1,2,0,3).contiguous().view(batch_size, sentence_length, -1)
        ##print("Self Attention Concat MultiHeads = " + str(multihead.size()))
        out = torch.matmul(multihead, self.W_o)
        out = torch.add(out, self.bias)
        ##print("Self Attention Final Output = " + str(out.size()))
        return out
    
class FeedForward(nn.Module):
    """Initializing Feed Forward Layer"""
    def __init__(self, drop_prob, d_filters = 128):
        super(FeedForward, self).__init__()
        self.d_filters = d_filters
        self.drop_prob = drop_prob
        self.ffn1 = nn.Linear(self.d_filters, self.d_filters, bias = True)
        self.ffn2 = nn.Linear(self.d_filters, self.d_filters, bias = True)
    def forward(self, x):
        ##print("FFN x = " + str(x.size()))
        out = self.ffn2(F.relu(self.ffn1(x)))
        ##print("FFN x with relu = " + str(out.size()))
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
        return self.layer_norm(x)
        ##print("LayerNorm output = " + str(x.size()))
        
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
    def __init__(self, device, n_conv, kernel_size, d_filters = 128, drop_prob = 0.1):
        super(EncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.n_conv = n_conv
        self.d_filters = d_filters
        self.device = device
        self.pos_enc = PositionalEncoding(self.d_filters, self.drop_prob)
        self.layernorm = nn.ModuleList([LayerNorm(self.drop_prob, self.d_filters) for i in range(self.n_conv+2)])
        self.conv = nn.ModuleList([DepthSepCNN(self.drop_prob, self.d_filters, self.d_filters, kernel_size) for i in range(self.n_conv)])
        self.self_attention = SelfAttention(self.device, self.drop_prob, self.d_filters)
        self.ffn = FeedForward(self.drop_prob, self.d_filters)
        #self.residual = ResidualBlock(self.drop_prob)
    def forward(self, x, mask, l, blks):
        total_layers = (self.n_conv + 1) * blks
        dropout = self.drop_prob
        # Positional Encoding Block
        out = self.pos_enc(x)
        # Convolutional Block
        for i in range(self.n_conv):
            residual = out
            out = self.layernorm[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = self.conv[i](out.permute(0,2,1)).permute(0,2,1)
            out = self.layer_dropout(out, residual, dropout*float(l)/total_layers)
            l += 1
        # Self Attention Block
        residual = out
        out = self.layernorm[-2](out)
        out = F.dropout(out, p=dropout, training=self.training)
        out= self.self_attention(out, mask)
        out = self.layer_dropout(out, residual, dropout*float(l)/total_layers)
        l += 1
        #print((out.permute(0,2,1))[0,0,:])
        #print((out.permute(0,2,1)).size())
        # Feed Forward Block
        residual = out
        out = self.layernorm[-1](out)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.ffn(out)
        out = self.layer_dropout(out, residual, dropout*float(l)/total_layers)
        return out
    
    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
    
class EmbeddingEncoder(nn.Module):
    """Create Embedding Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters, drop_prob, n_blocks=1, embed_size = 500):
        super(EmbeddingEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_conv = n_conv
        #self.conv = DepthSepCNN(drop_prob, embed_size, d_filters, kernel_size)
        self.enc_blocks = XL_EncoderBlock(n_conv, kernel_size, d_filters)
        #self.enc_blocks = nn.ModuleList([EncoderBlock(n_conv, kernel_size, d_filters, drop_prob) for i in range(n_blocks)])
    def forward(self, x, mems, mask):
        ##print("Embedding Encoder Input = " + str(x.size()))
        #x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        ##print("Embedding Encoder after conv = " + str(x.size()))
        if mems is None: mems = self.init_mems()
        for i in range(self.n_blocks):
            x, mems = self.enc_blocks(x, mems, mask, 1, 1)
        out = x
        ##print(out[0,0,:])
        return out, mems
    
    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

class ModelEncoder(nn.Module):
    """Create Model Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters = 128, drop_prob = 0.1, n_blocks = 7):
        super(ModelEncoder, self).__init__()
        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.drop_prob = drop_prob
        self.conv = DepthSepCNN(drop_prob, d_filters*4, d_filters, kernel_size)
        #self.enc_blocks = EncoderBlock(n_conv, kernel_size, d_filters, drop_prob)
        self.enc_blocks = nn.ModuleList([XL_EncoderBlock(n_conv, kernel_size, d_filters) for i in range(3)])
    def forward(self, x, mems, mask):
        ##print("Model Encoder Input = " + str(x.size()))
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("Model Encoder Output M0 Starting")
        for i in range(self.n_blocks):
            x, mems = self.enc_blocks[0](x, mems, mask, i*(2+2)+1, 7)
        M0 = x
        #print("M0 shape = " + str(M0.size()))
        ##print(M0[0,0,:])
        ##print("Model Encoder Output M Starting")
        for i in range(self.n_blocks):
            x, mems = self.enc_blocks[1](x, mems, mask, i*(2+2)+1, 7)
        M1 = x
        #print(M1[0,0,:])
        x = F.dropout(x, self.drop_prob, self.training)
        ##print("Model Encoder Output M2 Starting")
        for i in range(self.n_blocks):
            x, mems = self.enc_blocks[2](x, mems, mask, i*(2+2)+1, 7)
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

################################Incorporating Transformer-XL in QANet#####################################

class XL_PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(XL_PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class XL_RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(XL_RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class XL_RelPartialLearnableMultiHeadAttn(XL_RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(XL_RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class XL_PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(XL_PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class XL_EncoderBlock(nn.Module):
    """"Initializing Encoder Block"""
    def __init__(self, n_conv, kernel_size, d_filters = 128, drop_prob = 0.1, n_head = 8, d_head = 50, tgt_len = 70, ext_len = 0, mem_len = 0, dropatt = 0.0, d_inner = 1000):
        super(XL_EncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.n_conv = n_conv
        self.d_filters = d_filters
        self.n_head = n_head
        self.d_head = d_head
        self.pos_enc = XL_PositionalEmbedding(self.d_filters)
        self.layernorm = nn.ModuleList([LayerNorm(self.drop_prob, self.d_filters) for i in range(self.n_conv)])
        self.conv = nn.ModuleList([DepthSepCNN(self.drop_prob, self.d_filters, self.d_filters, kernel_size) for i in range(self.n_conv)])
        self.self_attention = XL_RelPartialLearnableMultiHeadAttn(n_head, self.d_filters, d_head, self.drop_prob, tgt_len, ext_len, mem_len, dropatt, True)
        self.ffn = XL_PositionwiseFF(self.d_filters, d_inner, self.drop_prob, True)
        self._create_params()
        self.drop = nn.Dropout(drop_prob)
        #self.residual = ResidualBlock(self.drop_prob)
    def forward(self, x, mems, mask, l, blks):
        total_layers = (self.n_conv + 1) * blks
        dropout = self.drop_prob
        # Positional Encoding Block
        out = x#self.pos_enc(x)
        # Convolutional Block
        for i in range(self.n_conv):
            residual = out
            out = self.layernorm[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = self.conv[i](out.permute(0,2,1)).permute(0,2,1)
            out = self.layer_dropout(out, residual, dropout*float(l)/total_layers)
            l += 1
        # Self Attention Block
        mlen = mems.size(0) if mems is not None else 0
        qlen = out.size(1)
        klen = mlen + qlen
        pos_seq = torch.arange(klen-1, -1, -1.0, device=out.device, dtype=out.dtype)
        pos_emb = self.pos_enc(pos_seq)
        core_out = self.drop(out)
        pos_emb = self.drop(pos_emb)
        mems_i = None if mems is None else mems[0]
        out = self.self_attention(core_out, pos_emb, self.r_w_bias,self.r_r_bias, dec_attn_mask=mask, mems=mems_i)
        hids = []
        hids.append(out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        l += 1
        #print((out.permute(0,2,1))[0,0,:])
        #print((out.permute(0,2,1)).size())
        # Feed Forward Block
        out = self.ffn(out)
        out = self.layer_dropout(out, residual, dropout*float(l)/total_layers)
        return out, new_mems
    
    def _create_params(self):
            self.pos_emb = XL_PositionalEmbedding(self.d_filters)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
    
    
    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems
    
    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual   
##########################################################################################################