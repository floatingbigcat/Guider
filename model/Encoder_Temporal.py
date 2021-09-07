import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attn import *
import numpy as np
from math import sqrt

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(AttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        B,N,T,D = x.shape
        x = x.reshape(B*N,T,-1)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y).reshape(B,N,T,D), attn

class ConvLayer(nn.Module):
    def __init__(self,c_in):
        super(ConvLayer,self).__init__()
        self.downConv = nn.Conv2d(in_channels=c_in,
                                  out_channels=c_in,
                                  padding = (1,0),
                                  kernel_size=(3,1),
                                )
        self.norm = nn.BatchNorm2d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1), padding=(1,0))

    def forward(self,x):
        x = self.downConv(x.permute(0,3,2,1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,3)
        return x

class Encoder_Temporal(nn.Module):
    def __init__(self,attn_layers,conv_layers,norm_layer = None):
        super(Encoder_Temporal,self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer
    def forward(self, x, attn_mask = None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
        attns.append(attn)
        x = self.norm(x)
        return x, attns

if __name__ == '__main__':
    layers = 2
    d_model = 40
    m = Encoder_Temporal(
        [AttentionLayer(
            Attention(
                ProbAttention(mask_flag = False), 
                d_model, n_heads = 4, mix=False),
                d_model)for l in range(layers)],
        [ConvLayer(d_model)for l in range(layers-1)],
        norm_layer=torch.nn.LayerNorm(d_model)
        )
    x = torch.rand(8,11,96,40)
    print(x.shape)
    out,attn= m(x)
    print(out.shape)





