import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.Attn import *
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, self_attention,cross_attention, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4*d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         B,N,T,D = x.shape
#         _,_,t,d = cross.shape
#         x = x.reshape(B*N,T,-1)       
#         cross = cross.reshape(B*N,t,-1)  
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x = self.norm1(x)

#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])

#         y = x = self.norm2(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
#         y = self.dropout(self.conv2(y).transpose(-1,1))

#         return self.norm3(x+y).reshape(B,N,T,D)

# class Decoder(nn.Module):
#     def __init__(self, d_model,num_nodes,n_heads,d_layers,dropout=0.1) -> None:
#         super(Decoder,self).__init__()
#         self.decoder = nn.ModuleList(
#             [
#                 DecoderLayer(d_model = d_model,
#                             self_attention = Attention(ProbAttention(True, attention_dropout=dropout, output_attention=False), 
#                                 d_model,n_heads),
#                             cross_attention = Attention(FullAttention(False, attention_dropout=dropout, output_attention=False), 
#                                 d_model, n_heads, mix=False),)
#                 for i in range(d_layers)
#             ]
#         )
#     def forward(self,x,cross):
#         for decoderlayer in zip(self.decoder):
#             x = decoderlayer[0](x,cross)
#         # x = F.layer_norm(x,x.shape[0:])
#         return x

class Decoder(nn.Module):
    def __init__(self,d_model,seq_in,seq_out,e_layers,c_out=1):
        super().__init__()
        # self.T1_decoder= nn.Linear(int(seq_in/(2**(e_layers))),int(seq_out/2)) 
        # self.T2_decoder = nn.Linear(int(seq_out/2),seq_out)
        self.T_decoder= nn.Linear(math.ceil(seq_in/(2**(e_layers))),seq_out)

        self.D1_decoder = nn.Linear(d_model, int(d_model/2), bias=True)
        self.D2_decoder = nn.Linear(int(d_model/2), c_out, bias=True)
    def forward(self,enc_out):
        
        # dec_out = F.relu(self.T1_decoder(enc_out.transpose(3,2)))
        # dec_out = self.T2_decoder(dec_out)
        dec_out = self.T_decoder(enc_out.transpose(3,2))
        dec_out = F.relu(self.D1_decoder(dec_out.transpose(3,2)))
        dec_out = self.D2_decoder(dec_out)

        return dec_out

if __name__ == '__main__':

    cross = torch.rand(8,11,24,40)
    input = torch.rand(8,11,13,40)
    adj = torch.rand(11,11)
    m = Decoder(d_model=40,n_heads=4,d_layers = 2)
    output = m (input,cross)
    print(output.shape)