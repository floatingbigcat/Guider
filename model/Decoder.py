import math
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,d_model,seq_in,seq_out,e_layers,c_out=1):
      super().__init__()
      self.T_decoder= nn.Linear(math.ceil(seq_in/(2**(e_layers))),seq_out)
      self.D1_decoder = nn.Linear(d_model, int(d_model/2), bias=True)
      self.D2_decoder = nn.Linear(int(d_model/2), c_out, bias=True)

    def forward(self,enc_out):
      dec_out = self.T_decoder(enc_out.transpose(3,2))
      dec_out = F.relu(self.D1_decoder(dec_out.transpose(3,2)))
      dec_out = self.D2_decoder(dec_out)
      return dec_out
