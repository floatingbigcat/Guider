import torch
import torch.nn as nn
from model.Attn import Attention,ProbAttention
from model.Encoder_layers import Encoder_Temporal,Encoder_Spatial,AttentionLayer,ConvLayer

class EncoderLayer(nn.Module):
  #Martin Sun @ DLUT
  def __init__(self,d_model,num_nodes, n_heads, pre_hier,args,gcn_layers=1):
    super(EncoderLayer,self).__init__()
    self.num_nodes = num_nodes
    self.d_model = d_model
    self.n_heads = n_heads
    self.alpha = args.alpha
    self.e_layers = 2 
    self.ES = Encoder_Spatial(in_feature = d_model,out_feature = d_model,num_layers= gcn_layers,args = args)
    self.ET = Encoder_Temporal(
        [AttentionLayer(
            Attention(
                ProbAttention(mask_flag = False), 
                d_model, n_heads = n_heads, mix=False),
                d_model)for l in range(self.e_layers)],
        [ConvLayer(d_model)for l in range(self.e_layers-1)],
        norm_layer=torch.nn.LayerNorm(d_model)
        )

  def forward(self, x, adj):
    et_out, _ = self.ET(x)
    es_out =self.ES(et_out,adj)
    output = self.alpha*et_out + (1-self.alpha)*es_out
    return output
 

class Encoder(nn.Module):
  #Martin Sun @ DLUT
  def __init__(self, d_model, num_nodes, n_heads,e_layers,seq_in,pre_hier,args) -> None:
    super().__init__()
    self.layers = e_layers
    self.encoder = nn.ModuleList(
      [
        EncoderLayer(d_model = d_model,n_heads=n_heads,num_nodes=num_nodes,pre_hier=pre_hier,args=args)
        for i in range(e_layers)
      ]
    )
    self.residual_conv = nn.ModuleList(
      [
        nn.Linear(in_features=int(seq_in*(2**(-i))),out_features= int(seq_in*(2**(-i-1))))
        for i in range(e_layers)
      ]
    )

  def forward(self,x,adj):
    for i in range(self.layers):
      residual = self.residual_conv[i](x.transpose(-1,-2))
      x = self.encoder[i](x, adj)
      # residual connection
      x = x + residual.transpose(-1,-2)
    return x
