import torch
import torch.nn as nn
from utils.embed import DataEmbedding
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.GrapyLearn import graph_constructor


class Guider(nn.Module):
  def __init__(self,d_model, num_nodes,device,n_heads,seq_in,args,pre_hier=None,e_layers=3,d_layers=2) -> None:
    super(Guider,self).__init__()
    self.graphLearn = graph_constructor(num_nodes,d_model,device)
    self.enc_embedding = DataEmbedding(d_model, d_model)
    self.encoder = Encoder(d_model,n_heads=n_heads,num_nodes=num_nodes,seq_in=seq_in,e_layers=e_layers,pre_hier=pre_hier,args=args)
    self.decoder = Decoder(d_model,seq_in=seq_in,seq_out=args.seq_out,e_layers=e_layers)

  def forward(self,x_enc):
    adj = self.GraphLearn(x_enc)
    x_enc = self.enc_embedding(x_enc)
    enc_out = self.encoder(x_enc,adj)
    dec_out = self.decoder(enc_out) 
    Yhat = dec_out.squeeze(-1)
    return Yhat