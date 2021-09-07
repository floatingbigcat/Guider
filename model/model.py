import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.embed import DataEmbedding
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.GrapyLearn import graph_constructor
class Guider(nn.Module):
    def __init__(self,data,d_model, num_nodes,c_out,device,n_heads,seq_in,args,pre_hier=None,e_layers=3,d_layers=2) -> None:
        super(Guider,self).__init__()
        self.GraphLearn = graph_constructor(num_nodes,d_model,device)
        self.enc_embedding = DataEmbedding(d_model, d_model)
        self.dec_embedding = DataEmbedding(d_model, d_model)

        self.encoder = Encoder(d_model,n_heads=n_heads,num_nodes=num_nodes,seq_in=seq_in,e_layers=e_layers,pre_hier=pre_hier,args=args)
        self.decoder = Decoder(d_model,seq_in=seq_in,seq_out=args.seq_out,e_layers=e_layers)
        # self.decoder = nn.Linear(int(seq_in/(2**(e_layers))),args.seq_out)


        #decoder oral:
        # self.projection = nn.Linear(d_model, c_out, bias=True)
        #decoder complex:
        # self.projection = nn.Linear(d_model, int(d_model/2), bias=True)
        # self.mlp = nn.Linear(int(d_model/2), c_out, bias=True)
    def forward(self,x_enc,Y):

        adj = self.GraphLearn(x_enc)
        x_enc = self.enc_embedding(x_enc)
        # x_dec = self.dec_embedding(x_dec)
        enc_out = self.encoder(x_enc,adj)
        # dec_out = self.decoder(x = x_dec, cross = enc_out)
        dec_out = self.decoder(enc_out) 

        # dec_out = F.relu(self.projection(dec_out.transpose(3,2)))#add relu & mlp! 9/6/00点06分
        # dec_out = self.mlp(dec_out)


        Yhat = dec_out.squeeze(-1)
        Y = Y[:,:,:,0]

        # Yhat = F.layer_norm(Yhat,Yhat.shape[0:])
        # Y = F.layer_norm(Y,Y.shape[0:])
        
        return Yhat,Y
    
if __name__ == '__main__':
    input = torch.rand(8,11,96,40)
    x_dec = torch.rand(8,11,14,40)
    m = Guider(d_model=40,c_out = 20,n_heads=4,num_nodes=11)
    output = m (input,x_dec)
    print(output.shape)