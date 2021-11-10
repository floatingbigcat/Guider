import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import H_matrix

#---------------Spatial Section-----------------#

class GCNLayer(nn.Module):
  """
  original GCN
  """
  def __init__(self,in_feature,out_feature):
      super(GCNLayer,self).__init__()
      self.mlp = torch.nn.Linear(in_feature, out_feature)

  def forward(self,x,A):
      adj = A + torch.eye(A.size(0)).to(x.device)
      d = torch.sqrt(adj.sum(1))
      a = 1/d.view(-1, 1) * A /d.view(-1, 1)
      x = torch.einsum('nwcl,vw->nvcl',(x,a))
      #B N T D
      x = self.mlp(x)
      x = F.relu(x)
      return x


class H_GCNLayer(nn.Module):
  """
  Hierarchy GCN
  """
  def __init__(self, in_feature, out_feature,args):
    super(H_GCNLayer, self).__init__()
    self.gcn1 = GCNLayer(in_feature, out_feature)
    self.gcn2 = GCNLayer(in_feature, out_feature)
    self.mlp1 = nn.Linear(in_feature, out_feature)
    self.mlp2 = nn.Linear(in_feature, out_feature)
    self.hier = args.hier
    self.assign_matrix = ()
    if args.hier == True:
        h = np.load(args.hier_data,allow_pickle=True)
        self.gcn_h = GCNLayer(in_feature,len(h))
        self.H_matrix = H_matrix(h)
    else:
        self.gcn_h = GCNLayer(in_feature, args.next_nodes_num)
        self.H_matrix = None # without predefine hierarchey

  def assignment(self, x, adj):
    s = self.gcn_h(x, adj)
    if self.H_matrix != None: #with H data
        assignment = F.normalize(self.H_matrix.float(),p=1,dim=1).squeeze()
    else:
        assignment = torch.softmax(s, dim=-1)
    return assignment

  def dense_pool(self, z, adj, s):   
    """
    compute next level Adj
    """
    out = torch.einsum('abcd,bf->afcd',(z,s))
    out_adj = torch.mm(torch.mm(s.T,adj),s)
    return out, out_adj

  def forward(self,x, A):
      output = []
      x = self.gcn1.forward(x,A)
      output.append(x)
      if self.hier:# using hierarchicay info
        s = self.assignment(x,A).to(x.device)
        x1,A1 = self.dense_pool(x,A,s)
        x1 = self.gcn2.forward(x1,A1)
        x = torch.einsum('abcd,fb->afcd',(x1,s))
        output.append(x)
        output = self.mlp1(output[0])+self.mlp2(output[1])
        return output.contiguous()
      output = self.mlp1(output[0])
      return output.contiguous()


class Encoder_Spatial(nn.Module):
  def __init__(self, in_feature, out_feature, num_layers,args):
    super(Encoder_Spatial, self).__init__()
    self.GCNlayers = nn.ModuleList(
            [H_GCNLayer(in_feature, out_feature,args)
            for l in range(num_layers)]
            )

  def forward(self, x, adj):
    for gc in zip(self.GCNlayers):
      x = gc[0](x,adj)
    return x

#---------------Temporal Section-----------------#

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