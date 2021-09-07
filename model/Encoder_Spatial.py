import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("..") 
from utils.tools import H_matrix
class GCNLayer(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(GCNLayer,self).__init__()
        self.mlp = torch.nn.Linear(in_feature, out_feature)
    def forward(self,x,A):
        if A.dim() == 2:# nodes share same A
            adj = A + torch.eye(A.size(0)).to(x.device)
            d = torch.sqrt(adj.sum(1))
            a = 1/d.view(-1, 1) * A /d.view(-1, 1)
            x = torch.einsum('nwcl,vw->nvcl',(x,a))
            #B N T D
            x = self.mlp(x)
            x = F.relu(x)
            return x
        else: #A are different
            adj = A.transpose(1,2) + torch.eye(A.size(1)).to(x.device)
            d = torch.sqrt(adj.sum(3))

            a = torch.einsum('abf,abcd-> abfd',1/d,adj)
            a = torch.einsum('abcd,abf-> abfd',a,d)

            x = torch.einsum('abcd,acbf->afcd',x,a)
            x = self.mlp(x)
            x = F.relu(x)
            return x
class H_GCNLayer(nn.Module):
    def __init__(self, in_feature, out_feature,args):
        super(H_GCNLayer, self).__init__()
        self.gcn1 = GCNLayer(in_feature, out_feature)
        self.gcn2 = GCNLayer(in_feature, out_feature)
        self.mlp1 = nn.Linear(in_feature, out_feature)
        self.mlp2 = nn.Linear(in_feature, out_feature)
        self.hier = args.hier
        self.assign_matrix = ()
        if args.pre_hier == True:
            h = np.load(args.hier_data,allow_pickle=True)
            self.gcn_h = GCNLayer(in_feature,len(h))
            self.H_matrix = H_matrix(h)
        else:
            self.gcn_h = GCNLayer(in_feature, args.next_nodes_num)
            self.H_matrix = None
             # without predefine hierarchey

    def assignment(self, x, adj):
        s = self.gcn_h(x, adj)
        if self.H_matrix != None: #with H data
            # s = torch.sigmoid(s).transpose(1,2)
            # assignment = s.mul(self.H_matrix.squeeze().to(x.device))
            # assignment = F.normalize(assignment,p=1,dim=2).transpose(1,2)
            assignment = F.normalize(self.H_matrix.float(),p=1,dim=1).squeeze()
        else:
            assignment = torch.softmax(s, dim=-1)
            
        return assignment
    def dense_pool(self, z, adj, s):   
        if s.dim()==4:
            out = torch.einsum('abcd,abcf->afcd',(z,s))
            out_adj = torch.einsum('abcd,bf->afcd',s,adj)
            out_adj = torch.einsum('abcd,abcf->afcd',out_adj,s)
        else:
            out = torch.einsum('abcd,bf->afcd',(z,s))
            out_adj = torch.mm(torch.mm(s.T,adj),s)
        return out, out_adj
    def forward(self,x, A):

        output = []
        x = self.gcn1.forward(x,A)
        output.append(x)
        if self.hier:# 1 hierarchicay
            s = self.assignment(x,A).to(x.device)
            x1,A1 = self.dense_pool(x,A,s)
            x1 = self.gcn2.forward(x1,A1)

            if s.dim()==4:
                x = torch.einsum('abcd,adce->abce',(s,x1))
            else:
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
        a = x
        for gc in zip(self.GCNlayers):
            x = gc[0](x,adj)
        # x = F.log_softmax(x,-1)
        return x

if __name__ == '__main__':
    m = Encoder_Spatial(in_feature = 40,out_feature = 40,num_layers = 1)
    #m = GCNLayer(in_feature = 40,out_feature = 40)
    #input = rand(batch_size = 8, Node_nums = 11, Time_seq = 96, Node_dims = 40)
    input = torch.rand(8,11,96,40)
    adj = torch.rand(11,11)
    output = m(input,adj)
    print(output.shape)