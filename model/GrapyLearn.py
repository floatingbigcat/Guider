
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def Matrix_factorization(N,R,device,steps = 10,alpha=0.1,beta=0.02):
    P = N.to(device)
    Q = torch.rand((R.shape[0],R.shape[-1],N.shape[1])).to(device)
    # P.requires_grad = True
    P.requires_grad = True
    Q.requires_grad = True
    optimizer = torch.optim.Adam((P,Q),lr=alpha)

    result = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = ((R - torch.einsum('ab,cdb -> cad',P,Q))**2).sum()# + beta/2*(torch.norm(P,2)+torch.norm(Q,2))
        loss.backward()
        optimizer.step()

        result.append(loss)
        if(loss < 0.1):break
        #print("*********loss = {.%3f}*********",math.sqrt(loss.item()))
    return P,Q.mean(0)
class graph_constructor(nn.Module):
    def __init__(self,num_nodes,dim, device, k = 0.5,alpha=3): # k is saving rate of adj// dim is node's orl dim
        super(graph_constructor, self).__init__()

        self.line1 = nn.Linear(dim,dim)
        self.line2 = nn.Linear(dim,dim)

        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)
        
        #print('Before模型参数parameter:\t', self._parameters)
        
        self.N_matrix = torch.rand((num_nodes,dim)).to(device)
        # self.N_matrix,self.T_matrix = self.Matrix_factorization(self.dat,dim,device)
        self.N_matrix = nn.Parameter(self.N_matrix)
        # self.T_matrix = nn.Parameter(self.T_matrix)
        self.device = device
        self.alpha = alpha
        self.k = k

        self.emb1 = nn.Embedding(num_nodes, dim)
        self.emb2 = nn.Embedding(num_nodes, dim)
    def forward(self,x_enc):
        torch.is_grad_enabled()
        
        if torch.is_grad_enabled():
            N_m,T_m = Matrix_factorization(self.N_matrix,x_enc[:,:,:,0],self.device)
        else:
            torch.set_grad_enabled(True)
            N = torch.tensor(self.N_matrix)
            N_m,T_m = Matrix_factorization(N,x_enc[:,:,:,0],self.device)
            torch.set_grad_enabled(False)
        
        T_m = T_m[-1,:]
        N_m = self.N_matrix
        
        nodevec1 = torch.relu(self.line1(N_m)+self.line2(T_m.repeat(N_m.shape[0],1)))
        nodevec2 = torch.relu(self.line1(N_m)+self.line2(T_m.repeat(N_m.shape[0],1)))
        
        
        # idx = torch.tensor(range(305)).to(self.device)
        # nodevec1 = self.emb1(idx)
        # nodevec2 = self.emb2(idx)
        
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(N_m.size(0), N_m.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(int(self.k * N_m.size(0)),1) #save top self.k 
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

        # a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        # adj = F.relu(torch.tanh(self.alpha*a))
        # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))
        # s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(int(self.k * len(idx)),1)
        # mask.scatter_(1,t1,s1.fill_(1))
        # adj = adj*mask
        # return adj

if __name__ == '__main__':
    data = r'data/sale_data_for Guider.npy'
    dat = np.load(data)
    m = graph_constructor(dat,12,"cuda:0")
    m = m.to("cuda:0")
    adj = m(8)
    print(adj.shape)

    