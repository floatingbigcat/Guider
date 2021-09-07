
import torch
import torch.nn as nn
import torch.nn.functional as F
class diff_pool(torch.nn.Module):
    def __init__(self, node_feature, next_node_num):
        super(diff_pool, self).__init__()
        self.gnn1 = nn.Linear(node_feature, node_feature)
        self.gnn2 = nn.Linear(node_feature, next_node_num)

    def my_assignment2(self, h):
        l = []
        for i in h:
            for j in i:
                l.append(j)
        l = list(set(l))
        my_out = torch.zeros((len(l), len(h)), dtype=torch.long)

        for i in range(len(h)):
            for j in h[i]:
                my_out[j][i] = 1    
        my_out = torch.unsqueeze(my_out, 0)

        # d = torch.sum(my_out, dim=-1)
        # my_out = my_out.transpose(1,2)
        # s = torch.div(my_out, d)
        # my_out = s.transpose(1,2)
    
        return my_out

    def dense_diff_pool(self, x, adj, s, mask=None, EPS = 1e-15):
        
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()

        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / adj.numel()

        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

        return out, out_adj, link_loss, ent_loss

    
    def forward(self, x, adj, hierarchical = None, adj1 = None):
        # if hierarchical is None:
        z = self.gnn1(x)       
        s = self.gnn2(x)  #(6,3)

        S = torch.sigmoid(s)

        # s = torch.softmax(s, dim=-1)  #(6,3)
        out, out_adj, link_loss, ent_loss = self.dense_diff_pool(x, adj, s)
        if hierarchical is None:
            return out, out_adj, link_loss, ent_loss
        else:
            own_assignment2 = self.my_assignment2(hierarchical)
            own_ass_down = own_assignment2.squeeze(0) if own_assignment2.dim()==3 else x
            Assignment = own_ass_down.mul(S)
            Assignment = F.normalize(Assignment,p=1,dim=0)

            return out, out_adj, link_loss, ent_loss, Assignment



d = diff_pool(node_feature=5, next_node_num=3)

if __name__ == '__main__':
    x = torch.tensor([[2,1,3,4,1],[5,6,2,3,2],[12,0,7,9,7],
                    [2,1,5,4,1],[8,6,2,3,2],[3,7,9
                    ,4,4]], dtype=torch.float)
    A = torch.tensor([[0,1,0,0,0,1],
                        [1,0,1,0,0,0],
                        [0,1,0,1,0,0],
                        [0,0,1,0,1,0],
                        [0,0,0,1,0,0],
                        [1,0,0,0,0,1]],dtype=torch.float)
    A1 = torch.tensor([[0,0,1,1,2,2,3,3,4,5,5],
                 [1,5,0,2,1,3,2,5,3,0,5]], dtype=torch.long)   
    h = [[0,1,2],[2,5],[3,4]] 
    # f1 = d(x, A, hierarchical = None, adj1=A)  

    f2 = d(x, A, hierarchical = h, adj1=A)  
        