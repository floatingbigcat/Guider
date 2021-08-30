import torch

def H_matrix(h):
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
    return my_out