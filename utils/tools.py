import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.lr * int(0.8 **(epoch//100))}
    elif args.lradj=='type2':
        lr_adjust = {
            100: 1.5e-4, 180:1.4e-5,400:1.35e-4
        }
    elif args.lradj=='type3':
        lr_adjust = {epoch:args.lr}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

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

    # d = torch.sum(my_out, dim=-1)
    # my_out = my_out.transpose(1,2)
    # s = torch.div(my_out, d)
    # my_out = s.transpose(1,2)

    return my_out