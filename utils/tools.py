import torch
import numpy as np

def adjust_learning_rate(optimizer, epoch, args):
  if args.lradj=='type1':
    lr_adjust = {epoch: args.lr * int(0.5 **(epoch//300))}
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
  """
  get H_matrix from hier data
  """
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