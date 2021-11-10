import os
import torch
import argparse
from math import sqrt
import torch.nn as nn
from utils.metrics import *
from model.model import Guider
from torch.optim import optimizer
from utils.dataloader import DataLoader
from utils.tools import adjust_learning_rate
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Guider Time series forecasting')
parser.add_argument('--data', type=str, default='data/LZ_onehot.npy',
                    help='npy format. [items, time, dims] eg: 2247 items saledat in 37 months, 53 dim per record(onehot) -> npy.shape:[2247,37,53] ')
parser.add_argument('--hier_data', type=str, default='data/LZ_H_type.npy',
                    help='npy format.[hier_items,] eg:2247 items belong to 119 type, each type have diff num items->npy.shape:[type_num,] npy[i].shape:[num_items (i_th type)]')
parser.add_argument('-ms','--model save path', type=str, default='model_save/model_Guider.pt',
                    help='model file save path')
parser.add_argument('-ls','--logger folder save path', type=str,require=True, help='eg: ../result/')
parser.add_argument('-en','--experiment name', type=str,require=True, help='eg: ablation study with free Embedding')
parser.add_argument('--hier', type=bool, default=True,
                    help='True:Hierarchy GCN with hier_data. False:Normal GCN')
parser.add_argument('--Free',type=bool,default=False,help='with FreeEmbedding or not')
parser.add_argument('--next_nodes_num',type=int,default=7,help='number of nodes after pooled')
parser.add_argument('--num_nodes',type=int,default=2247,help='number of nodes/variables')
parser.add_argument('--d_model',type=int,default=53,help='input dimension per record')
parser.add_argument('--seq_in',type=int,default=6,help='input sequence length')
parser.add_argument('--seq_out',type=int,default=1,help='output sequence length')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('--lradj',type=str,default='type3',
                    help='[type1,type2,type3] || type1:lr * int(0.5 **(epoch//300)) type2: given lr in epoch type3: No adjust')
parser.add_argument('--n_heads',type=int,default=4,help='mult_head attentions')
parser.add_argument('--e_layer', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layer', type=int, default=1, help='num of decoder layers')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--delta', type=float, default=1e-6,help='rate of (R-PQ)**2 in loss')
parser.add_argument('--alpha', type=float, default=0.5,help='rate of et')
args = parser.parse_args()


def train(Data,model, criterion, optimizer, batch_size):
  model.train()
  train_loss = []
  for enc_X, Y in Data.get_batches(Data.train[0],Data.train[1],batch_size,shuffle=True):
    model.zero_grad() 
    Y = Y[:,:,:,0]
    Yhat = model(enc_X)
    Yhat.retain_grad
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    train_loss.append(sqrt(loss))
  return sum(train_loss)/len(train_loss)

def evaluate(Data,model, batch_size, flag):
    model.eval()
    mcr = [] # mean changing rate
    mse = []
    if flag == 'val':
      for enc_X, Y in Data.get_batches(Data.valid[0],Data.valid[1],batch_size,shuffle=False):
        with torch.no_grad():
          Y = Y[:,:,:,0]
          Yhat = model(enc_X)
          Y = Y*Data.std + Data.mean
          Yhat = Yhat*Data.std + Data.mean
        mcr.append(mCR(Yhat,Y))    
        mse.append(MSE(Yhat,Y))
    elif flag == 'test':
      for enc_X, Y in Data.get_batches(Data.test[0],Data.test[1],batch_size,shuffle=False):
        with torch.no_grad():
          Y = Y[:,:,:,0]
          Yhat = model(enc_X)
          Y = Y*Data.std + Data.mean
          Yhat = Yhat*Data.std + Data.mean
        mcr.append(mCR(Yhat,Y))  
        mse.append(MSE(Yhat,Y))
    evl_mse = sum(mse)/len(mse)
    evl_mcr = sum(mcr)/len(mcr)
    return  evl_mse,evl_mcr

if __name__ == "__main__":
    # train initial
    device = torch.device(args.device)
    dataloader = DataLoader(args.data, 0.6, 0.2, device = device, seq_out=args.seq_out, seq_in= args.seq_in,normalize = args.normalize)
    model = Guider(d_model=args.d_model,seq_in = args.seq_in,
                device = device,pre_hier = args.pre_hier,
                n_heads=args.n_heads,num_nodes=args.num_nodes,
                e_layers=args.e_layer,d_layers=args.d_layer,args = args)
    model = model.to(device) 
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    criterion = nn.MSELoss().to(device)
    evaluateL2 = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.01)
    writer = SummaryWriter(log_dir=os.path.join(args.ls,args.en))
    # begin training
    best_mcr = 1e9 # model saving metic
    for epoch in range(args.epochs):
      # train
      train_loss = train(dataloader,model,criterion, optimizer, args.batch_size)
      writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}train_loss'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),train_loss,epoch)
      # eval val
      val_mse,val_mcr = evaluate(dataloader,model, args.batch_size,'val')
      writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}val_mse_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),val_mse,epoch)
      writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}val_mcr_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),val_mcr,epoch)
      # eval test
      if epoch % 5 == 0:
          test_mse, test_mcr = evaluate(dataloader, model,args.batch_size,'test')
          writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}test_mse_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),test_mse,epoch)
          writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}test_mcr_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),test_mcr,epoch)
          # saving best performance model
          if test_mcr < best_mcr:
              best_mcr = test_mcr
              torch.save(model,"model_save/AHGNN.pth")
              print("metric is {}, model save".format(best_mcr))
      # change lr rate
      adjust_learning_rate(optimizer,epoch+1,args)