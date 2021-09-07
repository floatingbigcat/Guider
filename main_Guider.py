import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter


import argparse
from utils.dataloader import DataLoader
from utils.metrics import *
from utils.tools import adjust_learning_rate
from model.model import Guider
from math import sqrt
#from exp import train



parser = argparse.ArgumentParser(description='Guider Time series forecasting')
parser.add_argument('--data', type=str, default='data/LZ_onehot.npy',
                    help='location of the data file')
parser.add_argument('--hier_data', type=str, default='data/LZ_H_type.npy',
                    help='location of the data file')
# parser.add_argument('--save', type=str, default='model_save/model_Guider.pt',
#                     help='location of the model save')
parser.add_argument('--pre_hier', type=bool, default=True,#r'/export/Martin/Guider/data/HSD_H_dept.npy',
                    help='True->selflearned,hier False->none hier')
parser.add_argument('--hier',type=bool,default=True,help='with hier or not')
parser.add_argument('--next_nodes_num',type=int,default=7,help='number of nodes after pooled')
parser.add_argument('--num_nodes',type=int,default=2247,help='number of nodes/variables')
parser.add_argument('--d_model',type=int,default=53,help='input dimension')
parser.add_argument('--seq_in',type=int,default=3,help='input sequence length')
parser.add_argument('--seq_out',type=int,default=1,help='output sequence length')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--lr',type=float,default=5e-5,help='learning rate')
parser.add_argument('--lradj',type=str,default='type3',help='learning rate')
parser.add_argument('--n_heads',type=int,default=4,help='mult_head attentions')
parser.add_argument('--e_layer', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layer', type=int, default=1, help='num of decoder layers')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--epochs',type=int,default=1000,help='')
parser.add_argument('--delta', type=float, default=1e-6,help='rate of (R-PQ)**2 in loss')
parser.add_argument('--alpha', type=float, default=0.5,help='rate of et')
args = parser.parse_args()
#args = parser.parse_known_args()[0]

device = torch.device(args.device)

# result_path = 'compare/compare_e_layers/'+str(args.e_layer)
result_path='result_LZ/'
def train(Data,model, criterion, optimizer, batch_size):
    model.train()

    train_loss = []
    for enc_X, Y in Data.get_batches(Data.train[0],Data.train[1],batch_size):
        model.zero_grad()
        Yhat, Y = model(enc_X,Y)
        Yhat.retain_grad
 
        loss_p1 = criterion(Yhat,Y)
        # loss_p2 = args.delta*((enc_X - torch.einsum(model.GraphLearn.N_matrix,model.GraphLearn.T_matrix.T))**2).sum()
        loss = loss_p1# + loss_p2
        loss.backward()

        optimizer.step()
        train_loss.append(sqrt(loss))


    return sum(train_loss)/len(train_loss)

def evaluate(Data,model, batch_size, flag):
    model.eval()

    mse = []
    mae = []
    corr = []
    if flag == 'val':
        for enc_X, Y in Data.get_batches(Data.valid[0],Data.valid[1],batch_size):
            with torch.no_grad():
                Yhat, Y = model(enc_X,Y)          
            mae.append(MAE(Yhat,Y))
            mse.append(MSE(Yhat,Y))
            corr.append(CORR(Yhat,Y))
    elif flag == 'test':
        for enc_X, Y in Data.get_batches(Data.test[0],Data.test[1],batch_size):
            with torch.no_grad():
                Yhat, Y = model(enc_X,Y)
            mae.append(MAE(Yhat,Y))
            mse.append(MSE(Yhat,Y))
            corr.append(CORR(Yhat,Y))
    evl_mse = sum(mse)/len(mse)
    evl_mae = sum(mae)/len(mae)
    evl_corr = sum(corr)/len(corr)

    return  evl_mse,evl_mae,evl_corr 

if __name__ == "__main__":
    Data = DataLoader(args.data, 0.6, 0.2, device = device, seq_out=args.seq_out, seq_in= args.seq_in,normalize = args.normalize)
    
    model = Guider(data = Data,d_model=args.d_model,c_out = 1,seq_in = args.seq_in,
                device = device,pre_hier = args.pre_hier,
                n_heads=args.n_heads,num_nodes=args.num_nodes,
                e_layers=args.e_layer,d_layers=args.d_layer,args = args)

    model = model.to(device) 
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    criterion = nn.MSELoss().to(device)
    evaluateL2 = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss = []
    best_val = 1e9
    writer = SummaryWriter(log_dir=result_path)

    for epoch in range(args.epochs):
        train_loss = train(Data,model,criterion, optimizer, args.batch_size)
        #print("******rmse = {:.3f}*********",train_loss)
        val_mse,val_mae,val_corr = evaluate(Data,model, args.batch_size,'val')
        #print("******rmse = {:.3f}*********",val_loss)

        print("第{:d}个EPOCH*****train_loss:{:5.4f}******val_mse:{:5.4f}******val_mae:{:5.4f}******val_corr:{:5.4f}".format(int(epoch),train_loss,float(val_mse),float(val_mae),float(val_corr)))
        # writer.add_scalar('Alpha:{:f}Horizon:{:d}train_loss_Guiderdecoderchanged'.format(args.alpha,args.seq_out),train_loss,epoch)
        writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}val_mse_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),val_mse,epoch)
        writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}val_mae_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),val_mae,epoch)
        
        if epoch % 5 == 0:
            test_mse, test_mae, test_corr = evaluate(Data, model,args.batch_size,'test')
            print("test mse {:5.4f} | test mae {:5.4f} | test corr {:5.4f}".format(test_mse, test_mae, test_corr), flush=True)
            writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}test_mse_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),test_mse,epoch)
            writer.add_scalar('Alpha:{:f}Horizon:{:d}Hier:{:d}pre_define:{:d}test_mae_Guider'.format(args.alpha,args.seq_out,args.hier,args.pre_hier),test_mae,epoch)
        
        adjust_learning_rate(optimizer,epoch+1,args)

            # writer.add_scalar('Horizon:{:d}test_mae_Guider'.format(args.seq_out),test_mae,epoch)
            # writer.add_scalar('Horizon:{:d}test_corr_Guider'.format(args.seq_out),test_corr,epoch) 
