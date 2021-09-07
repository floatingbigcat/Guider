import numpy as np
import torch
from torch.autograd import Variable
#from sklearn.preprocessing import StandardScaler
import math
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(1)
        self.std = data.std(1)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        temp = data.transpose(1,0,2)
        temp = np.nan_to_num(((temp - mean) / std).transpose(1,0,2))
        return temp

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
class DataLoader(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, filename, train, valid, device, seq_in, seq_out,normalize=True):
        self.P = seq_in
        self.H = seq_out
        self.rawdat = np.load(filename)
        self.dat = np.zeros(self.rawdat.shape)
        #print(self.dat)
        self.m, self.n, self.d= self.dat.shape

        #m is node_nums, n is t

        self.scale = StandardScaler()
        self._normalized(normalize,train)
        self.R = torch.tensor(self.dat[:,:,0]).to(device)
        #self.N_matrix,self.T_matrix = Matrix_factorization(self.R,self.d,device = device)
        self.device = device
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
    
    def normal_std(x):
        return x.std() * np.sqrt((len(x) - 1.)/(len(x)))
    def _normalized(self, normalize,train):
        # normalized by the maximum value of entire matrix.

        if (normalize == False):
            self.dat = self.rawdat

        else:
            train_len = int(train * self.n)
            self.scale.fit(self.rawdat[:,0:train_len,:])
            # self.scale.fit(self.rawdat)
            self.dat = self.scale.transform(self.rawdat)
            self.dat = np.nan_to_num (self.dat) 
        # normlized by the maximum value of each row(sensor).

    def _split(self, train, valid, test):

        train_set = range(self.P + self.H - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n - self.H)
        #print(train_set)
    
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.m, self.P, self.d))

        Y = torch.zeros((n, self.m, self.H, self.d))
        #Y = torch.zeros((n, self.m, 1))

        for i in range(n):
            end = idx_set[i] - self.H + 1
            start = end - self.P
            X[i,:,:,:] = torch.from_numpy(self.dat[:,start:end, :])
           
            Y[i,:,:,:] = torch.from_numpy(self.dat[:,end:end+self.H,:])

            #Y[i,:,:] = torch.from_numpy(self.dat[:,idx_set[i]][:,0].reshape(self.m,1))

        return [X,Y]

    def get_batches(self, enc_in ,targets, batch_size, shuffle=False):
        length = len(enc_in)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]

            X = enc_in[excerpt]

            Y = targets[excerpt]
            # N_m = self.N_matrix
            # T_m = self.T_matrix[end_idx]
            # R = self.R[end_idx]

            X = X.to(self.device)

            Y = Y.to(self.device)
            # N_m = N_m.to(self.device)
            # T_m = T_m.to(self.device)


            yield Variable(X),Variable(Y)
            start_idx += batch_size

if __name__ == "__main__":
    
    data = r'data/CA_1_mini.npy'
    dat = np.load(data)
    m = torch.tensor(dat[:,:,0]).to("cuda:0")
    #p,q = Matrix_factorization(m,5,"cuda:0")

    ''' data = 'D:\Data\Datasets\m5\CA_1_mini.npy'
    Data = DataLoader(data, 0.6, 0.2, device = 'cpu', seq_out = 4, seq_in = 12,label_len=7,normalize = 0)
    #train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
    for enc_X,dec_X,Y,N_m,T_m in Data.get_batches(Data.train[0],Data.train[1],Data.train[2],batch_size=8):
        print(enc_X.shape,dec_X.shape, Y.shape,N_m.shape,T_m.shape)
    '''
    