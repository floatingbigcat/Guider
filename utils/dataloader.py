import torch
import numpy as np
from torch.autograd import Variable


class StandardScaler():
  """
  normalize input data
  """
  def __init__(self):
    self.mean = 0.
    self.std = 1.
  
  def fit(self, data):
    self.mean = data.mean(1)
    self.std = data.std(1)
    self.std[self.std == 0] = 1.0
    print("mean:{},var:{}".format(self.mean,self.std))

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
    self.m, self.n, self.d= self.dat.shape
    #m,n,d: items, time, dim
    self.scale = StandardScaler()
    self._normalized(normalize,train)
    self._split(int(train * self.n), int((train + valid) * self.n), self.n)
    self.device = device
  
  def _normalized(self, normalize,train):
    if (normalize == False):
      self.dat = self.rawdat
    else:
      train_len = int(train * self.n)
      self.scale.fit(self.rawdat[:,0:train_len,:])
      # self.scale.fit(self.rawdat)
      self.dat = self.scale.transform(self.rawdat)
      self.dat = np.nan_to_num (self.dat) 

  def _split(self, train, valid, test):
    train_set = range(self.P + self.H - 1, train)
    valid_set = range(train, valid)
    test_set = range(valid, self.n - self.H)
    self.train = self._batchify(train_set)
    self.valid = self._batchify(valid_set)
    self.test = self._batchify(test_set)

  def _batchify(self, idx_set):
    n = len(idx_set)
    X = torch.zeros((n, self.m, self.P, self.d))
    Y = torch.zeros((n, self.m, self.H, self.d))
    for i in range(n):
      end = idx_set[i] - self.H + 1
      start = end - self.P
      X[i,:,:,:] = torch.from_numpy(self.dat[:,start:end, :])
      Y[i,:,:,:] = torch.from_numpy(self.dat[:,end:end+self.H,:])
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
      X = X.to(self.device)
      Y = Y.to(self.device)
      yield Variable(X),Variable(Y)
      start_idx += batch_size
    