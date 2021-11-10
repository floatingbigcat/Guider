import torch
import numpy as np


def RSE(pred, true):
  return torch.sqrt(torch.sum((true-pred)**2)) / torch.sqrt(torch.sum((true-true.mean())**2))

def CR(pred, true):
  return torch.abs(pred-true)/true

def mCR(pred,true):
  temp = true
  temp[temp==0] = 1
  pred[true==0] = 0
  return torch.mean(torch.abs((pred-true)/temp))

def MAE(pred, true):
  return torch.mean(torch.abs(pred-true))

def MSE(pred, true):
  return torch.mean((pred-true)**2)

def RMSE(pred, true):
  return torch.sqrt(MSE(pred, true))

def MAPE(pred, true):
  return torch.mean(torch.abs((pred - true) / true))

def MSPE(pred, true):
  return torch.mean(torch.square((pred - true) / true))

def CORR(pred, true):
  predict = pred
  Ytest = true
  sigma_p = (predict).std(axis=0)
  sigma_g = (Ytest).std(axis=0)
  mean_p = predict.mean(axis=0)
  mean_g = Ytest.mean(axis=0)
  index = (sigma_g != 0)
  correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
  correlation = torch.where(torch.isnan(correlation), torch.full_like(correlation, 0), correlation)
  correlation = torch.where(torch.isinf(correlation), torch.full_like(correlation, 0), correlation)    
  correlation = (correlation[index]).mean()
  return correlation