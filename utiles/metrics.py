import torch

def RSE(pred, true):
    return torch.sqrt(torch.sum((true-pred)**2)) / torch.sqrt(torch.sum((true-true.mean())**2))

def CORR(pred, true):
    # u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    # d = torch.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    # return (u/d).mean(-1)
    predict = pred
    Ytest = true
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation
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

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe