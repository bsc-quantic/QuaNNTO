import numpy as np

def mse(expected, obtained):
    return ((expected - obtained)**2).sum() / len(obtained)

def nll(expected, obtained):
    return

def retrieve_loss_function(loss_name):
    if loss_name == 'mse':
        return mse
    elif loss_name == 'nll':
        return nll