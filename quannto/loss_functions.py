import numpy as np
from .data_processors import softmax_discretization

def mse(expected, obtained):
    return ((expected - obtained)**2).sum() / len(obtained)

def nll(expected, obtained):
    return

def cross_entropy(true_labels, obtained_outputs):
    obtained_probs = softmax_discretization(obtained_outputs)
    return (-np.sum(true_labels * np.log(obtained_probs))) / len(obtained_outputs)

def retrieve_loss_function(loss_name):
    if loss_name == 'mse':
        return mse
    elif loss_name == 'nll':
        return nll
    elif loss_name == 'cross_entropy':
        return cross_entropy