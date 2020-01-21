import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


def weighted_nn_loss(weight_ratio,thres):
    def weighted(X,Y):
        base_loss = F.mse_loss(X,Y,reduction = 'mean')
        index = Y > thres
        plus_loss = F.mse_loss(X[index],Y[index], reduction = 'mean') if index.any() > 0 else 0
        total_loss = base_loss + (weight_ratio -1) * plus_loss
        return total_loss
    return weighted
    
    
def CrossEntropy_nn_loss(weightLoss,thres):
    def weighted(X,Y):
        realizations=len(Y)
        Y = (Y > thres).long().flatten()
        X=F.softmax(torch.cat((X[:,0,:,:,:].reshape(realizations*32768,1), X[:,1,:,:,:].reshape(realizations*32768,1)), 1),dim=1)
        loss = F.cross_entropy(X,Y,weight=weightLoss,reduction = 'mean')
        return loss
    return weighted
