import sys
import math
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def mc_loss(u_0, u_1, delta_t, config):
    '''
    INPUT
    u_0: torch.tensor, size = [batch_size, size, 1]
    u_1: torch.tensor, size = [batch_size, size, 1]
    delta_t: torch.tensor, size = [batch_size,]
    config: setups of experiments
    OUTPUT
    loss: torch.tensor, size = [1,], value of loss function
    '''
    device = config.device
    FI = config.FI
    size = config.size
    sup_size = size * config.sup
    num_sample = config.num_sample
    kappa = 0.02
    delta_t = delta_t.to(device).reshape(-1, 1, 1)

    batch_size = u_0.shape[0]
    x_1 = torch.tensor(np.linspace(0, 1-1/size, size), dtype=torch.float).to(device).reshape(1, 1, -1)

    if FI == False:
        sup_size = size
        u_0_super = u_0
    else:
        u_0_super = config.sup * torch.fft.irfft(torch.fft.rfft(u_0.reshape(batch_size, -1)), n=sup_size)
        u_0_super = torch.concat([u_0_super, torch.zeros(batch_size, 1).to(device)], dim=1)

        noise = torch.sqrt(2 * kappa * delta_t) * torch.randn(batch_size, num_sample, size, out = torch.cuda.FloatTensor((1))).to(device)
        x_1 = x_1 + noise
        x_1 = x_1 - x_1.floor() 
        index = (x_1 * sup_size).round()
        index[index==sup_size] = 0
        index = index.long()
        index_batch = torch.Tensor(list(range(batch_size))).reshape(-1, 1).repeat(1, num_sample * size).reshape(-1, 1).long().to(device)
        index = torch.concat([index_batch, index.reshape(-1, 1)], dim=1).t()
        index = (index[0, :] * (sup_size+1) + index[1, :]).long()
        u_hat = torch.take(u_0_super, index).reshape(batch_size, num_sample, size).mean(axis=1)[:, :, None]
    return torch.mean(torch.square(u_1 - u_hat))