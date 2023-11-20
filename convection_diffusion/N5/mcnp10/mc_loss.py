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


def mc_loss(u_0, model, config):
    device = config.device
    size = config.size
    batch_size = config.batch_size

    u_input = u_0.reshape(u_0.shape[0], 1, size).repeat(1, config.time_steps, 1).reshape(-1, size, 1)
    t = config.delta_t * torch.arange(1, config.time_steps+1, device=device).reshape(1, -1).repeat(u_0.shape[0], 1).reshape(-1,)
    u = model(u_input, t).reshape(batch_size, -1, size)
    u = torch.concat([u_0.reshape(batch_size, 1, size), u], dim=1)

    kappa = config.kappa
    b = config.b
    pad = config.pad
    u_0, u_1 = u[:, :-1, :], u[:, 1:, :]
    delta_t = config.T / config.time_steps
    sigma = math.sqrt(2 * kappa * delta_t)
    x = torch.linspace(-pad, 1+pad-1/size, (2*pad+1) * size, device=device).reshape((2*pad+1), -1, 1)
    mu = torch.linspace(0, 1-1/size, size, device=device) + b * delta_t
    mu = (mu - mu.floor()).reshape(1, 1, -1) 
    w = (1/(math.sqrt(2*torch.pi) * sigma) * torch.exp(-((x-mu)**2)/(2 * sigma**2)).sum(axis=0)) /size
    u_hat = torch.einsum('bts, sw->btw', u_0, w)
    return torch.sqrt(torch.mean(torch.square(u_hat - u_1)))