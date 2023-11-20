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


def mc_loss(u_0, model, p, config):
    device = config.device
    size = config.size
    batch_size = config.batch_size
    time_steps = config.time_steps
    delta_t = config.delta_t

    u_input = u_0.reshape(u_0.shape[0], 1, size+1).repeat(1, time_steps, 1).reshape(-1, size+1, 1)
    t = delta_t * torch.tensor(range(1, time_steps+1)).to(device).reshape(1, -1).repeat(u_0.shape[0], 1).reshape(-1,)
    u = model(u_input, t).reshape(batch_size, -1, size+1)
    u = torch.concat([u_0.reshape(batch_size, 1, size+1), u], dim=1)

    u_0, u_1 = u[:, :-1, :], u[:, 1:, :]
    f_temp = 0.5*(u_0+u_1)
    f_temp = f_temp - f_temp**3
    u_hat = torch.einsum('bts, gs->btg', u_0 + f_temp* delta_t, p) 
    return torch.sqrt(torch.mean(torch.square(u_hat - u_1)))