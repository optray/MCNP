import torch
import math
import numpy as np
from tools import w2v_g, get_grid

def vortex_loss(w_0, w_1, delta_t, config):
    """
    param w_0: vortex at initial time; shape: batch-size * size * size
    param w_1: vortex at the next time step
    param delta_t: the time step in mcnp loss
    return: random vortex loss
    """
    device = config.device
    batch_size = w_0.shape[0]
    num_sample = config.num_sample
    size = config.size

    gridx = torch.tensor(np.linspace(0, 1 - 1 / size, size), dtype=torch.float)
    gridx = gridx.reshape(1, size, 1, 1).repeat([1, 1, size, 1])
    gridy = torch.tensor(np.linspace(0, 1 - 1 / size, size), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size, 1).repeat([1, size, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).to(device).reshape(-1, 2)

    if config.FI == True:
        sup = int(config.o_size // size)
    else:
        sup = 1

    sup_size = size * sup
    nu = config.nu
    coeff = torch.sqrt(2 * nu * delta_t)
    w_0, w_1 = w_0.reshape(batch_size, sup_size, sup_size), w_1.reshape(batch_size, size, size)

    grid = get_grid((1, size, size), device)
    f = 0.1 * (torch.sin(2*math.pi*(grid).sum(axis=-1)) + torch.cos(2*math.pi*(grid).sum(axis=-1)))
    v_1 = w2v_g(w_1.detach(), size)
    noise = coeff.reshape(-1, 1, 1, 1) * torch.randn(batch_size, num_sample, size * size, 2, out = torch.cuda.FloatTensor((1))).to(device)
    x = grid.reshape(1, 1, -1, 2) - v_1.reshape(batch_size, 1, size * size, 2) * delta_t.reshape(-1, 1, 1, 1) - noise
    x = x - x.floor()

    index = (x * (sup_size)).round().reshape(-1, 2)
    index[index==sup_size] = 0
    index = index.long()
    index_batch = torch.Tensor(list(range(batch_size))).reshape(-1, 1).repeat(1, num_sample * size * size).reshape(-1,).long().to(device)
    index  = index_batch * sup_size ** 2 + index[:, 0] * sup_size + index[:, 1]
    w_hat = torch.take(w_0, index.long()).reshape(batch_size, num_sample, size, size)
    w_hat = w_hat + f.reshape(1, 1, size, size) * delta_t.reshape(-1, 1, 1, 1)
    return torch.mean(torch.square(w_hat - w_1.reshape(batch_size, 1, size, size).repeat(1, num_sample, 1, 1))) 


