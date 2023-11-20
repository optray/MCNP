import torch
import math
import numpy as np
from tools import w2v_g



def mc_loss(w_0, x, f_1, model, config):
    device = config.device
    size = config.size
    sup_w, sup_u = config.sup_w, config.sup_u
    sup_size_w, sup_size_u = sup_w * size, sup_u * size
    batch_size = config.batch_size
    time_steps = config.time_steps
    delta_t = config.delta_t
    k = config.k
    nu = config.nu
    sigma = math.sqrt(2 * nu * delta_t)
    w_0 = w_0.reshape([1, batch_size, size, size, 1])

    w_input = w_0.repeat(time_steps, 1, 1, 1, 1).reshape(-1, size, size, 1)
    t = delta_t * torch.arange(1, time_steps+1, device=device).reshape(-1, 1).repeat(1,batch_size).reshape(-1,)
    w = model(w_input, t).reshape(time_steps, batch_size, size, size, 1)
    w = torch.concat([w_0.reshape(1, -1, size, size, 1), w], dim=0)

    w_0, w_1 = w[:-1, ...], w[1:, ...]
    v = w2v_g(w.reshape(-1, size, size, 1), size).reshape(time_steps+1, batch_size, size, size, 2)
    v_0, v_1 = v[:-1, ...], v[1:, ...]

    if sup_u > 1:
        v_0 = torch.fft.irfft(torch.fft.rfft(v_0, dim=2), dim=2, n=sup_size_u)
        v_0 = sup_u ** 2 * torch.fft.irfft(torch.fft.rfft(v_0, dim=3), dim=3, n=sup_size_u)
    mu_1 = x - v_1 * delta_t
    index = mu_1 - mu_1.floor()
    index = (index * (sup_size_u)).round().reshape(-1, 2)
    index[index==sup_size_u] = 0
    index = index.long()
    index_batch = torch.arange(batch_size * time_steps, device=device).reshape(-1, 1).repeat(1, size**2).reshape(-1,).long()
    index = index_batch * sup_size_u ** 2 + index[:, 0] * sup_size_u + index[:, 1]

    v_2 = torch.concat([torch.take(v_0[..., 0], index.long()).reshape(time_steps, batch_size, size, size, 1), 
                         torch.take(v_0[..., 1], index.long()).reshape(time_steps, batch_size, size, size, 1)], dim=-1)
    
    mu = x - 0.5*(v_1+v_2) * delta_t
    f_2 = 0.1* (torch.sin(2*math.pi*(mu).sum(axis=-1)) + torch.cos(2*math.pi*(mu).sum(axis=-1)))

    if sup_w > 1:
        w_0 = torch.fft.irfft(torch.fft.rfft(w_0, dim=2), dim=2, n=sup_size_w)
        w_0 = sup_w ** 2 * torch.fft.irfft(torch.fft.rfft(w_0, dim=3), dim=3, n=sup_size_w)

    delta_gridx = torch.linspace(-k/sup_size_w, k/sup_size_w, 2*k+1, device=device)
    delta_gridx = delta_gridx.reshape(-1, 1, 1).repeat([1, 2*k+1, 1])
    delta_gridy = torch.linspace(-k/sup_size_w, k/sup_size_w, 2*k+1, device=device)
    delta_gridy = delta_gridy.reshape(1, -1, 1).repeat([2*k+1, 1, 1])
    delta_grid = torch.cat((delta_gridx, delta_gridy), dim=-1)
    delta_grid = delta_grid[delta_grid.norm(dim=-1) <= (k)/sup_size_w]
    delta_size = delta_grid.shape[0]

    loc_p = mu.reshape(time_steps, batch_size, size, size, 1, 2) + delta_grid.reshape(1, 1, 1, 1, -1, 2)
    loc_p = (loc_p * sup_size_w).round()/sup_size_w
    loc_density = 1/(2*torch.pi*sigma**2) * torch.exp(-((loc_p-mu.reshape(time_steps, batch_size, size, size, 1, 2))**2).sum(axis=-1)/(2 * sigma**2)) * (1/sup_size_w**2)
    loc_density = loc_density/loc_density.detach().sum(dim=-1, keepdim=True)
    w_0 = w_0.reshape(-1, sup_size_w, sup_size_w)
    loc_p = loc_p - loc_p.floor()
    loc_p = (loc_p * (sup_size_w)).round().reshape(-1, 2)
    loc_p[loc_p==sup_size_w] = 0
    loc_p = loc_p.long()
    index_batch = torch.arange(batch_size * time_steps, device = device).reshape(-1, 1).repeat(1, delta_size * size**2).reshape(-1,).long()
    loc_p = index_batch * sup_size_w ** 2 + loc_p[:, 0] * sup_size_w + loc_p[:, 1]
    w_hat = torch.take(w_0, loc_p.long()).reshape(time_steps, batch_size, size, size, delta_size, 1)
    f = 0.5 * (f_1.reshape(1, 1, size, size, 1) + f_2[..., None])
    w_hat = (w_hat * loc_density[..., None]).sum(axis=-2) + f * delta_t

    return torch.sqrt(torch.mean(torch.square(w_hat - w_1)))