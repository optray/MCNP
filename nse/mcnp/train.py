from model import FNO
from tools import GaussianRF, multiscale
from vortex_loss import vortex_loss
 


import sys
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def test(config, net, test_data):
    num = test_data.shape[0]
    device = config.device
    delta_t = config.T / config.time_steps
    w_0 = test_data[:, :, :, 0].to(device)[:, :, :, None]
    rela_err = torch.zeros(config.time_steps)
    rela_err_1 = torch.zeros(config.time_steps)
    for time_step in range(1, config.time_steps+1):
        net.eval()
        k = (time_step-1) // (config.time_steps // config.num_mile)
        t = (time_step*delta_t) * torch.ones(num).to(device)
        w = net[k](w_0, t - (config.T // config.num_mile)*k).detach()
        w_t = test_data[..., time_step].to(device)[:, :, :, None]
        rela_err[time_step-1] = (torch.norm((w- w_t).reshape(w.shape[0], -1), dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), dim=1)).mean()
        rela_err_1[time_step-1] = (torch.norm((w- w_t).reshape(w.shape[0], -1), p=1, dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), p=1, dim=1)).mean()
        if k < (time_step) // (config.time_steps // config.num_mile):
            w_0 = w
        if time_step%10 == 0:
            print(time_step, 'relative l_2 error', rela_err[time_step-1].item(), 'relative l_1 error', rela_err_1[time_step-1].item())
    print('mean relative l_2 error', rela_err.mean().item(), 'mean relative l_1 error', rela_err_1.mean().item())


def train(config, net):
    device = config.device
    size = config.size
    batch_size = config.batch_size
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    test_data = torch.load(config.data_path+'data_test').to(device).float()[..., :int(10*config.T)+1]

    if config.FI == False:
        sup = 1
    else:
        sup = int(config.o_size // config.size)

    for step in range(config.num_iterations+1):
        net.train()
        GRF = GaussianRF(size * sup, device=device)
        w_0_sup = GRF(batch_size).reshape(batch_size, size*sup, size*sup, 1)
        w_0 = w_0_sup[:, ::sup, ::sup, :]
        loss = torch.tensor([0.0]).to(device)
        for k in range(config.num_mile):
            w_0_pre = net[k](w_0, torch.zeros(w_0.shape[0]).to(device))
            loss += config.lamb * torch.mean(torch.square(w_0 - w_0_pre))

            w_0_repeat = w_0.repeat(40, 1, 1, 1) 
            Delta_t = max(0.1, 0.5 - step * 0.0001)
            t_rand = (config.T/config.num_mile) * torch.rand(w_0_repeat.shape[0]).to(device)
            delta_t = Delta_t * torch.rand(w_0_repeat.shape[0]).to(device) + 0.05
            w_init = net[k](w_0_repeat, t_rand).detach()
            w_end = net[k](w_0_repeat, t_rand + delta_t)
            w_init = torch.fft.irfft(torch.fft.rfft(w_init, dim=1), dim=1, n=size * sup)
            w_init = sup ** 2 * torch.fft.irfft(torch.fft.rfft(w_init, dim=2), dim=2, n=size * sup)
            loss += vortex_loss(w_init, w_end, delta_t, config)     
            w_0 = net[k](w_0, (config.T/config.num_mile) * torch.ones(w_0.shape[0]).to(device)).detach()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 50 == 0:
            print('########################')
            print('training loss', step, loss.detach().item())
            test(config, net, test_data)
            sys.stdout.flush()