from mc_loss import mc_loss

import sys
import math
import copy
import random
import numpy as np
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def test(config, net, test_data):
    device = config.device
    num = test_data.shape[0]
    delta_t = config.T / config.time_steps
    u_0 = test_data[:, 0, :].to(device)[:, :, None]
    rela_err = torch.zeros(config.time_steps)
    rela_err_1 = torch.zeros(config.time_steps)
    rela_err_max = torch.zeros(config.time_steps)
    for time_step in range(1, config.time_steps+1):
        net.eval()
        t = (time_step*delta_t) * torch.ones(num).to(device)
        u = net(u_0, t).detach()
        u_t = test_data[:, time_step, :].to(device)[:, :, None]
        rela_err[time_step-1] = (torch.norm((u - u_t).reshape(u.shape[0], -1), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), dim=1)).mean()
        rela_err_1[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=1, dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=1, dim=1)).mean()
        rela_err_max[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=float('inf'), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=float('inf'), dim=1)).mean()
        if time_step % 10 == 0:
            print(time_step, 'relative l_2 error', rela_err[time_step-1].item(), 'relative l_1 error', rela_err_1[time_step-1].item(), 'relative l_inf error', rela_err_max[time_step-1].item())
    print('mean relative l_2 error', rela_err.mean().item())
    print('mean relative l_1 error', rela_err_1.mean().item())
    print('mean relative l_inf error', rela_err_max.mean().item())


def train(config, net):
    device = config.device
    size = config.size
    batch_size = config.batch_size
    if config.experiment == 'E1':
        N = 6
    if config.experiment == 'E2':
        N = 12
    grid = torch.tensor(np.linspace(0, 1-1/size, size), dtype=torch.float).to(device).reshape(1, 1, -1, 1)
    n = 2 * torch.Tensor(range(1, N + 1)).to(device).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), config.lr, weight_decay = config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    test_data = torch.load(config.data_path + config.experiment + '_test_data').to(device).float()
    for step in range(config.num_iterations+1):
        net.train()
        B_n = torch.rand(batch_size, N, 1, 1).to(device)
        u_0 = (B_n * sin_grid).sum(axis=1)
        delta_t = 0.2 * torch.rand(u_0.shape[0]).to(device) 
        u_0_delta = net(u_0, delta_t)
        loss = config.lamb * mc_loss(u_0, u_0_delta, delta_t, config)   

        u_0_repeat = u_0.repeat(20, 1, 1) 
        t_rand = config.T * torch.rand(u_0_repeat.shape[0]).to(device)
        delta_t = 0.2 * torch.rand(u_0_repeat.shape[0]).to(device) + 0.2
        u_init = net(u_0_repeat, t_rand).detach()
        u_end = net(u_0_repeat, t_rand + delta_t)
        loss += mc_loss(u_init, u_end, delta_t, config)     

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 100 == 0:
            print('########################')
            print('training loss', step, loss.detach().item())
            test(config, net, test_data)
            sys.stdout.flush()