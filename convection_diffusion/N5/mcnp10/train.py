from mc_loss import mc_loss

import sys
import math
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
    device = config.device
    num = test_data.shape[0]
    delta_steps = 10
    delta_t = config.T / delta_steps
    u_0 = test_data[:, 0, :].to(device)[:, :, None]
    rela_err = torch.zeros(delta_steps)
    rela_err_1 = torch.zeros(delta_steps)
    rela_err_max = torch.zeros(delta_steps)
    for time_step in range(1, 11):
        net.eval()
        t = (time_step*delta_t) * torch.ones(num).to(device)
        u = net(u_0, t).detach()
        u_t = test_data[:, 2 * delta_steps * time_step, :].to(device)[:, :, None]
        rela_err[time_step-1] = (torch.norm((u - u_t).reshape(u.shape[0], -1), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), dim=1)).mean()
        rela_err_1[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=1, dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=1, dim=1)).mean()
        rela_err_max[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=float('inf'), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=float('inf'), dim=1)).mean()
        print(time_step, 'relative l_2 error', rela_err[time_step-1].item(), 'relative l_1 error', rela_err_1[time_step-1].item(), 'relative l_inf error', rela_err_max[time_step-1].item())
    print('mean relative l_2 error', rela_err.mean().item())
    print('mean relative l_1 error', rela_err_1.mean().item())
    print('mean relative l_inf error', rela_err_max.mean().item())
    return rela_err.mean().item()



def train(config, net):
    device = config.device
    size = config.size
    batch_size = config.batch_size
    N = config.N
    grid = torch.tensor(np.linspace(0, 1-1/size, size), dtype=torch.float).to(device).reshape(1, 1, -1, 1)
    n = 2 * torch.Tensor(range(1, N + 1)).to(device).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    val_data = torch.load(config.data_path + config.experiment + '_val_data').to(device)
    test_data = torch.load(config.data_path + config.experiment + '_test_data').to(device)
    val_loss = 1e50
    for step in range(config.num_iterations+1):
        B_n = torch.rand(batch_size, N, 1, 1).to(device)
        u_0 = (B_n * sin_grid).sum(axis=1)
        net.train()
        loss = mc_loss(u_0, net, config)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 50 == 0:
            print('########################')
            print('training loss', step, loss.detach().item())
            print('VAL_LOSS')
            print('########################')
            temp = test(config, net, val_data)
            if temp < val_loss:
                val_loss = temp
                torch.save(net.state_dict(), f'model/net_seed_{config.seed}_{config.timestring}.pt')    
                print('TEST_LOSS')
                print('########################')
                test(config, net, test_data)
            sys.stdout.flush()
    net.load_state_dict(torch.load(f'model/net_seed_{config.seed}_{config.timestring}.pt'))
    print('FINAL_LOSS')
    print('########################')
    test(config, net, test_data)
    sys.stdout.flush()