from mc_loss import mc_loss
from initial_field import GaussianRF
import sys
import math
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
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
        u_t = test_data[:, delta_steps * time_step, :].to(device)[:, :, None]
        rela_err[time_step-1] = (torch.norm((u - u_t).reshape(u.shape[0], -1), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), dim=1)).mean()
        rela_err_1[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=1, dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=1, dim=1)).mean()
        rela_err_max[time_step-1] = (torch.norm((u- u_t).reshape(u.shape[0], -1), p=float('inf'), dim=1) / torch.norm(u_t.reshape(u.shape[0], -1), p=float('inf'), dim=1)).mean()
        print(time_step, 'relative l_2 error', rela_err[time_step-1].item(), 'relative l_1 error', rela_err_1[time_step-1].item(), 'relative l_inf error', rela_err_max[time_step-1].item())
    print('mean relative l_2 error', rela_err.mean().item())
    print('mean relative l_1 error', rela_err_1.mean().item())
    print('mean relative l_inf error', rela_err_max.mean().item())
    return rela_err.mean().item()


def p_matrix(config):
    size_x = config.size
    dt = config.delta_t
    p = np.zeros([size_x+1, size_x+1])
    x = np.linspace(0, 1, size_x+1)
    dx = 1/size_x
    kappa = 0.01
    sigma = math.sqrt(2 * kappa * dt)
    for i in range(size_x+1):
        if x[i] <= 2/3:
            d = x[i]
            p[i, :] = (1/(math.sqrt(2*torch.pi) * sigma) * np.exp(-((x-d)**2)/(2 * sigma**2))) * dx + (1/(math.sqrt(2*torch.pi) * sigma) * np.exp(-((-x-d)**2)/(2 * sigma**2))) * dx       
            p[i, 0] = (1/(math.sqrt(2*torch.pi) * sigma) * np.exp(-((-d)**2)/(2 * sigma**2))) * dx       
        if x[i] > 2/3:
            p[i, :] = np.flip(p[size_x-i, :])
    return torch.tensor(p).float()


def generate_u0(b_size, N, num_grid, device):
    '''
    Input:
    b_size: int, num of pre-generated data
    N: int, highest frequency fourier basis;
    num_grid: int, number of grids;
    Output:
    u_0: torch.Tensor, size = (b_size, num_grid, 1), initialization
    '''
    B_n = torch.rand(b_size, N, device=device).reshape(b_size, N, 1, 1)
    grid = torch.linspace(0.5/num_grid, 1 - 0.5/num_grid, num_grid, device=device).reshape(1, 1, -1, 1)
    n = 2 * torch.range(1, N, device=device).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    return (B_n * sin_grid).sum(axis=1)


def train(config, net):
    N = 5
    device = config.device
    size = config.size
    batch_size = config.batch_size
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    val_data = torch.load(config.data_path + 'data_val').to(device).float()
    test_data = torch.load(config.data_path + 'data_test').to(device).float()
    val_loss = 1e50
    p = p_matrix(config).to(device)
    for step in range(config.num_iterations+1):
        u = generate_u0(batch_size, N, 1024, device)
        u_0 = (u[:, 15:-1:16] + u[:, 16::16])/2
        u_0 = torch.concat([u[:, 0, None], u_0, u[:, -1, None]], dim=1)
        net.train()
        loss = mc_loss(u_0, net, p, config)
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