from model import FNO
from tools import GaussianRF
from mc_loss import mc_loss
import math
 


import sys
import time
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
    delta_steps = 10

    delta_t = config.T / delta_steps
    w_0 = test_data[:, :, :, 0].to(device)[:, :, :, None]
    rela_err = torch.zeros(delta_steps)
    rela_err_1 = torch.zeros(delta_steps)
    rela_err_max = torch.zeros(delta_steps)
    for time_step in range(1,  delta_steps+1):
        net.eval()
        t = (delta_t * time_step) * torch.ones(num).to(device)
        w = net(w_0, t).detach()
        w_t = test_data[..., delta_steps*time_step].to(device)[:, :, :, None]
        rela_err[time_step-1] = (torch.norm((w- w_t).reshape(w.shape[0], -1), dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), dim=1)).mean()
        rela_err_1[time_step-1] = (torch.norm((w- w_t).reshape(w.shape[0], -1), p=1, dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), p=1, dim=1)).mean()
        rela_err_max[time_step-1] = (torch.norm((w- w_t).reshape(w.shape[0], -1), p=float('inf'), dim=1) / torch.norm(w_t.reshape(w.shape[0], -1), p=float('inf'), dim=1)).mean()
        print(time_step, 'relative l_2 error', rela_err[time_step-1].item(), 'relative l_1 error', rela_err_1[time_step-1].item(), 'relative l_inf error', rela_err_max[time_step-1].item())
    print('mean relative l_2 error', rela_err.mean().item())
    print('mean relative l_1 error', rela_err_1.mean().item())
    print('mean relative l_inf error', rela_err_max.mean().item())
    return rela_err.mean().item()


def train_epoch(net, GRF, config, x, f, optimizer, scheduler):
    batch_size, size = config.batch_size, config.size
    net.train()
    w_0 = GRF(batch_size).reshape(batch_size, size, size, 1)
    loss = mc_loss(w_0, x, f, net, config)  
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()



def train(config, net):
    device = config.device
    size = config.size
    batch_size = config.batch_size
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    val_data = torch.load(config.data_path + 'data_val').to(device).float()[..., :int(10*config.T)+1]
    test_data = torch.load(config.data_path+'data_test').to(device).float()[..., :int(10*config.T)+1]
    val_loss = 1e50

    gridx = torch.linspace(0, 1 - 1 / size, size, device=device)
    gridx = gridx.reshape(size, 1, 1).repeat([1, size, 1])
    gridy = torch.linspace(0, 1 - 1 / size, size, device=device)
    gridy = gridy.reshape(1, size, 1).repeat([size, 1, 1])
    x = torch.cat((gridx, gridy), dim=-1)
    f = 0.1*torch.cos(8*math.pi*(x[..., 0]))
    GRF = GaussianRF(size, device=device)
    for step in range(config.num_iterations+1):
        train_epoch(net, GRF, config, x, f, optimizer, scheduler)
        if step % 50 == 0:
            print('########################')
            print('training loss', step)
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