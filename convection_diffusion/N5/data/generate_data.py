import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
from config_pde import *


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = True


def generate_u0(b_size, N, num_grid, device='cuda'):
    '''
    Input:
    b_size: int, num of pre-generated data
    N: int, highest frequency fourier basis;
    num_grid: int, number of grids;
    Output:
    u_0: torch.Tensor, size = (b_size, num_grid, 1), initialization
    '''
    B_n = torch.rand(b_size, N).to(device).reshape(b_size, N, 1, 1)
    grid = torch.tensor(np.linspace(0, 1-1/num_grid, num_grid)).to(device).reshape(1, 1, -1, 1)
    n = 2 * torch.Tensor(range(1, N + 1)).to(device).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    return (B_n * sin_grid).sum(axis=1)


def psm_cde(u_0, kappa, b, T, record_steps, dt = 1e-5):
    '''
    Input:
    u_0: torch.Tensor, size = (b_size, num_grid, 1), initialization;
    kappa: float, positive constant;
    b: float, convection term;
    T: final time
    Output:
    u: torch.Tensor, size = (b_size, num_grid, num_steps+1)
    '''
    total_steps = int(T/dt)
    b_size, num_grid = u_0.shape[0], u_0.shape[1]
    u_psm = u_0
    u = u_0
    sub_t = int(total_steps // record_steps)
    for time_step in range(0, total_steps):
        u_h = torch.fft.fft(u, dim=1)
        k_max = num_grid//2
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,num_grid,1)
        ux_h = 2j *np.pi*k_x*u_h
        ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=1, n=num_grid)
        uxx_h = 2j *np.pi*k_x*ux_h
        uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=1, n=num_grid)
        u_1 = u + b * ux * dt + kappa * dt * uxx

        u_h = torch.fft.fft(u_1, dim=1)
        k_max = num_grid//2
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,num_grid,1)
        ux_h = 2j *np.pi*k_x*u_h
        ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=1, n=num_grid)
        uxx_h = 2j *np.pi*k_x*ux_h
        uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=1, n=num_grid)
        u_2 = u + b * ux * dt + kappa * dt * uxx
        u = (u_1+u_2)/2
        if (time_step+1) % sub_t == 0:
            u_psm = torch.concat([u_psm, u], dim=-1)
    return u_psm



def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'dataset'):
        os.mkdir(f'dataset')



check_directory()



for experiment in ['E1', 'E2']:
    setup_seed(0)
    cfg = eval('Config_' + experiment + '()')
    device = cfg.device
    kappa = cfg.kappa
    b = cfg.b
    b_size = cfg.num_train
    N = cfg.N
    num_grid = cfg.sup_size
    sub_x = cfg.sup_size // cfg.size
    total_time = cfg.total_time
    record_steps = cfg.record_steps
    
    u_0 = generate_u0(b_size, N, num_grid, device)
    u = psm_cde(u_0, kappa, b, total_time, record_steps)
    u = u[:, ::sub_x, :]
    print(u.shape, u.max())
    torch.save(u.float().permute(0, 2, 1), './dataset/' + experiment + '_train_data')


for experiment in ['E1', 'E2']:
    setup_seed(1)
    cfg = eval('Config_' + experiment + '()')
    device = cfg.device
    kappa = cfg.kappa
    b = cfg.b
    b_size = cfg.num_test
    N = cfg.N
    num_grid = cfg.sup_size
    sub_x = cfg.sup_size // cfg.size
    total_time = cfg.total_time
    record_steps = cfg.record_steps
    
    u_0 = generate_u0(b_size, N, num_grid, device)
    u = psm_cde(u_0, kappa, b, total_time, record_steps)
    u = u[:, ::sub_x, :]
    print(u.shape, u.max())
    torch.save(u.float().permute(0, 2, 1), './dataset/' + experiment + '_test_data')


for experiment in ['E1', 'E2']:
    setup_seed(2)
    cfg = eval('Config_' + experiment + '()')
    device = cfg.device
    kappa = cfg.kappa
    b = cfg.b
    b_size = cfg.num_val
    N = cfg.N
    num_grid = cfg.sup_size
    sub_x = cfg.sup_size // cfg.size
    total_time = cfg.total_time
    
    u_0 = generate_u0(b_size, N, num_grid, device)
    u = psm_cde(u_0, kappa, b, total_time, record_steps)
    u = u[:, ::sub_x, :]
    print(u.shape, u.max())
    torch.save(u.float().permute(0, 2, 1), './dataset/' + experiment + '_val_data')

