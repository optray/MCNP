import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
from config_pde import *


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda')



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


def exact_sol_heat(B_n, t, num_grid, kappa, periodic, alpha):
    '''
    Input:
    B_n: torch.Tensor, size = (b_size * N), fourier basis of f;
    t: torch.Tensor, size = (num_steps, ), recording steps;
    num_grid: int, number of grids;
    kappa: float, positive constant;
    periodic: bool, if periodical or not.
    alpha: float, in (0, 2], fractional order
    Output:
    u: torch.Tensor, size = (b_size, num_steps, num_grid), ground-truth solution of heat equation
    '''
    device = B_n.device
    b_size, N = B_n.shape[0], B_n.shape[1]
    grid = torch.tensor(np.linspace(0, 1-1/num_grid, num_grid)).to(device).reshape(1, 1, 1, -1)
    if periodic == True:
        n = 2 * torch.Tensor(range(1, N + 1)).to(device).reshape(1, -1, 1, 1)
    else:
        n = torch.Tensor(range(1, N + 1)).to(device).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    exp_nt = torch.exp(- kappa * (n * torch.pi)**(alpha) * t.reshape(1, 1, -1, 1))
    return (B_n.reshape(b_size, -1, 1, 1) * sin_grid * exp_nt).sum(axis=1)


def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'dataset'):
        os.mkdir(f'dataset')


setup_seed(0)
check_directory()

for experiment in ['E1', 'E2']:
    cfg = eval('Config_' + experiment + '()')
    B_n = torch.rand(cfg.num_test, cfg.N).to('cuda')
    t = (cfg.total_time/cfg.record_steps) * torch.Tensor(range(cfg.record_steps+1)).to('cuda')
    u = exact_sol_heat(B_n, t, cfg.size, cfg.kappa, cfg.periodic, cfg.alpha)
    torch.save(u, './dataset/' + experiment + '_test_data')


for experiment in ['E1', 'E2']:
    cfg = eval('Config_' + experiment + '()')
    B_n = torch.rand(cfg.num_test, cfg.N).to('cuda')
    t = (cfg.total_time/cfg.record_steps) * torch.Tensor(range(cfg.record_steps+1)).to('cuda')
    u = exact_sol_heat(B_n, t, cfg.size, cfg.kappa, cfg.periodic, cfg.alpha)
    torch.save(u, './dataset/' + experiment + '_val_data')


for experiment in ['E1', 'E2']:
    cfg = eval('Config_' + experiment + '()')
    B_n = torch.rand(cfg.num_train, cfg.N).to('cuda')
    t = (cfg.total_time/cfg.record_steps) * torch.Tensor(range(cfg.record_steps+1)).to('cuda')
    u = exact_sol_heat(B_n, t, cfg.size, cfg.kappa, cfg.periodic, cfg.alpha)
    torch.save(u, './dataset/' + experiment + '_train_data')