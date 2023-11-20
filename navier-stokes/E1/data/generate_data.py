import torch
import numpy as np
import random
import math
import os
import time
from nse import navier_stokes_2d, GaussianRF
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'dataset'):
        os.mkdir(f'dataset')


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



check_directory()
device = torch.device('cuda:1')

s = 256
sub = 4

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(s, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0: -1]

X, Y = torch.meshgrid(t, t, indexing='ij')

# Number of snapshots from solution


setup_seed(0)
record_steps = 200
T = 20
N = 200
bsize = 200
c = 0
u = torch.zeros(N, s//sub, s//sub, record_steps+1)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))


for j in range(N//bsize):
    # Sample random feilds
    w0 = GRF(bsize)
    visc = 10 ** (- 4 * torch.ones(bsize).to(device))
    sol, sol_t = navier_stokes_2d(w0, f, visc, T, 1e-4, record_steps)
    w0 = w0[:, ::sub, ::sub].reshape(-1,  s//sub, s//sub, 1)
    sol = torch.concat([w0, sol[:, ::sub, ::sub, :]], dim=3)
    u[c:(c+bsize),...] = sol
    c += bsize
    print(j, c)
torch.save(u, './dataset/data_test')

setup_seed(1)
record_steps = 200
T = 20
N = 200
bsize = 200
c = 0
u = torch.zeros(N, s//sub, s//sub, record_steps+1)
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))


for j in range(N//bsize):
    # Sample random feilds
    w0 = GRF(bsize)
    visc = 10 ** (- 4 * torch.ones(bsize).to(device))
    sol, sol_t = navier_stokes_2d(w0, f, visc, T, 1e-4, record_steps)
    w0 = w0[:, ::sub, ::sub].reshape(-1,  s//sub, s//sub, 1)
    sol = torch.concat([w0, sol[:, ::sub, ::sub, :]], dim=3)
    u[c:(c+bsize),...] = sol
    c += bsize
    print(j, c)
torch.save(u, './dataset/data_val')
