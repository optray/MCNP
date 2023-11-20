import math
import torch
import os
import random
import numpy as np
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
from initial_field import GaussianRF


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


setup_seed(0)
size_x = 1024
size_t = 200
T = 1.0
delta_t = 1e-6
num_test = 200
num_val = 200

def generate_u0(b_size, N, num_grid):
    '''
    Input:
    b_size: int, num of pre-generated data
    N: int, highest frequency fourier basis;
    num_grid: int, number of grids;
    Output:
    u_0: torch.Tensor, size = (b_size, num_grid, 1), initialization
    '''
    B_n = torch.rand(b_size, N).reshape(b_size, N, 1, 1)
    grid = torch.tensor(np.linspace(0.5/num_grid, 1 - 0.5/num_grid, num_grid)).reshape(1, 1, -1, 1)
    n = 2 * torch.Tensor(range(1, N + 1)).reshape(1, -1, 1, 1)
    sin_grid = torch.sin(torch.pi * n * grid)
    return (B_n * sin_grid).sum(axis=1)


setup_seed(1)
N = 10
size_x = 65
size_t = 100
T = 1.0
data_test = np.zeros([num_test, size_t+1, size_x])
initial = GaussianRF(1024)
grid = CartesianGrid([[0, 1]], 1024) # generate grid
field = ScalarField(grid, 2)
for i in range(num_test):
    maxvalue = np.nan
    while np.isnan(maxvalue) or maxvalue>10:
        bc_x_left = {"derivative": 0}
        bc_x_right =  {"derivative": 0}
        term_1 = f"laplace(c) * 0.01 + c - c**3"
        eq = PDE({"c": f"{term_1}"}, bc=[bc_x_left, bc_x_right])
        b = generate_u0(1, N, 1024)
        field.data = np.array(b).reshape(-1,)
        storage = MemoryStorage() # store intermediate information of the simulation
        res = eq.solve(field, T, dt=delta_t, tracker=storage.tracker(T/size_t)) # solve the PDE
        a = torch.tensor(storage.data)
        a_in = (a[:, 15:-1:16] + a[:, 16::16])/2
        data_test[i, :, 1:-1] = a_in
        data_test[i, :, 0], data_test[i, :, -1] = a[:, 0], a[:, -1]
        maxvalue = np.abs(data_test[i]).max()
    print(i, np.abs(data_test[i]).max())

data_test = torch.tensor(data_test)
torch.save(data_test, 'dataset/data_test')


setup_seed(2)
N = 10
size_x = 65
size_t = 100
T = 1.0
data_val = np.zeros([num_val, size_t+1, size_x])
grid = CartesianGrid([[0, 1]], 1024) # generate grid
field = ScalarField(grid, 2)
for i in range(num_val):
    maxvalue = np.nan
    while np.isnan(maxvalue) or maxvalue>10:
        bc_x_left = {"derivative": 0}
        bc_x_right =  {"derivative": 0}
        term_1 = f"laplace(c) * 0.01 + c - c**3"
        eq = PDE({"c": f"{term_1}"}, bc=[bc_x_left, bc_x_right])
        b = generate_u0(1, N, 1024)
        field.data = np.array(b).reshape(-1,)
        storage = MemoryStorage() # store intermediate information of the simulation
        res = eq.solve(field, T, dt=delta_t, tracker=storage.tracker(T/size_t)) # solve the PDE
        a = torch.tensor(storage.data)
        a_in = (a[:, 15:-1:16] + a[:, 16::16])/2
        data_val[i, :, 1:-1] = a_in
        data_val[i, :, 0], data_val[i, :, -1] = a[:, 0], a[:, -1]
        maxvalue = np.abs(data_val[i]).max()
    print(i, np.abs(data_val[i]).max())

data_val = torch.tensor(data_val)
torch.save(data_val, 'dataset/data_val')