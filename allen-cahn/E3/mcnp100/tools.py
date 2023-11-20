import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_sol(sol):
    '''
    sol: torch.Tensor, size = (num_grid, )
    return: figs
    '''
    temp = sol.detach().cpu().numpy().reshape(-1, )
    num_grid = temp.shape[0] - 1
    x_axis = (1 / num_grid) * torch.Tensor(range(num_grid + 1))
    plt.figure()
    plt.plot(x_axis.numpy(), temp.numpy())
    plt.show()


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