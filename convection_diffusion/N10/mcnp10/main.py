
from model import FNO
from train import train
from config_pde import *

import json
import sys
import copy
from datetime import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True
     torch.backends.cudnn.enabled = True



def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    setup_seed(cfg.seed)
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}_{dateTimeObj.date().day}_{dateTimeObj.time().hour}_{dateTimeObj.time().minute}_{dateTimeObj.time().second}'
    cfg.timestring = timestring
    logfile = f'log/20_{cfg.experiment}_seed_{cfg.seed}_Adam_{cfg.num_iterations}_{cfg.step_size}_{cfg.gamma}_{cfg.lr}_{timestring}.csv'

    with open('log/cfg_'+ timestring +'.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')

    sys.stdout.flush()
    
    net = FNO(cfg.modes, 32) 
    
    cfg.delta_t = cfg.T/cfg.time_steps
    train(cfg, net)
    torch.save(net.state_dict(), f'model/net_seed_{cfg.seed}_{timestring}.pt')
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')

    parser.add_argument('--data_path', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), "../")) + '/data/dataset/',
                        help='path of data')

    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=0,
            help='seed')

    parser.add_argument('--batch_size', type=int, default=200,
            help='batchsize of the operator learning')
    
    parser.add_argument('--step_size', type=int, default=1000,
            help='step_size of optim')
        
    parser.add_argument('--gamma', type=float, default=0.5,
            help='gamma of optim')
    
    parser.add_argument('--lr', type=float, default=0.01,
            help='lr of optim')

    parser.add_argument('--weight_decay', type=float, default=0.0,
            help='lr of optim')

    parser.add_argument('--num_iterations', type=int, default=10000,
            help='num_iterations of optim')

    parser.add_argument('--size', type=int, default=64,
            help='data spatial size')

    parser.add_argument('--T', type=float, default=2.0,
            help='final time')

    parser.add_argument('--time_steps', type=int, default=10,
            help='number of time_steps in data')

    parser.add_argument('--EPS', type=float, default=1e-8,
            help='epsilon')

    parser.add_argument('--experiment', type=str, default='E2',
            help='index of the experiments')

    parser.add_argument('--pad', type=int, default=1,
            help='num_copy in fk loss')
        
    parser.add_argument('--modes', type=int, default=20,
            help='modes in fno')

    for seed in [0, 1, 2]:
        for experiment in ['E2', 'E1']:
            cfg = parser.parse_args()
            cfg.seed = seed
            cfg.experiment = experiment   
            cfg_pde = eval('Config_' + cfg.experiment + '()')
            cfg.N, cfg.kappa, cfg.b = cfg_pde.N, cfg_pde.kappa, cfg_pde.b
            main(cfg)
