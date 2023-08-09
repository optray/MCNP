from model import FNO
from train import train
from tools import setup_seed

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


def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    setup_seed(cfg.seed)
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}_{dateTimeObj.date().day}_{dateTimeObj.time().hour}_{dateTimeObj.time().minute}'
    logfile = f'log/seed_{cfg.seed}_Adam_{cfg.num_iterations}_{cfg.step_size}_{cfg.gamma}_{cfg.lr}_{cfg.weight_decay}_loss_{cfg.num_sample}_ms_{cfg.ms_trick}_FI_{cfg.FI}_{timestring}.csv'

    with open('log/cfg_'+ timestring +'.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')

    sys.stdout.flush()
    
    net = torch.nn.ModuleList([FNO(cfg.NUM_TRUNCATION, cfg.NUM_TRUNCATION, cfg.num_channel) for _ in range(cfg.num_mile)])
    train(cfg, net)
    torch.save(net.state_dict(), f'model/net_seed_{cfg.seed}_Adam_{cfg.num_iterations}_{cfg.step_size}_{cfg.gamma}_{cfg.lr}_{cfg.weight_decay}_loss_{cfg.num_sample}_ms_{cfg.ms_trick}_FI_{cfg.FI}_{timestring}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')

    parser.add_argument('--data_path', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), "../")) + '/data/dataset/',
                        help='path of data')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=0,
            help='seed')

    parser.add_argument('--batch_size', type=int, default=16,
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

    parser.add_argument('--nu', type=float, default=1/10000,
            help='nu in NSE')

    parser.add_argument('--o_size', type=int, default=256,
            help='original spatial size')
    
    parser.add_argument('--size', type=int, default=64,
            help='data spatial size')

    parser.add_argument('--num_sample', type=int, default=16,
            help='number of sampling in vortex loss')

    parser.add_argument('--T', type=float, default=15.0,
            help='final time')

    parser.add_argument('--time_steps', type=int, default=150,
            help='number of time_steps in data')

    parser.add_argument('--EPS', type=float, default=1e-8,
            help='epsilon')

    parser.add_argument('--NUM_TRUNCATION', type=int, default=16,
            help='NUM_TRUNCATION in FNO')

    parser.add_argument('--num_channel', type=int, default=24,
            help='num_channel in FNO')
    
    parser.add_argument('--num_mile', type=int, default=3,
            help='num_channel in FNO')

    parser.add_argument('--FI', type=bool, default=True,
            help='if use FI trick')
    
    parser.add_argument('--ms_trick', type=str, default='b',
                        help='multi sacle rollout trick')

    parser.add_argument('--lamb', type=float, default=0.2,
            help='lamb in loss function')

    cfg = parser.parse_args()
    main(cfg)