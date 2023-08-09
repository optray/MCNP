import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_contour(w):
    '''
    :param w: shape: s * s
    :return: figs
    '''
    temp = w.detach().cpu().numpy()
    s = w.shape[0]
    plt.figure()
    plt.contour(temp.reshape(s, s), 50)
    plt.show()


def get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1 - 1 / size_x, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1 - 1 / size_y, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def kernel_x(x, config):
    device = x.device
    xkt = 2 * torch.pi * torch.mm(x, config.K.t().to(device))
    return torch.sin(xkt), torch.cos(xkt)

def w2v_g(w, size):
    '''
    input: vortex field w, size = [bw, Batch_grid, 1]
    output: velocity field u, size = [bw, Batch_size, 2]
    '''
    device = w.device
    N = size
    bw = w.shape[0]
    #Maximum frequency
    k_max = math.floor(N/2.0)
    w0 = w.reshape(bw, N, N)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    psi_h = w_h / lap

    #Velocity field in x-direction = psi_y
    q = 2. * math.pi * k_y * 1j * psi_h
    q = torch.fft.irfft2(q, s=(N, N))

    #Velocity field in y-direction = -psi_x
    v = -2. * math.pi * k_x * 1j * psi_h
    v = torch.fft.irfft2(v, s=(N, N))
    return torch.concat([q[:,:,:,None], v[:,:,:,None]], dim=-1).reshape(bw, -1, 2)


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


class GaussianRF(object):

    def __init__(self, size, alpha=2.5, tau=7, sigma=None, boundary="periodic", device=None):
        self.dim = 2
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size//2
        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

        k_x = wavenumers.transpose(0,1)
        k_y = wavenumers

        self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def __call__(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


def multiscale(config, Delta_t):
    device = config.device
    if config.ms_trick == 'a':
        t_rand = config.T * torch.rand(config.batch_size).to(device)
        delta_t = Delta_t * torch.rand(config.batch_size).to(device) + 0.05
    if config.ms_trick == 'b':
        t_rand = config.T * torch.rand(config.batch_size).to(device)
        delta_t = 0.3 * torch.rand(config.batch_size).to(device) + 0.2
    if config.ms_trick == 'c':
        t_rand = config.T * torch.rand(config.batch_size).to(device)
        delta_t = 0.0 * torch.rand(config.batch_size).to(device) + 0.2
    return t_rand, delta_t