import math
import torch
import numpy as np

class GaussianRF(object):

    def __init__(self, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.device = device

        sigma = tau**(0.5*(2*alpha - 1))

        k_max = size//2

        k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

        self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0] = 0.0

        self.size = []
        self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[...,0] = self.sqrt_eig*coeff[...,0]
        coeff[...,1] = self.sqrt_eig*coeff[...,1]

        u = torch.fft.ifft(torch.view_as_complex(coeff)).real

        return u