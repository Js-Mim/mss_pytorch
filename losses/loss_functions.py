# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import numpy as np

# definitions
l_f = np.log(10.)


def kullback_leibler(x, x_hat):
    # Generalized KL
    rec = torch.sum(x * (torch.log(x + 1e-6) - torch.log(x_hat + 1e-6)) + (x_hat - x), dim=-1)
    return torch.mean(rec)


def nmr(x, x_hat, imt):
    # NMR
    err = torch.mean(torch.mean((0.5 * (x - x_hat).pow(2)) * imt.pow(2), dim=-1))
    return 10. * (torch.log(err + 1e-6)/l_f)


def mse(x, x_hat):
    # MSE
    return torch.mean(torch.sum(torch.pow(x - x_hat, 2.), dim=-1))

# EOF
