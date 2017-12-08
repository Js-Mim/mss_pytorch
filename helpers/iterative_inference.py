# -*- coding: utf-8 -*-
__author__ = 'J.F. Santos, K. Drossos'

import torch
from losses import loss_functions


def iterative_inference(module, x, criterion=None, tol=1e-9, max_iter=10):
    if criterion is None:
        criterion = loss_functions.mse

    y0 = module(x)
    for k in range(max_iter):
        y = module(y0)
        if criterion(y0, y).data[0] < tol:
            break
        else:
            y0 = y
    return y, k


def iterative_recurrent_inference(module, H_enc, criterion=None, tol=1e-9, max_iter=10):
    if criterion is None:
        criterion = loss_functions.mse

    y0 = module(H_enc)
    for iter in range(max_iter):
        y = module(y0)
        if criterion(y0, y).data[0] < tol:
            break
        else:
            y0 = y
    return y


if __name__ == '__main__':
    model = torch.nn.Linear(10, 10)
    x = torch.autograd.Variable(torch.rand(4, 10))
    ground_truth = torch.autograd.Variable(x.data + 0.1 * torch.rand(4, 10))
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())

    for k in range(10000):
        y, n_iter = iterative_inference(model, x)
        loss = criterion(y, ground_truth)
        loss.backward()
        opt.step()
        print('Iter. {} - loss = {}'.format(k, loss))

