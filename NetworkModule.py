import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class functional:
    class SeLU(nn.Module):
        def __init__(self):
            super(functional.SeLU, self).__init__()
            self.alpha = 1.6732632423543772848170429916717
            self.scale = 1.0507009873554804934193349852946

        def forward(self, x):
            return self.scale * F.elu(x, self.alpha)

    class Sin(nn.Module):
        def __init__(self):
            super(functional.Sin, self).__init__()

        def forward(self, x):
            return torch.sin(x)

    class Cos(nn.Module):
        def __init__(self):
            super(functional.Cos, self).__init__()

        def forward(self, x):
            return torch.cos(x)

    class Identity(nn.Module):
        def __init__(self):
            super(functional.Identity, self).__init__()

        def forward(self, x):
            return x


class Block(nn.Module):
    def __init__(self,in_f, out_f, activation_func, *args, **kwargs):
        super(Block, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_f, out_f, *args, **kwargs),
            activation_func,
            nn.BatchNorm1d(out_f),
            # nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.seq(x)


class Network(nn.Module):
    def __init__(self, L, activation_funcs):
        super(Network, self).__init__()
        if len(L) != (len(activation_funcs)+1):
            raise Exception('Error: # of layers should be # of actication functions + 1.')
        self.L = L
        bias = True
        self.blocks = []
        for i in range(len(L) - 1):
            self.blocks.append(Block(L[i], L[i + 1], activation_funcs[i],bias=bias))
        self.architecture = nn.Sequential(*self.blocks)
        # print(self.architecture)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        self.loss_value = 0.
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)


    def forward(self, x):
        x = self.architecture(x)
        return x

    def learn(self, X_data, Y_labels, n_epochs, lr=1e-3, train=True):
        self.lr = lr
        # lamb = 1e3
        # self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=1e-2, alpha=0.99, eps=1e-8,  momentum=0, centered=False)
        for epochs in range(n_epochs):
            self.optimizer.zero_grad()
            loss_per_sample = torch.mean(torch.pow(Y_labels - self.forward(X_data),2),dim=1)
            # loss_raw = (Y_labels - self.forward(X_data)).cosh().log() #L1smooth
            if train is True:
                loss = loss_per_sample.mean()
                # print(loss.item())
                loss.backward(retain_graph=True)
                # for param in self.parameters():
                #     print(param.max())
                #     param.grad *= 1-1/(lamb*param.pow(2)+1)
                self.optimizer.step()
        return loss_per_sample