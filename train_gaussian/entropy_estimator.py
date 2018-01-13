# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def entropy(x, th, eps):
    mat = x.view(x.size(0), -1)
    x2 = torch.mm(mat, mat.t())
    diag = x2.diag().unsqueeze(0)
    diag = diag.expand_as(x2)
    dist = diag + diag.t() - 2*x2 + eps
    # dist = -F.threshold(-dist, -150, -1)
    return dist.clamp(min=th).sqrt().log().sum(1)

def cross_entropy(x1, x2, th, eps):
    x1 = x1.view(x1.size(0), -1)
    x2 = x2.view(x2.size(0), -1)
    x1x2 = torch.mm(x1, x2.t())
    x1_2 = torch.mm(x1,x1.t()).diag().unsqueeze(0).expand_as(x1x2).t()
    x2_2 = torch.mm(x2,x2.t()).diag().unsqueeze(0).expand_as(x1x2)
    dist = x1_2 + x2_2 - 2*x1x2 + eps
    # dist = F.threshold(dist, th, 1)
    return dist.clamp(min=th).sqrt().log().sum(1)

def mqkl(input, target, th, eps):
    n = input.size()[0]
    ce = cross_entropy(input, target, th, eps)
    se = entropy(input, th, eps) * n / (n-1)
    return torch.mean(ce - se)

def mqsymkl(input, target, th, eps=1e-6):
    n = input.size()[0]
    ce = cross_entropy(input, target, th, eps) + cross_entropy(target, input, th, eps)
    se = entropy(input, th, eps) * n / (n-1)
    return torch.mean(ce - se)

class MQKLLoss(nn.Module):
    def __init__(self, th, eps=1e-6):
        super().__init__()
        self.th = th
        self.eps = eps

    def forward(self, input, target):
        return mqkl(input, target, self.th, self.eps)

class MQSymKLLoss(nn.Module):
    def __init__(self, th, eps=1e-6):
        super().__init__()
        self.th = th
        self.eps = eps

    def forward(self, input, target):
        return mqsymkl(input, target, self.th, self.eps)

# Energy
# ==================================
def entropy_energy(x):
    mat = x.view(x.size(0), -1)
    x2 = torch.mm(mat, mat.t())
    diag = x2.diag().unsqueeze(0)
    diag = diag.expand_as(x2)
    dist = diag + diag.t() - 2*x2
    return dist.sum(1)

def cross_entropy_energy(x1, x2):
    x1 = x1.view(x1.size(0), -1)
    x2 = x2.view(x2.size(0), -1)
    x1x2 = torch.mm(x1, x2.t())
    x1_2 = torch.mm(x1,x1.t()).diag().unsqueeze(0).expand_as(x1x2).t()
    x2_2 = torch.mm(x2,x2.t()).diag().unsqueeze(0).expand_as(x1x2)
    dist = x1_2 + x2_2 - 2*x1x2
    return dist.sum(1)

def mqsymklenergy(input, target):
    n = input.size()[0]
    ce = cross_entropy_energy(input, target) + cross_entropy_energy(target, input)
    se = entropy_energy(input) * n / (n-1)
    return torch.mean(ce - se)

class MQKLEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return mqsymklenergy(input, target)

# JS
# ==================================
def mqjs(input, target, eps=1e-6):
    n = input.size()[0]
    p_ = (input + target) / 2
    ce = cross_entropy(input, p_, eps) + cross_entropy(target, p_, eps)
    # se = (entropy(input, eps) + entropy(target, eps)) * n / (n-1)
    se = entropy(input, eps) * n / (n-1)
    return torch.mean(ce - se)

class MQJSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqjs(input, target, self.eps)

