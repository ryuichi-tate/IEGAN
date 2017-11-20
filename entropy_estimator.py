# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def entropy(x, th, eps=1e-6):
    mat = x.view(x.size(0), -1)
    x2 = torch.mm(mat, mat.t())
    diag = x2.diag().unsqueeze(0)
    diag = diag.expand_as(x2)
    dist = diag + diag.t() - 2*x2
    # dist.add_(Variable(torch.eye(x.size(0)).cuda())) # avoid log(0)
    # print('E >> max: {} | mean: {} | min: {}'.format(torch.max(dist), torch.mean(dist), torch.min(dist)))
    dist = F.threshold(dist, th, 1)
    return dist.clamp(min=1).sqrt().log().sum(1)

def cross_entropy(x1, x2, th, eps=1e-6):
    x1 = x1.view(x1.size(0), -1)
    x2 = x2.view(x2.size(0), -1)
    x1x2 = torch.mm(x1, x2.t())
    x1_2 = torch.mm(x1,x1.t()).diag().unsqueeze(0).expand_as(x1x2).t()
    x2_2 = torch.mm(x2,x2.t()).diag().unsqueeze(0).expand_as(x1x2)
    # x^2 + y^2 - xy - yx
    dist = x1_2 + x2_2 - 2*x1x2
    # print('CE >> max: {} | mean: {} | min: {}'.format(torch.max(dist), torch.mean(dist), torch.min(dist)))
    dist = F.threshold(dist, th, 1)
    return dist.clamp(min=1).sqrt().log().sum(1)

def mqkl(input, target, th, eps=1e-6):
    n = input.size()[0]
    ce = cross_entropy(input, target, th, eps)
    se = entropy(input, th, eps) * n / (n-1)
    return torch.mean(ce - se)

class MQKLLoss(nn.Module):
    def __init__(self, th, eps=1e-6):
        super().__init__()
        self.th = th
        self.eps = eps

    def forward(self, input, target):
        return mqkl(input, target, self.th, self.eps)

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

