# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def entropy(x, eps=1e-6):
    mat = x.view(x.size(0), -1)
    r = torch.mm(mat, mat.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    dist = diag + diag.t() - 2*r + eps
    dist.add_(Variable(torch.eye(x.size(0)).cuda())) # avoid log(0)
    return dist.clamp(min=1).sqrt().log().sum(1)

def cross_entropy(x1, x2, eps=1e-6):
    x1 = x1.view(x1.size(0), -1)
    x2 = x2.view(x2.size(0), -1)
    x1x2 = torch.mm(x1, x2.t())
    x2x1 = torch.mm(x2, x1.t())
    x1_2 = torch.mm(x1,x1.t()).diag().unsqueeze(0).expand_as(x1x2).t()
    x2_2 = torch.mm(x2,x2.t()).diag().unsqueeze(0).expand_as(x1x2)
    # x^2 + y^2 - xy - yx
    dist = x1_2 + x2_2 - 2*x1x2 + eps
    return dist.clamp(min=1).sqrt().log().sum(1)

def mqjs(input, target, eps=1e-6):
    n = input.size()[0]
    p_ = (input + target) / 2
    ce = cross_entropy(input, p_, eps) + cross_entropy(target, p_, eps)
    # se = (entropy(input, eps) + entropy(target, eps)) * n / (n-1)
    se = entropy(input, eps) * n / (n-1)
    return torch.mean(ce - se)

def mqjs2(input, target, eps=1e-6):
    n = input.size()[0]
    p_ = (input + target) / 2
    ce = cross_entropy(input, p_, eps) + cross_entropy(target, p_, eps)
    # se = (entropy(input, eps) + entropy(target, eps)) * n / (n-1)
    se = entropy(input, eps) * n / (n-1)
    return torch.mean(torch.exp(ce - se))

def mqkl(input, target, eps=1e-6):
    n = input.size()[0]
    ce = cross_entropy(input, target, eps)
    se = entropy(input, eps) * n / (n-1)
    return torch.mean(ce - se)

class MQJSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqjs(input, target, self.eps)

class MQKLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqkl(input, target, self.eps)

class MQJSLoss2(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqjs2(input, target, self.eps)

