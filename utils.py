# -*- coding: utf-8 -*-
import math
import numpy as np
import sys
import torch
import torch.nn.functional as F

def combine_images(generated_images):
    total,width,height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image

def show_progress(e,b,b_total,loss):
    sys.stdout.write("\r%d: [%d / %d] loss: %f" % (e,b,b_total,loss))
    sys.stdout.flush()

def self_distance(x, eps=1e-6):
    # mat = x.view(x.size(0),-1)
    mat = x
    r = torch.mm(mat, mat.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    dist = diag + diag.t() - 2*r + eps
    return dist.sqrt().sum(1)

def pairwise_distance(x1, x2, p=2, eps=1e-6):
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).view(x1.size(0), -1).sum(dim=1, keepdim=True)
    return torch.pow(out, 1. / p)

# Mean Quantile Information Estimator
def mqjs(input, target, eps=1e-6):
    # assert input.size() == target.size()
    n = input.size()[0]
    x1 = input.view(n, -1)
    x2 = target.view(n, -1)
    p_ = (x1 + x2) / 2
    ce = (torch.log(torch.clamp(F.pairwise_distance(x1, x2, p=2), min=eps)) +
          torch.log(torch.clamp(F.pairwise_distance(x1, x2, p=2), min=eps)))
    se = (torch.log(torch.clamp(self_distance(x1), min=eps)) +
          torch.log(torch.clamp(self_distance(x2), min=eps)) * n / (n-1)) / 2
    return torch.mean(torch.abs(ce - se))

