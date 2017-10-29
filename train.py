# coding: utf-8
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Generator
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='# of epochs to train (default: 100)')
# parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
#                     help='learning rate for Generator (default: 2e-4)')
# parser.add_argument('--b1', type=float, default=.9, metavar='B1',
#                     help='Adam momentum for Generator(default: 0.9)')
parser.add_argument('-z', '--zdim', type=int, default=100, metavar='Z',
                    help='dimension of latent vector (default: 0.5)')
parser.add_argument('--name', type=str, default='', metavar='NAME',
                    help='name of the output directories (default: None)')
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='GPU',
                    help='set CUDA_VISIBLE_DEVICES (default: 0)')

opt = parser.parse_args()

# set params
# ===============
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
opt.name = opt.name if opt.name == '' else '_'+opt.name
IMAGE_PATH = 'images'+opt.name
MODEL_PATH = 'models'+opt.name

BS = 32
cuda = 1
decay = 0

# custom loss function
# ==========================
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

def mqjs(input, target, eps=1e-6):
    # assert input.size() == target.size()
    n = input.size()[0]
    x1 = input.view(n, -1)
    x2 = target.view(n, -1)
    p_ = (x1 + x2) / 2
    ce = (torch.log(torch.clamp(F.pairwise_distance(x1, x2, p=2), min=eps)) +
          torch.log(torch.clamp(F.pairwise_distance(x1, x2, p=2), min=eps))) ** 2
    se = (torch.log(torch.clamp(self_distance(x1), min=eps)) +
          torch.log(torch.clamp(self_distance(x2), min=eps)) * n / (n-1))
    return torch.mean(torch.abs(ce - se))

class MQJSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqjs(input, target, self.eps)


def train():
    g = Generator(Zdim)
    # load trained model
    # model_path = ''
    # g.load_state_dict(torch.load(model_path))

    # custom loss function
    # ==========================
    criterion = MQJSLoss()

    # setup optimizer
    # ==========================
    # optimizer = optim.Adam(g.parameters(), lr=g_lr, betas=(g_b1, 0.999), weight_decay=decay)
    optimizer = optim.Adam(g.parameters())


    z = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)
    z_pred = torch.FloatTensor(64, Zdim, 1, 1).normal_(0, 1)
    # cuda
    if cuda:
        g.cuda()
        criterion.cuda()
        z, z_pred = z.cuda(), z_pred.cuda()

    z_pred = Variable(z_pred)

    # load dataset
    # ==========================
    kwargs = dict(num_workers=1, pin_memory=True) if cuda else {}
    dataloader = DataLoader(
        datasets.MNIST('mnist', download=True,
                       transform=transforms.Compose([
                           transforms.Scale(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BS, shuffle=True, **kwargs
    )
    N = len(dataloader)

    # train
    # ==========================
    for epoch in range(1,opt.epochs+1):
        loss_mean = 0.0
        for i, (imgs, labels) in enumerate(dataloader):
            if cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            imgs, labels = Variable(imgs), Variable(labels)

            g.zero_grad()
            # forward & backward & update params
            z.resize_(BS, Zdim, 1, 1).normal_(0, 1)
    #         z = np.concatenate()  # concatenate with label
            zv = Variable(z)
            outputs = g(zv)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            loss_mean += loss.data[0]
            show_progress(epoch, i, N, loss.data[0])

        print('\ttotal loss (mean): %f' % (loss_mean/N))
        # generate fake images
        vutils.save_image(g(z_pred).data,
                          os.path.join(IMAGE_PATH,'%d.png' % epoch),
                          normalize=True)
    # save models
    torch.save(g.state_dict(), os.path.join(MODEL_PATH, 'models.pth'))

if __name__ == '__main__':
    train()
