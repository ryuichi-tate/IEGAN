# coding: utf-8
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='GPU',
                    help='set CUDA_VISIBLE_DEVICES (default: 0)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='# of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=1e-5, metavar='D',
                    help='weight decay or L2 penalty (default: 1e-5)')
parser.add_argument('-z', '--zdim', type=int, default=100, metavar='Z',
                    help='dimension of latent vector (default: 0.5)')
parser.add_argument('--name', type=str, default='', metavar='NAME',
                    help='name of the output directories (default: None)')
parser.add_argument('-t', '--threshold', type=int, default=1, metavar='T',
                    help='param of threshold function (default: 1)')

opt = parser.parse_args()

# set params
# ===============
cuda = 1
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
opt.name = opt.name if opt.name == '' else '/'+opt.name
IMAGE_PATH = 'images'+opt.name
model_name = 'model' if opt.name == '' else opt.name
MODEL_FULLPATH = 'models/'+model_name+'.pth'

# ===============
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from discogan import Generator, Discriminator
from itertools import chain
from dataloader import FloorPlanWithGraphDataset
from utils import *

os.chdir('..')
if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)

def train():
    g_ab = Generator(Zdim)
    g_ba = Generator(Zdim)
    d_a = Discriminator()
    d_b = Discriminator()

    # custom loss function
    # ==========================
    criterion_gan = nn.BCELoss()
    criterion_recon = nn.MSELoss()
    criterion_feature = nn.HingeEmbeddingLoss()
    # setup optimizer
    # ==========================
    optim_g = optim.Adam(chain(g_ab.parameters(),g_ba.parameters()),
                         lr=opt.lr, betas=(.5, .999), weight_decay=opt.decay)
    optim_d = optim.Adam(chain(d_a.parameters(),d_b.parameters()),
                         lr=opt.lr, betas=(.5, .999), weight_decay=opt.decay)

    # load dataset
    # ==========================
    kwargs = dict(num_workers=1, pin_memory=True) if cuda else {}
    dataloader = DataLoader(
        FloorPlanWithGraphDataset(list_pkl='lists/filelist-809971.pkl',
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ])),
        batch_size=BS, shuffle=True, **kwargs
    )
    N = len(dataloader)

    # input & output
    z = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)
    z_pred = torch.FloatTensor(64, Zdim, 1, 1).normal_(0, 1)
    y_true = Variable(torch.ones(BS, 1))
    y_pred = Variable(torch.zeros(BS, 1))
    # cuda
    if cuda:
        g.cuda()
        ae.cuda()
        criterion_gan.cuda()
        criterion_recon.cuda()
        criterion_feature.cuda()
        z, z_pred = z.cuda(), z_pred.cuda()
        y_true, y_pred = y_true.cuda(), y_pred.cuda()

    z_pred = Variable(z_pred)

    # train
    # ==========================
    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)

            g.zero_grad()
            # forward & backward & update params
            z.resize_(BS, Zdim, 1, 1).normal_(0, 1)
            zv = Variable(z)
            outputs = g(zv)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            prog_disco(epoch+1, i+1, N, loss.data[0])

        # generate fake images
        # TODO: use combine_reconst_samples
        vutils.save_image(g(z_pred).data,
                          os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
                          normalize=False)
    # save models
    torch.save(g.state_dict(), MODEL_FULLPATH)


if __name__ == '__main__':
    train()
