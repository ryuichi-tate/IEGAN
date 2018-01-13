# coding: utf-8
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1', metavar='GPU',
                    help='set CUDA_VISIBLE_DEVICES (default: -1)')
parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='# of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0, metavar='D',
                    help='weight decay or L2 penalty (default: 0)')
parser.add_argument('-z', '--zdim', type=int, default=2, metavar='Z',
                    help='dimension of latent vector (default: 2)')
parser.add_argument('--std', type=float, default=0.01, metavar='S',
                    help='std of gaussian (default: 0.01)')
parser.add_argument('--gridsize', type=float, default=5, metavar='W',
                    help='gridsize (default: 5)')
parser.add_argument('--total', type=float, default=100000, metavar='N',
                    help='# train (default: 100000)')
parser.add_argument('--history', dest='history', action='store_true',
                    help='save loss history')
parser.add_argument('--name', type=str, default='gauss', metavar='NAME',
                    help='name of the output directories (default: None)')
parser.add_argument('-t', '--threshold', type=int, default=0, metavar='T',
                    help='param of threshold function (default: 0)')
parser.set_defaults(history=False)

opt = parser.parse_args()

os.chdir('..')
# set params
# ===============
cuda = 0 if opt.gpu == -1 else 1
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
opt.name = opt.name if opt.name == '' else '/'+opt.name
IMAGE_PATH = 'images'+opt.name
MODEL_PATH = 'models/'
model_name = 'model' if opt.name == '' else opt.name
MODEL_FULLPATH = 'models/'+model_name+'.pth'
N = opt.total

if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists('history'):
    print('mkdir history')
    os.mkdir('history')

# ===============
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from model import Generator
from sampler import grid_gaussian_mixture
from entropy_estimator import MQKLLoss, MQSymKLLoss

from utils import *

def train():
    g = Generator()
    print(g)

    # custom loss function
    # ==========================
    criterion = MQSymKLLoss(th=opt.threshold)
    # setup optimizer
    # ==========================
    optimizer = optim.Adam(g.parameters(), lr=opt.lr, weight_decay=opt.decay)

    z = torch.FloatTensor(BS, Zdim).normal_(0, 1)
    z_pred = torch.FloatTensor(100, Zdim).normal_(0, 1)
    # cuda
    if cuda:
        g.cuda()
        criterion.cuda()
        z, z_pred = z.cuda(), z_pred.cuda()

    z_pred = Variable(z_pred)

    # load dataset
    # ==========================
    data = Variable(torch.FloatTensor(
            grid_gaussian_mixture(N, opt.gridsize, opt.std)
           ).cuda())
    if opt.history:
        loss_history = np.empty(N*opt.epochs, dtype=np.float32)
    # train
    # ==========================
    assert data.size(0) % BS == 0  # TODO
    num_batches = int(np.ceil(data.size(0) / BS))
    print('# batch: ', num_batches)

    for epoch in range(opt.epochs):
        loss_mean = 0.0
        for i in range(num_batches):

            minibatch = data[i*BS:(i+1)*BS]
            g.zero_grad()
            # forward & backward & update params
            z.resize_(BS, Zdim).normal_(0, 1)
            zv = Variable(z)
            outputs = g(zv)
            loss = criterion(outputs, minibatch)
            loss.backward()
            optimizer.step()

            loss_mean += loss.data[0]
            if opt.history:
                loss_history[N*epoch + i] = loss.data[0]
            show_progress(epoch+1, i+1, num_batches, loss.data[0])

        print('\ttotal loss (mean): %f' % (loss_mean/num_batches))
        # generate fake images
        # save_scatter(g(z_pred).data.cpu().numpy(),
        #              os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)))
        torch.save(g.state_dict(), MODEL_FULLPATH)

    # save models
    torch.save(g.state_dict(), MODEL_FULLPATH)
    # save loss history
    if opt.history:
        np.save('history'+opt.name, loss_history)

# def save_scatter(out, filepath):
#     plt.clf()
#     plt.figure(figsize=(8,8))
#     plt.scatter(out[:,0],out[:,1],s=2,alpha=0.1,edgecolor='black',linewidth=0.5,color='blue')
#     plt.savefig(filepath)

if __name__ == '__main__':
    train()
