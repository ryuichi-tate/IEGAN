# coding: utf-8
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='GPU',
                    help='set CUDA_VISIBLE_DEVICES (default: 0)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='# of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0, metavar='D',
                    help='weight decay or L2 penalty (default: 0)')
parser.add_argument('-z', '--zdim', type=int, default=100, metavar='Z',
                    help='dimension of latent vector (default: 0.5)')
parser.add_argument('--history', dest='history', action='store_true',
                    help='save loss history')
parser.add_argument('--name', type=str, default='', metavar='NAME',
                    help='name of the output directories (default: None)')
parser.add_argument('-t', '--threshold', type=int, default=150, metavar='T',
                    help='param of threshold function (default: 150)')
parser.set_defaults(history=False)

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

if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists('history'):
    print('mkdir history')
    os.mkdir('history')

# ===============
import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Generator, Generator01
from entropy_estimator import MQKLLoss
from utils import *


def train():
    g = Generator01(Zdim)
    # load trained model
    # model_path = ''
    # g.load_state_dict(torch.load(model_path))

    # custom loss function
    # ==========================
    criterion = MQKLLoss(th=opt.threshold)

    # setup optimizer
    # ==========================
    optimizer = optim.Adam(g.parameters(), lr=opt.lr, weight_decay=opt.decay)


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
        datasets.MNIST('MNIST', download=True,
                       transform=transforms.Compose([
                           transforms.Scale(32),
                           transforms.ToTensor()
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BS, shuffle=True, **kwargs
    )
    N = len(dataloader)
    if opt.history:
        loss_history = np.empty(N*opt.epochs, dtype=np.float32)
    # train
    # ==========================
    for epoch in range(opt.epochs):
        loss_mean = 0.0
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

            loss_mean += loss.data[0]
            if opt.history:
                loss_history[N*epoch + i] = loss.data[0]
            show_progress(epoch+1, i+1, N, loss.data[0])

        print('\ttotal loss (mean): %f' % (loss_mean/N))
        # generate fake images
        vutils.save_image(g(z_pred).data,
                          os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
                          normalize=False)
                          # normalize=True)
    # save models
    torch.save(g.state_dict(), MODEL_FULLPATH)
    # save loss history
    if opt.history:
        np.save('history'+opt.name, loss_history)


if __name__ == '__main__':
    train()
