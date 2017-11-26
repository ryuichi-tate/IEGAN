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
parser.add_argument('--history', dest='history', action='store_true',
                    help='save loss history')
parser.add_argument('--name', type=str, default='autoencoder', metavar='NAME',
                    help='name of the output directories (default: autoencoder)')
parser.set_defaults(history=False)

opt = parser.parse_args()

# set params
# ===============
cuda = 1
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
IMAGE_PATH = 'images/'+opt.name
MODEL_FULLPATH = 'models/'+opt.name+'.pth'

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
from model import Autoencoder
from utils import *


def train():
    ae = Autoencoder()
    # load trained model
    # model_path = ''
    # g.load_state_dict(torch.load(model_path))

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.decay)

    # load dataset
    # ==========================
    kwargs = dict(num_workers=1, pin_memory=True) if cuda else {}
    dataloader = DataLoader(
        datasets.MNIST('MNIST', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=opt.batch_size, shuffle=True, **kwargs
    )
    N = len(dataloader)

    # get sample batch
    dataiter = iter(dataloader)
    samples, _ = dataiter.next()
    # cuda
    if cuda:
        ae.cuda()
        criterion.cuda()
        samples = samples.cuda()
    samples = Variable(samples)

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

            # forward & backward & update params
            ae.zero_grad()
            _, outputs = ae(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            loss_mean += loss.data[0]
            if opt.history:
                loss_history[N*epoch + i] = loss.data[0]
            show_progress(epoch+1, i+1, N, loss.data[0])

        print('\ttotal loss (mean): %f' % (loss_mean/N))
        # generate fake images
        _, reconst = ae(samples)
        vutils.save_image(reconst.data,
                          os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
                          normalize=False)
    # save models
    torch.save(ae.state_dict(), MODEL_FULLPATH)
    # save loss history
    if opt.history:
        np.save('history/'+opt.name, loss_history)


if __name__ == '__main__':
    train()
