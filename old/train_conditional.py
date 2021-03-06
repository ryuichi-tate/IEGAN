# coding: utf-8
import os
import argparse
parser = argparse.ArgumentParser()
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
parser.add_argument('--name', type=str, default='cond', metavar='NAME',
                    help='name of the output directories (default: cond)')
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

if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists(MODEL_PATH):
    print('mkdir ', MODEL_PATH)
    os.mkdir(MODEL_PATH)

cuda = 1
classes = 10

# ===============
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Generator
from utils import *

# custom loss function
# ==========================
class MQJSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        return mqjs(input, target, self.eps)

def sampler_normal_categorical():
    noise = np.random.normal(0,1,(classes**2, Zdim))
    label = to_categorical(np.array([range(classes) for _ in range(classes)]))
    return np.concatenate((noise,label), axis=-1).reshape(classes**2,Zdim+classes,1,1)

def train():
    g = Generator(Zdim+classes)
    # load trained model
    # model_path = ''
    # g.load_state_dict(torch.load(model_path))

    # custom loss function
    # ==========================
    criterion = MQJSLoss()

    # setup optimizer
    # ==========================
    optimizer = optim.Adam(g.parameters(), lr=opt.lr, weight_decay=opt.decay)


    z = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)
    # z_pred = torch.FloatTensor(64, Zdim, 1, 1).normal_(0, 1)
    z_pred = torch.from_numpy(sampler_normal_categorical()).float()

    lv = torch.FloatTensor(BS, classes)
    # cuda
    if cuda:
        g.cuda()
        criterion.cuda()
        z, z_pred = z.cuda(), z_pred.cuda()
        lv = lv.cuda()

    z_pred = Variable(z_pred)

    # load dataset
    # ==========================
    kwargs = dict(num_workers=1, pin_memory=True) if cuda else {}
    dataloader = DataLoader(
        datasets.MNIST('MNIST', download=True,
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
            imgs = Variable(imgs)

            g.zero_grad()
            # forward & backward & update params
            z.resize_(BS, Zdim, 1, 1).normal_(0, 1)
            lv.zero_()
            lv.scatter_(1, labels.view(BS,1), 1)
            zv = Variable(torch.cat([z,lv.view(BS,classes,1,1)], dim=1))
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
                          nrow=classes, normalize=True)
    # save models
    torch.save(g.state_dict(), os.path.join(MODEL_PATH, 'models.pth'))

if __name__ == '__main__':
    train()
