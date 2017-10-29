# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, zdim, ch=1):
        super().__init__()
        self.zdim = zdim
        self.ch = ch
        self.filters = 32
        self.convt1 = nn.ConvTranspose2d(zdim, self.filters*4, 4, stride=1, padding=0, bias=False)
        self.convt2 = nn.ConvTranspose2d(self.filters*4, self.filters*2, 4, stride=2, padding=1, bias=False)
        self.convt3 = nn.ConvTranspose2d(self.filters*2, self.filters, 4, stride=2, padding=1, bias=False)
        self.convt4 = nn.ConvTranspose2d(self.filters, self.ch, 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.filters*4)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.bn3 = nn.BatchNorm2d(self.filters)
        self.bn4 = nn.BatchNorm2d(self.ch)

    def forward(self, x):
        # x = F.selu(self.convt1(x))           # (?,zdim, 1, 1) => (?,128, 4, 4)
        x = F.selu(self.bn1(self.convt1(x))) # (?,zdim, 1, 1) => (?,128, 4, 4)
        x = F.selu(self.bn2(self.convt2(x))) # (?, 128, 4, 4) => (?, 64, 8, 8)
        x = F.selu(self.bn3(self.convt3(x))) # (?,  64, 8, 8) => (?, 32,16,16)
        x = F.tanh(self.bn4(self.convt4(x))) # (?,  32,16,16) => (?, ch,32,32)

        return x

