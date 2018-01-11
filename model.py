# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, zdim, ch=1, nf=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zdim, nf*2, 5, stride=1, padding=0), # (?,16, 5, 5)
            nn.BatchNorm2d(nf*2),
            nn.ELU(),
            nn.ConvTranspose2d(nf*2, nf, 5, stride=2, padding=0), # (?,32,13,13)
            nn.BatchNorm2d(nf),
            nn.ELU(),
            nn.ConvTranspose2d(nf, ch, 4, stride=2, padding=0), # (?, 1,28,28)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 1
        nf = 16
        k = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nf, k, 1, 1),
            nn.ELU(),
            nn.Conv2d(nf, nf//2, k, 2, 0),
            nn.ELU(),
            nn.Conv2d(nf//2, nf//4, k, 2, 0),
            nn.ELU(),
            nn.Conv2d(nf//4, 1, k, 1, 0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, nf//4, 3, 1, 0),
            nn.ConvTranspose2d(nf//4, nf//2, k, 2, 0),
            nn.ELU(),
            nn.ConvTranspose2d(nf//2, nf, k, 2, 0),
            nn.ELU(),
            nn.ConvTranspose2d(nf, ch, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)  # (?, nf//4, 6, 6)
        decoded = self.decoder(encoded)
        return encoded, decoded  # (?,1,4,4), (?,1,28,28)

