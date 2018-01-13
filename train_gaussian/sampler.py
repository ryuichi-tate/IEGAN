# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset

class GaussianMixtureDataset(Dataset):
    def __init__(self, total=100000, gridsize=5, std=0.05, transform=None):
        self.total = total
        self.gridsize = gridsize
        self.std = std
        self.transform = transform
        # generate 2D gaussian mixture dataset
        self._grid_gaussian_mixture()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        point = self.data[idx]
        if self.transform:
            point = self.transform(point)

        return point

    def _grid_gaussian_mixture(self):
        print('generating %d points...' % self.total)
        print('std: %f' % self.std)
        print('grid: %d x %d' % (self.gridsize, self.gridsize))
        indexes = np.random.randint(0,self.gridsize**2, self.total)
        x = np.expand_dims(indexes % self.gridsize, -1)
        y = np.expand_dims(indexes // self.gridsize, -1)
        means = np.concatenate((x,y), axis=1)
        self.data = np.random.normal(means, self.std**2)
        print('shape: ', self.data.shape)
        print('done.')

def grid_gaussian_mixture(total, gridsize, std=0.01):
    print('generating %d points...' % total)
    print('std: %f' % std)
    print('grid: %d x %d' % (gridsize, gridsize))
    indexes = np.random.randint(0,gridsize**2, total)
    x = np.expand_dims(indexes % gridsize, -1)
    y = np.expand_dims(indexes // gridsize, -1)
    means = np.concatenate((x,y), axis=1)
    means = (means + 1) / (gridsize + 1)

    return np.random.normal(means, std**2)

