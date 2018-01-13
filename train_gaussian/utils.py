# -*- coding: utf-8 -*-
import math
import numpy as np
import sys
import torch
import torch.nn.functional as F
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt

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

# def imshow_torch(im, figsize=(10,10), **kwargs):
#     '''
#     Args:
#       im(torch.FloatTensor)
#     '''
#     plt.figure(figsize=figsize)
#     plt.imshow(make_grid(im, **kwargs).numpy().transpose(1,2,0))
#     plt.axis('off')
#     plt.show()

def show_progress(e,b,b_total,loss):
    sys.stdout.write("\r%3d: [%5d / %5d] loss: %f" % (e,b,b_total,loss))
    sys.stdout.flush()

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
