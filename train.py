#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.datasets import mnist
from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
from model import *
from visualizer import *
import better_exceptions

BATCH_SIZE = 16
NUM_EPOCH = 100
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
Z_dim = 256
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

def mean_quantile_js(y_true, y_pred):
    def pairwise_euclid(T):
        T1 = K.expand_dims(T, 1)
        T2 = K.expand_dims(T, 0)
        return K.sum(tf.squared_difference(T1, T2), [2,3,4])

    N = y_true.shape[0].value
    y_bar = (y_true + y_pred) / 2

    # self_true = pairwise_euclid(y_true)
    # self_pred = pairwise_euclid(y_pred)

    # D_JS Estimator. but it doesn't work
    # =====================================
    # loss = K.abs(K.log(K.sum(K.square(y_true - y_bar), axis=[1,2,3]) * K.sum(K.square(y_pred - y_bar), axis=[1,2,3])) -
    #     K.log(K.sum(self_true, axis=1) * K.sum(self_pred, axis=1)) * N / (N-1))

    # loss = K.abs(K.log(K.sum(
    #     K.square(y_true - y_bar), axis=[1,2,3]) *
    #     K.sum(K.square(y_pred - y_bar), axis=[1,2,3])) -
    #     K.log(
    #         K.sum(K.square( K.expand_dims(y_true, 1) -
    #             K.expand_dims(y_true, 0)), axis=[1,2,3,4]) *
    #         K.sum(K.square(K.expand_dims(y_pred, 1) -
    #             K.expand_dims(y_pred, 0)), axis=[1,2,3,4])
    #     ) * N / (N-1)
    # )

    # CE only.
    # =====================================
    # L2
    # loss = K.log(K.sum(K.square(y_true - y_bar), axis=[1,2,3]) * K.sum(K.square(y_pred - y_bar), axis=[1,2,3]))
    # L1
    loss = K.log(K.sum(K.abs(y_true - y_bar), axis=[1,2,3]) * K.sum(K.abs(y_pred - y_bar), axis=[1,2,3]))
    return loss

def custom_generator(X, num_batch):
    while 1:
        for i in range(num_batch):
            last_index = min((i+1)*BATCH_SIZE, len(X))
            x = X[i*BATCH_SIZE:last_index]
            yield (np.random.normal(0,1,(len(x), Z_dim)), x)

def custom_generator_categorical(X, Y, num_batch):
    assert len(X) == len(Y)
    while 1:
        for i in range(num_batch):
            last_index = min((i+1)*BATCH_SIZE, len(X))
            x = X[i*BATCH_SIZE:last_index]
            y = Y[i*BATCH_SIZE:last_index]
            yield (np.append(np.random.normal(0,1,(len(x),Z_dim)), y, axis=1), x)

def sampler_uniform(size=BATCH_SIZE):
    return np.random.uniform(-1, 1, (size, Z_dim))

def sampler_normal(size=BATCH_SIZE):
    return np.random.normal(0, 1, (size, Z_dim))

def sampler_normal_categorical(num_classes=10):
    return np.append(np.random.normal(0,1,(num_classes**2, Z_dim)),
            to_categorical(np.array([range(num_classes) for _ in range(num_classes)])), axis=1)

class PredictionSaver(Callback):
    def __init__(self, g, sample):
        self.g = g;
        self.sample = sample;
    def on_epoch_end(self, epoch, logs={}):
        image = combine_images(self.g.predict(self.sample))
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
            .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch+1))

def train():
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    # load images
    (X_train, y_train), (_, _) = mnist.load_data()
    num_classes = 10
    num_train = len(X_train)
    num_batch = int(num_train/BATCH_SIZE)
    print("# of samples: ", num_train)
    print("# of batches: ", num_batch)
    # normalize images
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    y_train = to_categorical(y_train, num_classes)

    # original
    # g = generator(input_dim=Z_dim)
    # categorical
    g = generator(input_dim=Z_dim+num_classes)

    opt= Adam()
    # opt= Adam(lr=LR,beta_1=B1)
    g.compile(loss=mean_quantile_js, optimizer=opt)

    z_pred = sampler_normal_categorical(num_classes)

    pred_saver = PredictionSaver(g, z_pred)
    checkpointer = ModelCheckpoint(
            filepath=os.path.join(GENERATED_MODEL_PATH, 'model.h5'),
            verbose=0)

    g.fit_generator(custom_generator_categorical(X_train, y_train, num_batch),
            steps_per_epoch=num_batch,
            epochs=NUM_EPOCH,
            callbacks=[pred_saver, checkpointer])

    K.clear_session()

if __name__ == '__main__':
    train()
