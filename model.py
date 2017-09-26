# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.regularizers import l2
from keras.layers import Flatten, Dropout

def generator(input_dim=100,units=256,width=28,activation='relu',ch=1):
    w4 = int(width/4)
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units*(w4)*(w4), kernel_regularizer=l2(0.0001)))
    # model.add(Dropout(0.2))
    model.add(Activation(activation))
    model.add(Reshape((w4, w4, units), input_shape=(units*w4*w4,)))

    model.add(Conv2DTranspose(int(units/2), 4, strides=2, padding='same', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Activation(activation))

    model.add(Conv2DTranspose(ch, 4, strides=2, padding='same', kernel_regularizer=l2(0.0001)))
    # model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.summary()
    return model

def generator_conditional(input_dim=100,units=256,width=28,activation='relu',ch=1):
    w4 = int(width/4)
    model = Sequential()
    model.add(Dense(input_dim=input_dim+10, units=units*(w4)*(w4), kernel_regularizer=l2(0.0001)))
    # model.add(Dropout(0.2))
    model.add(Activation(activation))
    model.add(Reshape((w4, w4, units), input_shape=(units*w4*w4,)))

    model.add(Conv2DTranspose(int(units/2), 4, strides=2, padding='same', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Activation(activation))

    model.add(Conv2DTranspose(ch, 4, strides=2, padding='same', kernel_regularizer=l2(0.0001)))
    # model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.summary()
    return model

def generator_upsampling(input_dim=100,units=1024,activation='relu'):
    model = Sequential()
    model.add(Dense(units=units, input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(activation))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.summary()
    return model

