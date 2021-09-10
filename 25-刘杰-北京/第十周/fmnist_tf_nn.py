#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：fmnist_tf_nn.py
@Author  ：luigi
@Date    ：2021/9/10 4:20 下午 
'''

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds

def get_data(image_path):

    train_batch = tfds.load(name="mnist", split="train", data_dir=image_path, batch_size=32,
                         shuffle=True)

    return train_batch

def get_model():

    model = Sequential(
        Dense(256, input_shape=(784,), activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    )

    loss_fn = categorical_crossentropy()

    optimizer = SGD(0.01)

    return model, loss_fn, optimizer


def main():
  image_path = '/Users/liujie/工作相关/data/tensorflow_datasets'



