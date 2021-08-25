#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：NeuralNetwork.py
@Author  ：luigi
@Date    ：2021/8/16 上午11:35 
'''

import numpy as np
import cv2


class Layer():

    def __init__(self, in_features, out_features, activation_fn='sigmoid'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn

    def activate(self, z):
        if self.activation_fn == 'sigmoid':
            return 1/(1 + np.exp(-z))
        elif self.activation_fn == 'relu':
            return max(0, z)
        elif self.activation_fn == 'softmax':
            #refer https://deepnotes.io/softmax-crossentropy
            exps = np.exp(z-np.max(z))
            return exps/np.sum(exps)



class NeuralNetwork():

    def __init__(self, image, label):
        self.input = image.resize((32, 32)).flatten()
        self.label = label
        self.X = [self.input]
        self.layers = []
        self.weights = []

    def append(self, layer: Layer):
        return self.layers.append(layer)

    def loss(self, y, y_hat, loss_fn='mse'):
        if loss_fn == 'mse':
            return np.mean(np.square(y, y_hat))
        elif loss_fn == 'cross_entropy':
            return -np.dot(y, np.log(y_hat))

    def forward(self):
        for ix, layer in enumerate(self.layers):
            weight = np.random.normal(size=(layer.out_features, layer.in_features + 1))
            x = np.hstack([self.X[ix], np.ones(1)])
            z = np.dot(weight, x.T)
            a = layer.activate(z)
            self.weights.append(weight)
            self.X.append(a)


    def backward(self, learning_rate):
        assert len(self.weights) !=0, "weights list is empty, check if forward method is corrected called"
        for ix, layer in enumerate(self.layers[::-1]):
            self.weights[ix] += learning_rate*(self.label-self.X[-1])
            
def main():
    nn = NeuralNetwork()
    input_layer = Layer(784,128, activation_fn='sigmoid')
    hidden_layer = Layer(128,10, activation_fn='softmax')
    nn.append(input_layer)
    nn.append(hidden_layer)
    nn.forward()
