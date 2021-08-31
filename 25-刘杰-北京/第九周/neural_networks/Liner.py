#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Liner.py
@Author  ：luigi
@Date    ：2021/8/27 下午4:41 
'''
import numpy as np
import torch

class Liner():

    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if self.bias:
            self.weights = np.random.randn(self.in_features + 1, self.out_features)
        else:
            self.weights = np.random.randn(self.in_features, self.out_features)


    def forward(self, input):

        batch_size = input.shape[0]
        input = input.numpy() if torch.is_tensor(input) else input
        X = np.hstack((input, np.ones((batch_size, 1)))) if self.bias else input
        return np.dot(X, self.weights)

    def __call__(self, input):
        return self.forward(input)

