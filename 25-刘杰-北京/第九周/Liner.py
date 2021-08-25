#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Liner.py
@Author  ：luigi
@Date    ：2021/8/27 下午4:41 
'''
import numpy as np

class Liner():

    def __init__(self, in_features, out_features, bias = True):

        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(self.out_features, self.in_features + 1)

        if self.bias == False:
            self.weights = np.random.randn(self.out_features, self.in_features)

    def forward(self):
