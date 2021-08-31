#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Relu.py
@Author  ：luigi
@Date    ：2021/9/1 4:50 下午 
'''
import numpy as np


class Relu():

    def forward(self, input):
        return np.clip(input,0, np.inf)




