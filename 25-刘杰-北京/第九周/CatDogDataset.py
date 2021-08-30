#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：CatDogDataset.py
@Author  ：luigi
@Date    ：2021/8/30 下午9:41 
'''

import cv2

class CatDogDataSet():

    def __init__(self,x, y):
        x = x.reshape(-1,28*28)
        self.x = x
        self.y = y



