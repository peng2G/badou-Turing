#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Preprocessor.py
@Author  ：luigi
@Date    ：2021/8/31 3:37 下午 
'''

import cv2

class Preprocessor():

    def __init__(self, width, height, inter=cv2.INTER_AREA):

        self.width = width
        self.height = height
        self.inter = inter


    def preprocess(self, image):

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

