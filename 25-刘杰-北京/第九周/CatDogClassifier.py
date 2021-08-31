#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：CatDogClassifier.py
@Author  ：luigi
@Date    ：2021/9/1 3:29 下午 
'''

from torch.utils.data import DataLoader
from datasets.CatDogDataset import CatDogDataset
from neural_networks import Liner
from neural_networks import Relu

def get_data(image_path):
    train = CatDogDataset(image_path)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

def get_model():

    # model = [
    #     Liner(784,100),
    #     Relu(),
    #     Liner(784,2)]

    model = Liner(784,100)

    return model


def main():
    image_path = "/Users/liujie/工作相关/data/test/*"
    trn_dl = get_data(image_path)
    # model_list = get_model()


    for ix, batch in enumerate(iter(trn_dl)):
        x, y =batch
        # for model in model_list:
        #     x = model(x)
        x = Liner(784,100)(x)
        print(x)



if __name__ == '__main__':
    main()