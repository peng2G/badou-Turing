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
from neural_networks.linear.Linear import Liner
from neural_networks.activation.Relu import Relu
from neural_networks.activation.Sigmoid import Sigmoid
from neural_networks.loss.BCELoss import BCELoss

def get_data(image_path):
    train = CatDogDataset(image_path)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

def get_model():

    models = [
        Liner(784,100),
        Relu(),
        Liner(100,1),
        Sigmoid()
        ]

    loss_fn = BCELoss()


    return models, loss_fn


def main():
    image_path = "/Users/liujie/工作相关/data/test/*"
    trn_dl = get_data(image_path)
    models, loss_fn = get_model()


    for ix, batch in enumerate(iter(trn_dl)):

        x, y =batch
        for model in models:
            x = model(x)
        loss = loss_fn(x, y.numpy())
        print(x)
        print(loss)
        # print(x.shape,y.shape)




if __name__ == '__main__':
    main()