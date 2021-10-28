import numpy
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
# s手写一个简单的神经网络训练模型
class NeuralNetworkHandel():
    # 初始化函数
    def __init__(self, inputnodes,hiddennodes,outputnodes,learningrate):
        # 初始化神经网络,初始化各层节点数
        self.indodes = inputnodes
        self.hnodes = hiddennodes
        self.outnodes = outputnodes

        # 初始化学习率
        self.lr = learningrate

        # 初始化权重矩阵
        self.wi2h = np.random.rand(self.hnodes, self.indodes)-0.5
        self.wh2o = np.random.rand(self.outnodes, self.hnodes)-0.5

        # 初始化激活矩阵
        self.active_func = lambda x: scipy.special.expit(x)


    # 训练模型
    def train(self,input_list, target_list):
        # 训练过程分为两部分
        # 前向过程,与预测过程一样
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T


        hidden_inputs = np.dot(self.wi2h, inputs)  # 输入层

        hidden_outputs = self.active_func(hidden_inputs)  # 计算经过激活后，中间层的输出

        final_inputs = np.dot(self.wh2o, hidden_outputs)  # 输出层的输入

        final_outputs = self.active_func(final_inputs)  # 记过激活函数后 最终过得

        # 计算误差
        output_errors = targets -final_outputs  #两者位置是否有影响？

        hidden_errors = np.dot(self.wh2o,output_errors*final_outputs*(1-final_outputs))

        # 由误差计算梯度，根据梯度更新权重
        self.wh2o += self.lr*np.dot(output_errors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs))
        self.wi2h += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))

        # 后向过程
    # 预测模型
    def predict(self, inputs):

        hidden_inputs = np.dot(self.wi2h, inputs)  # 输入层
        hidden_outputs = self.active_func(hidden_inputs)  # 计算经过激活后，中间层的输出
        final_inputs =  np.dot(self.wh2o, hidden_outputs)  # 输出层的输入
        final_outputs = self.active_func(final_inputs)  # 记过激活函数后 最终过得输出
        return final_outputs


data_file = open("mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
len(data_list)
print(data_list[0])
all_values = data_list[0].split(',')

image_array = np.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
modelu = NeuralNetworkHandel(input_nodes, hidden_nodes, output_nodes, learning_rate)
