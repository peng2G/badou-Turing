import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 定义数据集

x_data = np.linspace(0,10,2000)[:, np.newaxis] # 转样本
noise = np.random.normal(0,0.01,x_data.shape)
y_data = np.sin(x_data)+noise


# 构建神经网络
#3层网络
# 定义占位符
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
# 定义L1 20个神经元
wl1 = tf.Variable(tf.random_normal([1,20]))
bsl1 = tf.Variable(tf.random_normal([1,20]))
l1_init = tf.matmul(x,wl1)+bsl1
l1 = tf.nn.tanh(l1_init)
# 定义L2层 20个神经元
wl2 = tf.Variable(tf.random_normal([20, 20]))
bsl2 = tf.Variable(tf.random_normal([1, 20]))
l2_init = tf.matmul(l1, wl2) + bsl2
l2 = tf.nn.tanh(l2_init)
# 输出
wl3 = tf.Variable(tf.random_normal([20,1]))
bsl3 =tf.Variable(tf.random_normal([1]))
prediction = tf.matmul(l2, wl3)+bsl3
# cost
loss = tf.reduce_mean(tf.square(y-prediction))
# optimizer
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

# 训练神经网络
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(6000):
        sess.run(train_step, feed_dict={x:x_data,y:y_data})
    prediction_val = sess.run(prediction, feed_dict={x:x_data})
    # 结果查看
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
plt.scatter(x_data,y_data,label='实际值')
plt.plot(x_data,prediction_val,'r-',lw=5,label = '预测值')
plt.legend()
plt.show()
