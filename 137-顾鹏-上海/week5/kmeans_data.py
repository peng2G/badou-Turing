from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# 生成随机数据


data_set = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]

div = KMeans(n_clusters=3)
y_label = div.fit_predict(data_set)
x= np.array([n[0] for n in data_set],dtype='float32')
y= np.array([n[1] for n in data_set],dtype='float32')
# 可视化结果
plt.scatter(x[y_label.flatten()==0],y[y_label.flatten()==0],marker='x')
plt.scatter(x[y_label.flatten()==1],y[y_label.flatten()==1],marker='x')
plt.scatter(x[y_label.flatten()==2],y[y_label.flatten()==2],marker='x')
plt.title('Kmeans result data')
plt.xlabel("x data")
plt.ylabel("y data")
plt.legend(['A', 'B', 'C'])
plt.savefig(fname="k_means_data",figsize=[4,3])
plt.show()