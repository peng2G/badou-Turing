import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../lenna.png")
# 将像素值转化为sample生成数据集
data = img.reshape((-1,3))
data = np.float32(data)

# 设置停止条件
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,
            10,1.0)
# 设置标签
flags =cv2.KMEANS_RANDOM_CENTERS

compactness,labels,centers=cv2.kmeans(data,16,None,criteria,10,flags)

centers=np.uint8(centers)
result= centers[labels.flatten()]
result = result.reshape(img.shape)

# BGR转RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
images =[img,result]

# 中文显示
plt.rcParams['font.sans-serif']=["SimHei"]

titles = ["原始图像","聚类图像K=16"]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.savefig(fname='图像聚类',figsize=[7,4])
plt.show()