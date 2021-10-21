### 第三周
# 直方图均衡化

import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('../lenna.png',1)


##################################
# 画出灰度直方图
##################################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 彩图转灰度图
# 直接用plt的hist方法绘直方图
plt.figure()
plt.hist(gray.ravel(), 256)  # hist()函数会统计数据集出现的次数，将其作为纵轴
plt.show()
#使用calcHist—计算图像直方图函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
hist = cv2.calcHist([gray], [0], None, [256], [0,256]) # 统计灰度个数
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist) # 该方法绘图，实际上x 与y是解耦的，x轴坐标要自己设置
plt.xlim([0, 256])  # 设置x轴坐标范围
plt.show()

##################################
# 画出三通道的直方图
##################################
chans = cv2.split(image)
colors = ('b','g','r')
for index, item in enumerate(colors):
    plt.figure()
    plt.title("Flattened Color  %s  Histogram" %item)
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    hist = cv2.calcHist([image], [index], None, [256], [0, 256])
    plt.plot(hist, color = item)   # 绘制方法一
    plt.xlim([0, 256])
    plt.hist(chans[index].ravel(), 256)    # 绘制方法二
    plt.show()


######################
# 将灰度图做均衡化处理
######################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
plt.figure()
plt.hist(dst.ravel(), 256)
plt.title("Histogram Equalizetion")
plt.show()
cv2.imshow("Histogram Equalizetion",np.hstack([gray,dst]))
cv2.waitKey(0)
#########################
# 彩色图片各个通道均衡化处理
########################
(chan_b, chan_g, chan_r) = cv2.split(image)
chan_b =cv2.equalizeHist(chan_b)
chan_g =cv2.equalizeHist(chan_g)
chan_r =cv2.equalizeHist(chan_r)
result = cv2.merge((chan_b, chan_g, chan_r))
cv2.imshow("Histogram Equalizetion with RGB",np.hstack([image, result]))
cv2.waitKey(0)
