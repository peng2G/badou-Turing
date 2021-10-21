import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义一个图像编辑器类
class ImageEditer():
    def __init__(self, image_path="../lenna.png"):
        self.image = cv2.imread(image_path)
        self.width = self.image.shape[0]
        self.heigh = self.image.shape[1]
        self.peix_num = self.width * self.heigh


    def add_salt_noise(self, noise_ratio):
        '''
        椒盐噪声
        image: 图像数据
        noise_ratio: 噪声比例，越大噪声越多
        '''
        # 获取图像的 width，heigh，depth
        noise_num = int(self.width * self.heigh * noise_ratio)
        # 设置随机噪声
        imaged = self.image
        for i in range(noise_num):
            w = int(np.random.randint(0, self.peix_num) % self.width)
            h = int(np.random.randint(0, self.peix_num) % self.heigh)
            imaged[w, h] = np.random.randint(0, 2) * 255
        imaged = cv2.cvtColor(imaged,cv2.COLOR_BGR2RGB) #切换成RGB
        return imaged

    def add_gauss_noise(self, mean, mse):
        '''
        高斯噪声
        avg: 均值
        stdd: 方差
        '''
        imaged = self.image
        for w in range(self.width):
            for h in range(self.heigh):
                noise = np.random.normal(mean, mse, 3)
                imaged[w, h, 0] = self.image[w, h, 0] + noise[0]
                imaged[w, h, 1] = self.image[w, h, 1] + noise[1]
                imaged[w, h, 2] = self.image[w, h, 2] + noise[2]
        imaged = cv2.cvtColor(imaged, cv2.COLOR_BGR2RGB)  # 切换成RGB
        return imaged

if __name__=="__main__":
    path_image = r"../lenna.png"
    editor_img = ImageEditer(path_image)
    salt_noise_img=editor_img.add_salt_noise(0.1)
    gauss_noise_img = editor_img.add_gauss_noise(0.1,0.1)
    list_images=[salt_noise_img,gauss_noise_img]
    title=["椒盐噪声","高斯噪声"]
    plt.rcParams['font.sans-serif']=["SimHei"]
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(list_images[i],'gray')
        plt.title(title[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

