# 第二周作业
import cv2
import numpy as np


class ImageEditor():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        (self.width, self.height, self.chanl_num) = self.image.shape

    def color2gray(self):
        # 灰度化
        (chan_b, chan_g, chan_r) = cv2.split(self.image)
        gray_img = 0.11 * chan_b + 0.59 * chan_g + 0.3 * chan_r
        return gray_img

    def nearest_interp(self, width_pixel, height_pixel):
        # 最邻近插值
        image_target = np.zeros((width_pixel, height_pixel, self.chanl_num), dtype='uint8')
        w_ratio = self.width / width_pixel
        h_ratio = self.height / height_pixel
        for chanl in range(self.chanl_num):
            for i in range(width_pixel):
                for j in range(height_pixel):
                    w = int(w_ratio * i)
                    h = int(h_ratio * j)
                    image_target[i, j, chanl] = self.image[w, h, chanl]
        return image_target

    def bilinear_interp(self, width_pixel, height_pixel):
        # 双线性插值
        image_target = np.zeros((width_pixel, height_pixel, self.chanl_num), dtype='uint8')
        w_ratio = self.width / width_pixel
        h_ratio = self.height / height_pixel
        for chanl in range(self.chanl_num):
            for i in range(width_pixel):
                for j in range(height_pixel):
                    w = w_ratio * (i + 0.5) - 0.5
                    h = h_ratio * (j + 0.5) - 0.5
                    w1 = max(int(np.floor(w)), 0)
                    w2 = min(int(np.ceil(w)), self.width - 1)
                    h1 = max(int(np.floor(h)), 0)
                    h2 = min(int(np.ceil(h)), self.height - 1)
                    value1 = (h2 - h) * ((w2 - w) * self.image[w1, h1, chanl] + (w - w1) * self.image[w2, h1, chanl])
                    value2 = (h - h1) * ((w2 - w) * self.image[w1, h2, chanl] + (w - w1) * self.image[w2, h2, chanl])
                    image_target[i, j, chanl] = int(value2 + value1)
        return image_target

if __name__=="__main__":
    image_editor = ImageEditor("../lenna.png")
    cv2.imshow("original_image", image_editor.image)
    np_img = image_editor.nearest_interp(900, 900)
    bilinear_img = image_editor.bilinear_interp(900, 900)
    cv2.imshow("nearest_interpolation", np_img)
    cv2.imshow("bilinear_interpolation", bilinear_img)

    cv2.waitKey(0)
