import numpy as np
import math
import cv2


class CannyUser():
    def __init__(self, img):
        self.img = img
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.width = self.gray_img.shape[0]
        self.heigh = self.gray_img.shape[1]

    def _get_gussion_kernal(self, dim=3, sigma=1.4):
        kernal_matrix = np.zeros((dim, dim))
        gussion_sum = 0
        for i in range(dim):
            for j in range(dim):
                kernal_matrix[i, j] = math.exp(
                    -((i - 0.5 * dim + 0.5) ** 2 + (j - 0.5 * dim + 0.5) ** 2) / (2 * sigma ** 2))
                gussion_sum = gussion_sum + kernal_matrix[i, j]
        # 归一化
        gussion_kernal = kernal_matrix / gussion_sum
        return gussion_kernal

    def do_gussion_filter(self, dim=3, sigma=1.4):
        gussion_kernal = self._get_gussion_kernal(dim, sigma)
        image_w, image_h = self.img.shape
        new_gray = np.zeros([image_w, image_h])
        for i in range(image_w):
            for j in range(image_h):
                new_gray[i, j] = np.sum(self.img[i:i + 5, j:j + 5] * gussion_kernal)  # 卷积操作
        return new_gray

    def get_gradients(self, gray_image):
        W, H = gray_image.shape
        dx = np.zeros([W - 1, H - 1])
        dy = np.zeros([W - 1, H - 1])
        M = np.zeros([W - 1, H - 1])
        theta = np.zeros([W - 1, H - 1])
        for i in range(W - 1):
            for j in range(H - 1):
                dx[i, j] = gray_image[i + 1, j] - gray_image[i, j]
                dy[i, j] = gray_image[1, j + 1] - gray_image[i, j]
                M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
                theta[i, j] = math.atan(dx[i, j] / dy[i, j] + 0.000000001)  # 防止除0

        self.dx = dx
        self.dy = dy
        self.M = M
        self.theta = theta
        return dx, dy, M, theta

    # 非极大值一直
    def NMS(self):
        nms = np.copy(self.M)
        W, H = self.M.shape
        nms[0, :] = nms[W - 1, :] = nms[:, 0] = nms[: H - 1] = 0  # 存储最终的结果
        for i in range(W - 1):
            for j in range(H - 1):
                # 判断是否是边缘点
                ndx = self.dx[i, j]
                ndy = self.dy[i, j]
                theta = self.theta[i, j]
                if self.M[i, j] == 0:
                    nms[i, j] = 0
                else:
                    ratio_y2x = ndy / ndx
                    ratio_x2y = ndx / ndy
                    if theta <= 0.25 * np.pi:
                        gdf = ratio_y2x * self.M[i + 1, j + 1] + (1 - ratio_y2x) * self.M[i + 1, j]
                        gdb = ratio_y2x * self.M[i - 1, j - 1] + (1 - ratio_y2x) * self.M[i - 1, j]

                    elif theta <= 0.5 * np.pi:
                        gdf = ratio_x2y * self.M[i + 1, j + 1] + (1 - ratio_x2y) * self.M[i, j + 1]
                        gdb = ratio_x2y * self.M[i - 1, j - 1] + (1 - ratio_x2y) * self.M[i, j - 1]
                    elif theta <= 0.75 * np.pi:

                        gdf = -ratio_x2y * self.M[i - 1, j + 1] + (1 + ratio_x2y) * self.M[i, j + 1]
                        gdb = -ratio_x2y * self.M[i + 1, j - 1] + (1 + ratio_x2y) * self.M[i, j - 1]
                    else:
                        gdf = -ratio_y2x * self.M[i - 1, j + 1] + (1 + ratio_y2x) * self.M[i - 1, j]
                        gdb = -ratio_y2x * self.M[i + 1, j - 1] + (1 + ratio_y2x) * self.M[i + 1, j]
                    # 梯度比较
                    if abs(gdf) > nms[i, j] and abs(gdb) > nms[i, j]:
                        nms[i, j] = 0
                    else:
                        nms[i,j] = 0.5
        self.nms=nms
    def bi_threshold(self,min_ratio, max_ratio):
        M_canny = np.zeros([self.width,self.heigh])
        max_m = np.max(self.nms)
        min_threshold=max_m*min_ratio
        max_threshold=max_m*max_ratio
        for i in range(1, self.width-1):
            for j in range(1, self.heigh-1):
                if self.nms[i,j]<min_threshold:
                    M_canny[i,j] = 0
                elif self.nms[i,j]>= max_threshold:
                    M_canny[i,j] = max_threshold
                elif (self.nms[i-1, j-1:j+1] < max_threshold).any() or (self.nms[i+1, j-1:j+1].any()
                    or (self.nms[i, [j-1, j+1]] < max_threshold).any()):
                    M_canny[i,j]=max_m
            return M_canny

img = cv2.imread("../lenna.png",1)
cv2.imshow('1',img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img_gray, 60,300)
cv2.imshow('2',img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


