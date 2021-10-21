# 求透视变换矩阵
####### opencv 接口
# M = cv2.getAffineTransform(src, dst)
# #src 源图像坐标  dst转换后图像坐标
# 仿射变换矩阵有六个未知参数，需要三组对应坐标
# A 为变换矩阵
# cv2.warpAffine(image, M, (w, h),) 对image图像进行仿射变换
#########
# 透视变换
# M = cv2.getPerspectiveTransform(src, dst)
# dst_image = cv2.warpPerspective(image, M, (W,H))
##########
if __name__ == '__main__':
    import numpy as np
    import cv2

    img = cv2.imread("src_photo.png")
    h, w, _ = img.shape
    src = [[161, 80], [449, 12], [1, 430], [480, 394]]
    dst = [[0, 0], [w, 0], [0, h], [h, w]]
    src = np.array(src, np.float32)
    dst = np.array(dst, np.float32)
    warp_matrix = cv2.getPerspectiveTransform(src, dst)  # 获取变换矩阵
    print("warpMatrix")
    print(warp_matrix)
    dst_img = cv2.warpPerspective(img, warp_matrix, (w, h))  # 应用变换
    images = np.concatenate([img, dst_img], axis=0)  # 拼合图像
    cv2.imshow('images show', images)
    cv2.waitKey(0)
