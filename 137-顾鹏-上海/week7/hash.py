# hash算法
import cv2

def avg_hash(img):
    img_norm = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_norm,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    sum=0
    for i in range(8):
        for j in range(8):
            sum = sum+img_gray[i,j]
    avg_px = sum/64

    for i in range(8):
        for j in range(8):
            if img_gray[i,j]>avg_px:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def dhash(img):
    img_norm = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j]>img_gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str


if __name__=="__main__":
    img = cv2.imread('../lenna.png')
    ret_hasha=avg_hash(img)
    print(ret_hasha)
    ret_hashd =dhash(img)
    print(ret_hashd)