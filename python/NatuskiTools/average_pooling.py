import numpy as  np
import cv2
import glob

# 读图
img_path=glob.glob(r"../Opencv/images/*.jpg")
# img_path=r"../Opencv/images/4.jpg"
grid_size=(8,8)

for i in range(len(img_path)):
    img=cv2.imread(img_path[i])
    # 平均池化
    h,w,c=img.shape
    # 设计一个ROI
    # h_upper=int(h/3.5)
    # h_lower=int(h/1.5)
    # h1=h_lower-h_upper
    # w_upper=int(w/3.5)
    # w_lower=int(w/1.5)

    g_w,g_h=grid_size
    # 算出需要多少个网格来覆盖图片
    h_nums=int(h/g_h)
    w_nums=int(w/g_w)
    # 图片是3d的，对每一层做池化
    for i in range(h_nums):
        for j in range(w_nums):
            for k in range(c):
                img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]=np.mean(img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]).astype(np.uint8)
                # img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]=np.max(img[i*g_h:(i+1)*g_h,j*g_w:(j+1)*g_w,k]).astype(np.uint8)
                # img[h_upper:(i+1)*g_h,j*g_w:(j+1)*g_w,k]=np.max(img[h_upper:(i+1)*g_h,j*g_w:(j+1)*g_w,k]).astype(np.uint8)

    # 图片的w,h
    # 缺陷：不能整除g_w,g_h的情况下处理?
    cv2.imshow('pic',img)
    cv2.waitKey(1500)