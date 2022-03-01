import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from torch.nn.functional import cosine_similarity as cos
from PIL import Image
from util.util import probreweighting
import matplotlib.image as mpimg
data_dir4= 'E:\google drive/ir_seg_dataset2/labels_boundary/'
data_dir3= 'E:\google drive/ir_seg_dataset2/labels_salient/'
data_dir = 'E:\google drive/ir_seg_dataset/'
data_dir2 = 'E:\google drive/ir_seg_dataset2/visual/'
m1 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
m2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
from matplotlib import pyplot as plt
# 第一步：完成高斯平滑滤波
file_path=os.path.join(data_dir2,'00003N'+'.jpg')
img = cv2.imread(file_path,0)
# plt.imshow(img)
# plt.show()
img = cv2.GaussianBlur(img,(3,3),2)
# cv2.imshow("0",img)

# 第二步：完成一阶有限差分计算，计算每一点的梯度幅值与方向
img1 = np.zeros(img.shape,dtype="uint8") # 与原图大小相同
theta = np.zeros(img.shape,dtype="float")  # 方向矩阵原图像大小
img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)
rows,cols = img.shape
for i in range(1,rows-1):
    for j in range(1,cols-1):
        # Gy
        Gy = (np.dot(np.array([1, 1, 1]), (m1 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
        # Gx
        Gx = (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
        if Gx[0] == 0:
            theta[i-1,j-1] = 90
            continue
        else:
            temp = (np.arctan(Gy[0] / Gx[0]) ) * 180 / np.pi
        if Gx[0]*Gy[0] > 0:
            if Gx[0] > 0:
                theta[i-1,j-1] = np.abs(temp)
            else:
                theta[i-1,j-1] = (np.abs(temp) - 180)
        if Gx[0] * Gy[0] < 0:
            if Gx[0] > 0:
                theta[i-1,j-1] = (-1) * np.abs(temp)
            else:
                theta[i-1,j-1] = 180 - np.abs(temp)
        img1[i-1,j-1] = (np.sqrt(Gx**2 + Gy**2))
for i in range(1,rows - 2):
    for j in range(1, cols - 2):
        if (    ( (theta[i,j] >= -22.5) and (theta[i,j]< 22.5) ) or
                ( (theta[i,j] <= -157.5) and (theta[i,j] >= -180) ) or
                ( (theta[i,j] >= 157.5) and (theta[i,j] < 180) ) ):
            theta[i,j] = 0.0
        elif (    ( (theta[i,j] >= 22.5) and (theta[i,j]< 67.5) ) or
                ( (theta[i,j] <= -112.5) and (theta[i,j] >= -157.5) ) ):
            theta[i,j] = 45.0
        elif (    ( (theta[i,j] >= 67.5) and (theta[i,j]< 112.5) ) or
                ( (theta[i,j] <= -67.5) and (theta[i,j] >= -112.5) ) ):
            theta[i,j] = 90.0
        elif (    ( (theta[i,j] >= 112.5) and (theta[i,j]< 157.5) ) or
                ( (theta[i,j] <= -22.5) and (theta[i,j] >= -67.5) ) ):
            theta[i,j] = -45.0

# 第三步：进行 非极大值抑制计算
img2 = np.zeros(img1.shape) # 非极大值抑制图像矩阵

for i in range(1,img2.shape[0]-1):
    for j in range(1,img2.shape[1]-1):
        if (theta[i,j] == 0.0) and (img1[i,j] == np.max([img1[i,j],img1[i+1,j],img1[i-1,j]]) ):
                img2[i,j] = img1[i,j]

        if (theta[i,j] == -45.0) and img1[i,j] == np.max([img1[i,j],img1[i-1,j-1],img1[i+1,j+1]]):
                img2[i,j] = img1[i,j]

        if (theta[i,j] == 90.0) and  img1[i,j] == np.max([img1[i,j],img1[i,j+1],img1[i,j-1]]):
                img2[i,j] = img1[i,j]

        if (theta[i,j] == 45.0) and img1[i,j] == np.max([img1[i,j],img1[i-1,j+1],img1[i+1,j-1]]):
                img2[i,j] = img1[i,j]

# 第四步：双阈值检测和边缘连接
img3 = np.zeros(img2.shape) #定义双阈值图像
# TL = 0.4*np.max(img2)
# TH = 0.5*np.max(img2)
TL = 0.004
TH = 0.008
#关键在这两个阈值的选择
for i in range(1,img3.shape[0]-1):
    for j in range(1,img3.shape[1]-1):
        if img2[i,j] < TL:
            img3[i,j] = 0
        elif img2[i,j] > TH:
            img3[i,j] = 255
        elif (( img2[i+1,j] < TH) or (img2[i-1,j] < TH )or( img2[i,j+1] < TH )or
                (img2[i,j-1] < TH) or (img2[i-1, j-1] < TH )or ( img2[i-1, j+1] < TH) or
                   ( img2[i+1, j+1] < TH ) or ( img2[i+1, j-1] < TH) ):
            img3[i,j] = 255


cv2.imshow("1",img)  		  # 原始图像
cv2.imshow("2",img1)       # 梯度幅值图
cv2.imshow("3",img2)       #非极大值抑制灰度图
cv2.imshow("4",img3)       # 最终效果图
cv2.imshow("theta",theta) #角度值灰度图

cv2.waitKey(0)
