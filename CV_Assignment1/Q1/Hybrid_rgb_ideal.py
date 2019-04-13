import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


img1 = plt.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/bird.bmp')
img2 = plt.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/plane.bmp')

img1r = img1[:,:,0]
img1g = img1[:,:,1]
img1b = img1[:,:,2]

img2r = img2[:,:,0]
img2g = img2[:,:,1]
img2b = img2[:,:,2]

alpha = 10000.0
beta = 25000000.0

def Apply_filter(img,alpha,flag):
    imgFT = np.fft.fft2(img)
    if flag == 0:                               #Low Pass Filter
        imgFT[abs(imgFT)>alpha] = 0
    else:                                       #High Pass Filter
        imgFT[abs(imgFT)<alpha] = 0
    return imgFT



img1rFT = Apply_filter(img1r,alpha,1)
img1gFT = Apply_filter(img1g,alpha,1)
img1bFT = Apply_filter(img1b,alpha,1)

img2rFT = Apply_filter(img2r,beta,0)
img2gFT = Apply_filter(img2g,beta,0)
img2bFT = Apply_filter(img2b,beta,0)

img3rFT = np.zeros(img1rFT.shape,dtype=complex)
img3gFT = np.zeros(img1gFT.shape,dtype=complex)
img3bFT = np.zeros(img1bFT.shape,dtype=complex)

img3rFT = img1rFT + img2rFT
img3gFT = img1gFT + img2gFT
img3bFT = img1bFT + img2bFT


img3r = abs(np.fft.ifft2(img3rFT))
img3g = abs(np.fft.ifft2(img3gFT))
img3b = abs(np.fft.ifft2(img3bFT))

img3=[]
for i in range(img3r.shape[0]):
    img3.append([])
    for j in range(img3r.shape[1]):
        img3[i].append([img3r[i][j],img3g[i][j],img3b[i][j]])

img3=np.array(img3)
img3.shape

plt.imshow(np.absolute(img3) / np.max(np.absolute(img3)))
plt.show()