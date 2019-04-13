import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc

image1 = cv2.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/einstein.bmp',0)
image2 = cv2.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/marilyn.bmp',0)


def gaussian_filter(rows,columns,sigma):
    if rows%2 == 0:
        row_center = rows/2
    else:
        row_center = rows/2 + 1

    if columns%2 == 0:
        col_center = columns/2
    else:
        col_center = columns/2 + 1
    def gaussian(i,j):
        return math.exp(-1.0 * ((i - row_center)**2 + (j - col_center)**2) / (2 * sigma**2))
    gaussian_array = np.zeros((rows,columns))
    for i in range(0,rows):
        for j in range(0,columns):
            gaussian_array[i][j] =  gaussian(i,j)
    return gaussian_array    

def Apply_Fitler(image,flag,sigma):
    n,m = image.shape
    if flag == True:                # High pass
        gaussian_matrix = (1 - gaussian_filter(n,m,sigma))
    else:                           #Low pass
        gaussian_matrix = gaussian_filter(n,m,sigma)
    dft = np.fft.fft2(image)
    dftshift = np.fft.fftshift(dft)
    filterImage = dftshift * gaussian_matrix
    ifftshift = np.fft.ifftshift(filterImage)
    ifftImage = np.fft.ifft2(ifftshift)
    plt.imshow(np.real(ifftImage),cmap = 'gray')
    if flag == True:
        plt.title('High Pass Filter'),plt.xticks([]),plt.yticks([])
    else:
        plt.title('Low Pass Filter'),plt.xticks([]),plt.yticks([])
    plt.show()
    return np.real(ifftImage)

def hybrid_image(highpassImg,lowpassImg,alpha,beta):
    lowpass = Apply_Fitler(lowpassImg,False,beta)
    misc.imsave("result/Marilyn-filtered.png",np.real(lowpass))
    highpass = Apply_Fitler(highpassImg,True, alpha)
    misc.imsave("result/einstein-filtered.png",np.real(highpass))
    return np.real(highpass + lowpass)

alpha = 30
beta = 15
hybrid_img = np.real(hybrid_image(image1,image2,alpha,beta))
plt.imshow(hybrid_img,cmap = 'gray')
plt.title('Hybrid result'),plt.xticks([]),plt.yticks([])
plt.show() 
misc.imsave("result/einstein-marilyn.png",np.real(hybrid_img))