import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.image as mpimg

img1 = plt.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/bird.bmp')
img2 = plt.imread('E:/STUDY/ACADEMIC/6th sem/CV/assign1/HW1_Q1/plane.bmp')

img1r = img1[:,:,0]
img1g = img1[:,:,1]
img1b = img1[:,:,2]

img2r = img2[:,:,0]
img2g = img2[:,:,1]
img2b = img2[:,:,2]

alpha = 50
beta = 10
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
    # plt.imshow(np.real(dftshift),cmap = 'gray')
    # plt.show()
    filterImage = dftshift * gaussian_matrix
    plt.imshow(np.real(filterImage))
    return filterImage

img1rFT = Apply_Fitler(img1r,True,alpha)
img1gFT = Apply_Fitler(img1g,True,alpha)
img1bFT = Apply_Fitler(img1b,True,alpha)

img2rFT = Apply_Fitler(img2r,False,beta)
img2gFT = Apply_Fitler(img2g,False,beta)
img2bFT = Apply_Fitler(img2b,False,beta)

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
#plt.title('Hybrid Image with alpha =  and beta ='),plt.xticks([]),plt.yticks([])
plt.show()
#misc.imsave("result/einstein-marilyn.png",np.real(img3))