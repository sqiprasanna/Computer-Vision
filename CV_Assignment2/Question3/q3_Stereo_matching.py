import numpy as np
import cv2
import time
from scipy import signal
import math
def Templte_Matching(half_window_size, weight, d_range, y, left_image, right_image, height, offset_adjust):
    for x in range(half_window_size, weight - half_window_size):
            best_offset = 0
            prev = 2345233234
            ssds = []
            numBlocks = 0
            for i in range(d_range):       
                #print(i)        
                left = left_image[y - half_window_size : y+ half_window_size , x - half_window_size : x+ half_window_size]
                right = right_image[y - half_window_size : y+ half_window_size  , x - half_window_size - i : x+ half_window_size - i]
                if(right.shape == left.shape):
                    numBlocks += 1
                    temp = left - right
                    temp = temp ** 2
                    Sum_squared_diff = np.sum(temp)
                    ssds.append(Sum_squared_diff)
                    if Sum_squared_diff < prev:
                        prev = Sum_squared_diff
                        best_offset = i
                else:
                    right = np.zeros(left.shape)
            depth[y, x] = best_offset * offset_adjust
    return depth

def Sum_squared_diff(left_image, right_image, height, weight, half_window_size, d_range,offset_adjust):
    for y in range(half_window_size, height - half_window_size):    
        depth = Templte_Matching(half_window_size, weight, d_range, y, left_image, right_image, height, offset_adjust)           
    return depth

def Template_matching_corr(left_image, right_image, height, weight, half_window_size, d_range, y, offset_adjust):
        for x in range(half_window_size, weight - half_window_size):
                best_offset = 0
                prev_corr = 0
                corrs = []
                numBlocks = 0
                for k in range(d_range):
                    temp = 0              
                    NCC = 0 
                    left = left_image[y - half_window_size : y+ half_window_size , x - half_window_size : x+ half_window_size]
                    right = right_image[y - half_window_size : y+ half_window_size  , x - half_window_size - k : x+ half_window_size - k]
                    for i in range(-half_window_size,half_window_size):
                        for j in range(-half_window_size,half_window_size):
                            #print("hello")
                            temp = (int(left_image[y+i,x+j]) - np.mean(left))*(int(right_image[y+i,x+j - k]) - np.mean(right))
                            temp = temp/(np.std(left) * np.std(right))
                            NCC += temp
                    NCC /= (2*half_window_size + 1)
                    corrs.append(NCC)
                    if NCC > prev_corr:
                        prev_corr = NCC
                        best_offset = k
                
                depth[y, x] = best_offset * offset_adjust
        return depth
def Correlation(left_image, right_image, height, weight, half_window_size, d_range):
    offset_adjust = 255 / d_range
    for y in range(half_window_size, height - half_window_size):    
        if (y%10 == 0):
            print("Image row  " + str(y) +"/" +str(height) +" ( " + str(int(y * 100/height)) + "% )\n")   
        depth = Template_matching_corr(left_image, right_image, height, weight, half_window_size, d_range, y, offset_adjust)   
    return depth
left_image = cv2.imread("tsukuba1.ppm",0)
right_image = cv2.imread("tsukuba2.ppm",0)
height, weight = left_image.shape   
depth = np.zeros((height, weight), np.uint8)
depth_corr = np.zeros((height, weight), np.uint8)

d_range = 30
half_window_size = 3
blockSize = 2 * half_window_size + 1 
offset_adjust = 255 / d_range   
SSD_time = time.time()
depth = Sum_squared_diff(left_image, right_image, height, weight, half_window_size, d_range,offset_adjust)
cv2.imwrite("SSD_with_"+str(blockSize)+"_"+str(d_range)+"_ws_.png",depth)
time_SSD = time.time() - SSD_time
Corr_time = time.time()
depth_corr = Correlation(left_image, right_image, height, weight, half_window_size, d_range)
cv2.imwrite("Corr_with_"+str(blockSize)+"_"+str(d_range)+"_ws.png",depth)
time_Corr = time.time() - Corr_time

print("Time taken for Sum_squared_Difference(SSD) :- " + str(time_SSD))
print("Time taken for Correlation is " + str(time_Corr))
