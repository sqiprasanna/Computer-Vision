from scipy import misc,signal 
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Circle
import cv2

#Laplacian of Guassian filter generation
def LaplacianofGuassian(sigma,log):
	for i in range(0,size):
		for j in range(0,size):
			y=int(size/2)-i	
			x=j-int(size/2)
		
			log[i][j]=((x*x)+(y*y)-(2*sigma*sigma))*(math.exp(-((x*x)+(y*y))/(2*sigma*sigma)))

#convolution of each pixel	
def convolution(image,filter1,size):
	sum=0
	for i in range(0,size):
		for j in range(0,size):
			sum=sum+image[i][j]*filter1[i][j]
	
	return sum*sum


#main.........................
print("started")
#Read an image
image=mpimg.imread('butterfly.jpg')	
gray=image[:,:,0]


m,n=gray.shape
sigma=4

print(m,n)


no=10

images_sigma=[]

for i in range(0,no):
	

	size=int(6*sigma)
	#Resizing
	if((size%2)==0):
		size=size-1
	#create a log matrix
	log=np.zeros((size,size))
	#generate log
	LaplacianofGuassian(sigma,log)
	

	#applying log

	s1=int(size/2)
	#print(s1)	
	m1=m+(2*s1)
	n1=n+(2*s1)
	#print(m1,n1)
	im_z=np.zeros((m1,n1))
	im_z1=np.zeros((m1,n1))
	#zero padding
	for i in range(0,m1):
		for j in range(0,n1):
			if(i<s1 or i>=m+s1 or j<s1 or j>=n+s1):
				im_z[i][j]=0
			
			else:
				im_z[i][j]=gray[i-s1][j-s1]
	
	for i in range(s1,m1-s1):
		for j in range(s1,n1-s1):
			
			im_z1[i][j]=sum(np.multiply(im_z[i-s1:i+s1+1,j-s1:j+s1+1],log).sum(1))**2 
	
	#stacking of images
	images_sigma.append(im_z1)
	sigma=sigma*1.25

print("filtering done,convolution")
print("nms")
#Non maxima supression




sigma=4
coordinates_list=[]

for i in range(0,no):
	coordinates_list.append([])

for k in range(0,no):

	size=int(6*sigma)
	#Resizing
	if((size%2)==0):
		size=size-1

	s1=int(size/2)	
	m1=m+(2*s1)
	n1=n+(2*s1)

	for i in range(s1,m1-s1):
		for j in range(s1,n1-s1):
			sample=[]
			for p in range(i-s1,i+s1+1):
				for q in range(j-s1,j+s1+1):
					sample.append(images_sigma[k][p][q])
			a=max(sample)
			
			if(k==0):
				if(images_sigma[k][i][j]==a):
					coordinates_list[k].append([i-s1,j-s1])

			else:
				print(k)
				if(images_sigma[k][i][j]==a):
					for b in range(0,k):
						if(coordinates_list[b].count([i-s1,j-s1])>0):
							coordinates_list[b].remove([i-s1,j-s1])
					coordinates_list[k].append([i-s1,j-s1])

				
	sigma=sigma*1.25

blobs=gray.copy()
for i in range(0,no):
	print(coordinates_list[i])

print("circles")
#create circles
for k in range(0,no):

	for j in range(0,len(coordinates_list[k])):
		xx=coordinates_list[k][j][0]
		yy=coordinates_list[k][j][1]
		cv2.circle(blobs,(yy,xx),int(math.sqrt(2)*(4*(1.25**k))),(0,0,255),thickness=1)

cv2.imwrite('blob2.jpg',blobs)
cv2.imshow('test',blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()