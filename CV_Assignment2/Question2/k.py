from math import floor,sqrt
import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image

def lineofpoints(points):
	lines=[]
	j=0
	for i in range(0,int(len(points)/2)):
		line=np.cross(points[j],points[j+1])
		j=j+2
		lines.append(line/line[2])

	return lines

def intersection_ps(lines):
	intersection_p=[]
	j=0
	for i in range(0,int(len(lines)/2)):
		ip=np.cross(lines[j],lines[j+1])
		j=j+2
		intersection_p.append(ip/ip[2])

	return intersection_p

def tractorheight(signline,signbase,br,vanishingline):
	tractorbase=[5332,2309,1]
	tractortop=[5332,2113,1]

	tbsb=np.cross(signbase,tractorbase)
	tbsb=tbsb/tbsb[2]
	p_vanline=np.cross(tbsb,vanishingline)
	p_vanline=p_vanline/p_vanline[2]

	lineback=np.cross(p_vanline,tractortop)
	lineback=lineback/lineback[2]

	t=np.cross(lineback,signline)
	t=t/t[2]

	bt=sqrt(((signbase[0]-t[0])**2)+((signbase[1]-t[1])**2))

	tractorheight=round((bt/br)*(1.65),2)
	print("Tractor Height:")
	print(tractorheight)

def buildingheight(signline,signbase,br,vanishingline):
	buildingtop=[4438,921,1]
	buildingbase=[4438,1986,1]
	
	bbsb=np.cross(signbase,buildingbase)
	bbsb=bbsb/bbsb[2]
	p_vanline=np.cross(bbsb,vanishingline)
	p_vanline=p_vanline/p_vanline[2]

	lineback=np.cross(p_vanline,buildingtop)
	lineback=lineback/lineback[2]

	t=np.cross(lineback,signline)
	t=t/t[2]

	bt=sqrt(((signbase[0]-t[0])**2)+((signbase[1]-t[1])**2))

	buildingheight=round((bt/br)*(1.65),2)
	print("Building Height:")
	print(buildingheight)

def cameraheight(signbase,signtop):

	p_vanline2=[4547,1577,1]

	stsb=round(sqrt(((signbase[0]-signtop[0])**2)+((signbase[1]-signtop[1])**2)),2)
	sbpv2=round(sqrt(((signbase[0]-p_vanline2[0])**2)+((signbase[1]-p_vanline2[1])**2)),2)


	cameraheight=round((sbpv2/stsb)*1.65,2)
	print("Camera Height:")
	print(cameraheight)

#Read image
image=ndimage.imread("img.jpg",flatten=False)
padding=5000
image_padded_list=[[[0,0,0] for j in range(5500+len(image[0]))] for i in range(len(image))]
img = mpimg.imread('img.jpg')
print(len(img),len(img[0]),img[0][0][0])
print(len(image_padded_list),len(image_padded_list[0]))
rows=len(img)
cols=len(img[0])
a=0
b=0
c=0
for i in range(rows):
	b=0
	for j in range(3500,cols+3500):
		c=0
		for k in range(3):
			image_padded_list[i][j][k]=img[a][b][c]
			c=c+1
		b=b+1
	a=a+1
	
image_padded=np.array(image_padded_list)
print(image_padded.shape)

misc.imsave('image_padded.jpg',image_padded)

imgplot=plt.imshow(img)
plt.show()

padded_i=mpimg.imread('image_padded.jpg')
imgplot=plt.imshow(padded_i)
plt.show()

points=[[3686,1048,1],[3890,1018,1],[3837,874,1],[3981,845,1],[3528,2438,1],[3762,2392,1],[3543,2528,1],[3747,2483,1]]

#Finding lines for the selected points
lines=lineofpoints(points)

#Finding intersection of parallel lines
intersection_p=intersection_ps(lines)

#Finding vnishing line
vanishingline=np.cross(intersection_p[0],intersection_p[1])
vanishingline=vanishingline/vanishingline[2]

signbase=[4547,2181,1]
signtop=[4547,2068,1]
signline=np.cross(signbase,signtop)
signline=signline/signline[2]
br=sqrt(((signbase[0]-signtop[0])**2)+((signbase[1]-signtop[1])**2))
tractorheight(signline,signbase,br,vanishingline)
buildingheight(signline,signbase,br,vanishingline)
cameraheight(signbase,signtop)

img2=cv2.line(padded_i,(3686,1048),(int(intersection_p[0][0]),int(intersection_p[0][1])),(0,0,255),10)
img2=cv2.line(padded_i,(3837,874),(int(intersection_p[0][0]),int(intersection_p[0][1])),(0,0,255),10)
img2=cv2.line(padded_i,(3528,2438),(int(intersection_p[1][0]),int(intersection_p[1][1])),(0,0,255),10)
img2=cv2.line(padded_i,(3543,2528),(int(intersection_p[1][0]),int(intersection_p[1][1])),(0,0,255),10)
img2=cv2.line(padded_i,(int(intersection_p[0][0]),int(intersection_p[0][1])),(int(intersection_p[1][0]),int(intersection_p[1][1])),(0,255,0),10)

imgplot1=plt.imshow(img2)
plt.show()