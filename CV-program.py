import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
image_black = np.zeros((256,256))
image_l = np.ones((256,256,3))*255
cv.imshow("black",image_black)
cv.waitKey(0)

#IMAGE IMPORTING
my_lap = cv.imread(r"D:\Movies\IMG_20220918_192736.jpg")
cv.imshow("lap",my_lap)
cv.waitKey(0)

#IMAGE RESIZING
dim=(1500,800)
resize_lap = cv.resize(my_lap,dim, interpolation=cv.INTER_AREA)
cv.imshow("MY_LAP",resize_lap)
cv.waitKey(0)

#IMAGE CROPING
crop_lap=resize_lap[350:1086, 300:1200]
cv.imshow("MY_LAP", crop_lap)
cv.waitKey(0)
cv.imwrite("crop_lap.png",crop_lap)

#IMAGE BLUR
blur_lap = cv.blur(crop_lap,(6,6))
cv.imshow("MY_LAP",blur_lap)
cv.waitKey(0)

#IMAGE THRESHOLDING
lap_img = cv.imread("crop_lap.png",0)
ret,th_lap = cv.threshold(lap_img,90,160,cv.THRESH_BINARY)
cv.imshow("lap_threshold",th_lap)
cv.waitKey(0)

#AVG and gaussian thersholding
th2 = cv.adaptiveThreshold(lap_img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(lap_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imshow("LAP",th3)
cv.waitKey(0)

#Image Gradients
laplacian = cv.Laplacian(crop_lap,cv.CV_64F)
sobelx = cv.Sobel(crop_lap,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(crop_lap,cv.CV_64F,0,1,ksize=5)
cv.imshow("Lap-1",sobely)
cv.waitKey(0)

#Canny edge detection in OpenCV
edges=cv.Canny(crop_lap,50,150)
cv.imshow("lap",edges)
cv.waitKey(0)
#PLT
edge_75 = cv.Canny(crop_lap,75,100)
edge_50 = cv.Canny(crop_lap,50,100)
plt.imshow(edge_50,cmap='gray')
cv.imshow("lap- 1",edge_50)
cv.imshow("lap - 2",edge_75)
cv.waitKey(0)

#Morphological Transformations
#1)Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(crop_lap,kernel,iterations = 1)
cv.imshow("lap",erosion)
cv.waitKey(0)
# Contours
imges = cv.imread("crop_lap.png")
imgray = cv.cvtColor(imges,cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(imgray, 11, 17, 17)
edged = cv.Canny(gray, 20, 90)
cv.imshow("lap", edged)
cv.waitKey(0)
contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, 
                                             cv.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv.boundingRect(contours[0])

for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    x,y,w,h = cv.boundingRect(contours[i])
    imgrect = cv.rectangle(imges,(x,y),(x+w,y+h),(0,255,0),2)
    outfile = ('%s.jpg' % i)
    cv.imwrite(outfile, imgrect)
cv.imshow("lap_plot", imgrect)
cv.waitKey(0)
cv.destroyAllWindows()
#to draw circles 
(x1,y1),rad = cv.minEnclosingCircle(contours[70])
cen = (int(x1),int(y1))
radi = int(rad)
imgcir = cv.circle(crop_lap,cen,radi,(0,255,0),2)
cv.imshow("Lap", imgcir)
cv.waitKey(0)  
# to draw a lot of circles 
for i in range(20):
    (xc,yc),radius = cv.minEnclosingCircle(contours[i])
    center = (int(xc),int(yc))
    radius = int(radius)
    imgcircle = cv.circle(crop_lap,center,radius,(0,255,0),2)
cv.imshow("Lap", imgcircle)
cv.waitKey(0)    
#histogram 
from matplotlib import pyplot as plt
kingimg = cv.imread("crop_lap.png")
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([kingimg],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# Harris Corner Detector
colimage=cv.imread("crop_lap.png")
graytall = cv.cvtColor(colimage,cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(graytall,2,3,0.04)
dst = cv.dilate(dst,None)
max=0.001*dst.max()
colimage[dst>max]=[0,0,255]
cv.imshow('dst',colimage)
cv.waitKey(0)
#to this 6th exp
#Brute-Force matcher 
img1 = cv.imread(r"D:\Movies\IMG_20220918_192736.jpg",0)          # queryImage
img2 = cv.imread("crop_lap.png",0) # trainImage







