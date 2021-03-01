# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:49:21 2021

@author: 20183245
"""

# import the necessary packages
import numpy as np
import cv2
from PIL import Image		#not currently used
from sympy import *		#To calculate intersections 
from sympy.geometry import *


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread('images/test.tiff')
lbound=475 			#x coord of the section
imageright=image[:,lbound:]
outputright = imageright.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect lines in the image
edges = cv2.Canny(gray,6000,6000,apertureSize = 7)	#ap=7 shows the sharper gradients better
edgesright=edges[:,lbound:]

# detect circles in the image
circles = cv2.HoughCircles(edgesright, cv2.HOUGH_GRADIENT,1,8, param1=100,param2=49,minRadius=20,maxRadius=75) #cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,8, param1=63,param2=26,minRadius=20,maxRadius=75)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(outputright, (x, y), r, (0, 255, 0), 1)
		cv2.rectangle(outputright, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        
# show the output image
cv2.imshow("output", np.hstack([imageright, outputright]))	#show og image and image with drawings next to eachother
#cv2.imwrite('images/rightsemi.png', np.hstack([imageright, outputright]) ) # save image in images folder
cright=Circle(Point(x,y),r)					#define the circle using the parameters found by HoughCircles
#cv2.imshow("edges",edges[:,lbound:]) 				#use this to check the edges detected
cv2.waitKey(0)
cv2.destroyAllWindows()						#When running, window with image pops up, closes when any key is pressed
