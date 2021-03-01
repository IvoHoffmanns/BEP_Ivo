# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:49:21 2021

@author: 20183245
"""

# import the necessary packages
import numpy as np
import cv2
from PIL import Image


# =============================================================================
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
# =============================================================================

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread('images/test.tiff')
rbound=550 #x coord of the section
imageleft=image[:,:rbound]
outputleft = imageleft.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect lines in the image
edges = cv2.Canny(gray,9000,9000,apertureSize = 7)
edgesleft=edges[:,:rbound]

# detect circles in the image
circles = cv2.HoughCircles(edgesleft, cv2.HOUGH_GRADIENT,1,8, param1=600,param2=36,minRadius=20,maxRadius=75) #cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,8, param1=63,param2=26,minRadius=20,maxRadius=75)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(outputleft, (x, y), r, (0, 255, 0), 1)
		cv2.rectangle(outputleft, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        
#cv2.imwrite('houghlines5.jpg',img)
# show the output image
cv2.imshow("output", np.hstack([imageleft, outputleft]))
cv2.imwrite('images/leftsemi.png', np.hstack([imageleft, outputleft]) ) # save image in images folder
#cleft=Circle(Point(x,y),r)
#cv2.imshow("edges",edgesleft) #use this to check the edges detected
#cv2.imwrite('images/canny50-50.png', edges ) # save image in images folder
cv2.waitKey(0)
cv2.destroyAllWindows()