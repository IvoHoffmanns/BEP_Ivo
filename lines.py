# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:07:06 2021

@author: 20183245
"""
import numpy as np

import cv2


image = cv2.imread('images/test.tiff')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect lines in the image
edges = cv2.Canny(gray,60,60,apertureSize = 3)  #ap=3 works best
lines = cv2.HoughLinesP(edges,1,np.pi/180,292,minLineLength=350,maxLineGap=300) 
#lines = np.round(lines[0,:]).astype("int")
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
cv2.imshow("output", np.hstack([image, output]))
#cv2.imwrite('images/minlines.png', np.hstack([image, output]) ) # save image in images folder

#cv2.imshow("edges",edges) #use this to check the edges detected
#cv2.imwrite('images/canny50-50.png', edges ) # save image in images folder
cv2.waitKey(0)
cv2.destroyAllWindows()