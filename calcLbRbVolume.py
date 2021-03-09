# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:09:39 2021

@author: 20183245
"""
# import the necessary packages
import numpy as np
import cv2
from PIL import Image
from sympy import *
from sympy.geometry import *

image = cv2.imread('images/test.tiff')
lbound=475 #x coord of the section
rbound=550 #x coord of the section
imageright=image[:,lbound:]
outputright = imageright.copy()
output= image.copy()
imageleft=image[:,:rbound]
outputleft = imageleft.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Canny edges
edges1 = cv2.Canny(gray,6000,6000,apertureSize = 7)
edgesright=edges1[:,lbound:]
edges2 = cv2.Canny(gray,9000,9000,apertureSize = 7)
edgesleft=edges2[:,:rbound]

# Find right circle
circlesright = cv2.HoughCircles(edgesright, cv2.HOUGH_GRADIENT,1,8, param1=100,param2=49,minRadius=20,maxRadius=75) #cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,8, param1=63,param2=26,minRadius=20,maxRadius=75)
# ensure at least some circles were found
if circlesright is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circlesright = np.round(circlesright[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (xr, yr, rr) in circlesright:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(outputright, (xr, yr), rr, (0, 255, 0), 1)
		cv2.rectangle(outputright, (xr - 2, yr - 2), (xr + 2, yr + 2), (0, 128, 255), -1)
cright=Circle(Point(xr+lbound,yr),rr) #right side of the particle

#Find left circle
circlesleft = cv2.HoughCircles(edgesleft, cv2.HOUGH_GRADIENT,1,8, param1=600,param2=36,minRadius=20,maxRadius=75) #cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,8, param1=63,param2=26,minRadius=20,maxRadius=75)
# ensure at least some circles were found
if circlesleft is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circlesleft = np.round(circlesleft[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (xl, yl, rl) in circlesleft:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(outputleft, (xl, yl), rl, (0, 255, 0), 1)
		cv2.rectangle(outputleft, (xl - 2, yl - 2), (xl + 2, yl + 2), (0, 128, 255), -1)
        
        
cleft=Circle(Point(xl,yl),rl)   #left side of the particle

# Find the walls
edges3 = cv2.Canny(gray,60,60,apertureSize = 3)  #ap=3 works best
lines = cv2.HoughLinesP(edges3,1,np.pi/180,292,minLineLength=350,maxLineGap=300) 
cv2.line(outputleft, (lines[2,0,0],lines[2,0,1]), (lines[2,0,2],lines[2,0,3]), (0,255,0), 1)    #check which lines are the correct ones
cv2.line(outputleft, (lines[3,0,0],lines[3,0,1]), (lines[3,0,2],lines[3,0,3]), (0,255,0), 1)
cv2.line(outputright, (lines[2,0,0],lines[2,0,1]), (lines[2,0,2],lines[2,0,3]), (0,255,0), 1)
cv2.line(outputright, (lines[3,0,0],lines[3,0,1]), (lines[3,0,2],lines[3,0,3]), (0,255,0), 1)
#cv2.line(output, (lines[2,0,0],lines[2,0,1]), (lines[2,0,2],lines[2,0,3]), (0,255,0), 1)
#cv2.line(output, (lines[3,0,0],lines[3,0,1]), (lines[3,0,2],lines[3,0,3]), (0,255,0), 1)

# Defining the correct lines as capillary walls
topwall=Line(Point(lines[2,0,0],lines[2,0,1]),Point(lines[2,0,2],lines[2,0,3]))
bottomwall=Line(Point(lines[3,0,0],lines[3,0,1]),Point(lines[3,0,2],lines[3,0,3]))
# Calculating all intersection points
[S1a,S1b]=intersection(topwall,cleft)       #first one is correct
[S2a,S2b]=intersection(topwall,cright)      #second one is correct  
[S3a,S3b]=intersection(bottomwall,cleft)    #first one is correct
[S4a,S4b]=intersection(bottomwall,cright)   #second one is correct
#Lbands
Lbandtop=float(sqrt((S1a[0]-S2b[0])**2+(S1a[1]-S2b[1])**2))
Lbandbottom=float(sqrt((S3a[0]-S4b[0])**2+(S3a[1]-S4b[1])**2))
Lband=int((Lbandtop+Lbandbottom)/2) #average of top and bottom contact length
cv2.line(output, (int(S1a[0]),int(S1a[1])), (int(S2b[0]),int(S2b[1])),(0,255,0),1) #draw top lband
cv2.line(output, (int(S3a[0]),int(S3a[1])), (int(S4b[0]),int(S4b[1])),(0,255,0),1) #draw bottom lband
#cv2.line(output, (int((S1a[0]+S2b[0])*0.5),int((S1a[1]+S2b[1])*0.5)), (int((S3a[0]+S4b[0])*0.5),int((S3a[1]+S4b[1])*0.5)),(0,255,0),1) #line trhough middle of the lbands
## Calculating points for spherical cap
Xmidright=int(0.5*S2b[0]+0.5*S4b[0])    #Xcoord of the right point on the middle horizontal line through the particle
Ymidright=int(0.5*S2b[1]+0.5*S4b[1])
Xmidleft=int(0.5*S1a[0]+0.5*S3a[0])     #Xcoord of the left point on the middle horizontal line though the particle
Ymidleft=int(0.5*S1a[1]+0.5*S3a[1])
# Create points from the x/y coords calculated above
p_midright=[Xmidright,Ymidright]    #Midpoint between wall intersections with right circle
p_midleft=[Xmidleft,Ymidleft]       #Midpoint between wall intersections with left circle
slope=(Ymidright-Ymidleft)/(Xmidright-Xmidleft) #slope of the midline, used to extend the midline line in the next part
# =============================================================================
# #Extended midline
cv2.line(output,(int(Xmidleft)-100,int(Ymidleft-100*slope)),(int(Xmidright)+100,int(Ymidright+100*slope)),(0,255,0),1) #draw extended midline in output
# cv2.circle(output, (xl, yl), rl, (0, 255, 0), 1) #draw left circle in output image
# cv2.circle(output, (xr+lbound, yr), rr, (255, 0, 0), 1) #draw right circle in output image
# =============================================================================
cv2.line(output,(int(S1a[0]),int(S1a[1])),(int(S3a[0]),int(S3a[1])),(0,255,0),1)    #draw the rbands
cv2.line(output,(int(S2b[0]),int(S2b[1])),(int(S4b[0]),int(S4b[1])),(0,255,0),1)    #draw the rbands
midline=Line(Point(Xmidleft-100,Ymidleft),Point(Xmidright+100,Ymidright))   #define the midline for sympy
# rband
rright1=float(midline.distance(S2b))    # top part of right rband
rright2=float(midline.distance(S4b))    # bottom part of right rband
rleft1=float(midline.distance(S1a))     # top part of left rband
rleft2=float(midline.distance(S3a))     # bottom part of left rband
Rband=int(((rright1+rright2)/2+((rleft1+rleft2)/2))/2) #average of all 4 rbands
# Point for calculating volume
[Smidleft1,Smidright1]=intersection(cleft,midline)  #intersections between midline and left side of particle
p_h1=[int(Smidleft1[0]),int(Smidleft1[1])]      # most left intersection point
h1=float(sqrt((Xmidleft-p_h1[0])**2+(Ymidleft-p_h1[1])**2)) #h1 parameter for spherical cap
[Smidleft2,Smidright2]=intersection(cright,midline) #intersections between midline and right side of particle
p_h2=[int(Smidright2[0]),int(Smidright2[1])]    # most right intersection point
h2=float(sqrt((Xmidright-p_h2[0])**2+(Ymidright-p_h2[1])**2)) #h2 parameter for spherical cap
# Volume sperical cap
V_sc1=(np.pi*h1**2)/3*(3*rl-h1)
V_sc2=(np.pi*h2**2)/3*(3*rr-h2)
## Calculating volume of fustrum
h3=Xmidright-Xmidleft
r1=(rright1+rright2)/2
r2=(rleft1+rleft2)/2
V_cf=np.pi*h3/3*(r1**2+r1*r2+r2**2)
#final volume
V_Particle=int(V_sc1+V_cf+V_sc2)

## add text to image
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(output,'Lband='+"{}".format(Lband),
    bottomLeftCornerOfText,
    font, 
    fontScale,
    fontColor,
    lineType)
bottomLeftCornerOfText = (10,430)
cv2.putText(output,'Rband='+"{}".format(Rband),
    bottomLeftCornerOfText,
    font, 
    fontScale,
    fontColor,
    lineType)
bottomLeftCornerOfText = (10,460)
cv2.putText(output,'Volume='+"{}".format(V_Particle),
    bottomLeftCornerOfText,
    font, 
    fontScale,
    fontColor,
    lineType)
# Checking figures
cv2.imshow("output", np.hstack([imageright, outputright]))  # show right circle
cv2.imshow("output", np.hstack([imageleft, outputleft]))    # show left circle
cv2.imshow("output", output)            # show wall lines
#cv2.imwrite('images/im_text.png', output ) # save image in images folder
cv2.waitKey(0)
cv2.destroyAllWindows()
