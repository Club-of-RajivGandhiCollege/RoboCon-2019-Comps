#code for turn
import cv2 as cv
import numpy as np
from time import sleep


def contour(img):
    #boundries_black = [([211, 237, 235], [255, 255, 255])]	#boundries of color detected white

    boundries_black = [([0, 0, 0], [50, 50, 50])]

    #for (lower, upper) in boundries_white:
    for (lower, upper) in boundries_black:
		# create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

		# find the colors within the specified boundaries and apply
		# the mask
        mask = cv.inRange(img, lower, upper)
        output = cv.bitwise_and(img, img, mask = mask)

    #Convert to grayscale
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

	# Gaussian blur
    blur = cv.GaussianBlur(gray,(5,5),0)
	
	# Color thresholding
    ret, thresh = cv.threshold(blur,60,255,cv.THRESH_BINARY)  #60


    im2, contours, n=cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    print(img[100][100])

    if(len(contours)>0):
        return 1, thresh
    else:
        return 0, thresh


cap = cv.VideoCapture(0)

while True:

    ret,frame=cap.read()

    totalCrop=frame[0:720 ,0:600]

    #cv.imshow('frame',totalCrop)
    imgf1=0
    imgf2=0
    imgf3=0
    imgf4=0
    imgf5=0
    imgf6=0
    imgf7=0
    imgf8=0
    imgf9=0


    grid1=totalCrop[0:150,0:200]
    grid4=totalCrop[150:300,0:200]
    grid7=totalCrop[300:450,0:200]
    grid2=totalCrop[0:150,200:400]
    grid5=totalCrop[150:300,200:400]
    grid8=totalCrop[300:450,200:400]
    grid3=totalCrop[0:150,400:600]
    grid6=totalCrop[150:300,400:600]
    grid9=totalCrop[300:450,400:600]
 
    
    cv.imshow('grid1',grid1)
    cv.imshow('grid2',grid2)
    cv.imshow('grid3',grid3)
    cv.imshow('grid4',grid4)
    cv.imshow('grid5',grid5)
    cv.imshow('grid6',grid6)
    cv.imshow('grid7',grid7)
    cv.imshow('grid8',grid8)
    cv.imshow('grid9',grid9)
    

    imgf1,h1 = contour(grid1)
    imgf2,h2 = contour(grid2)
    imgf3,h3 = contour(grid3)
    imgf4,h4 = contour(grid4)
    imgf5,h5 = contour(grid5)
    imgf6,h6 = contour(grid6)
    imgf7,h7 = contour(grid7)
    imgf8,h8 = contour(grid8)
    imgf9,h9 = contour(grid9)
    
    cv.imshow('h1', h1)
    cv.imshow('h2', h2)
    cv.imshow('h3', h3)
    cv.imshow('h4', h4)
    cv.imshow('h5', h5)
    cv.imshow('h6', h6)
    cv.imshow('h7', h7)
    cv.imshow('h8', h8)
    cv.imshow('h9', h9)

    #cv.imshow('grid5',grid5)
    #cv.imshow('thresh5',h5)

#Conditions for Black and white Background
    if imgf8:
        if (imgf2 and imgf4 and imgf5 and imgf6) :
            print("cross")
        elif imgf2 and imgf5:
            print("straight")
        elif imgf1:
            print("left")
        elif imgf3:
            print('right')
    
    elif imgf1 or imgf4 or imgf7:
        print("Left Balance")
    elif imgf3 or imgf6 or imgf9:
        print("Right Balance")
    elif ((not imgf1) and (not imgf2) and (not imgf3) and (not imgf4) and (not imgf5) and (not imgf6) and (not imgf7) and (not imgf8) and (not imgf9)):
        print("There ain't no line bitch")

    
    if cv.waitKey(1) == 27:
        break

cap.release()   
cv.destroyAllWindows()
