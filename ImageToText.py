import sys

import numpy as np
import cv2

im = cv2.imread('sampleLetters.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)          # makes the picture black and white
blur = cv2.GaussianBlur(gray,(5,5),0)               # blurs the image out
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)   # makes it black and white

#################      Now finding Contours         ###################

#gets all the contours
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
samples =  np.empty((0,100))                # creates an uninitialized array of 100 elements
responses = []                              # creates an empty responses list
keys = [i for i in range(48,58)]            # creates a keys list

for cnt in contours:                        #
    if cv2.contourArea(cnt)>50:             # if the area of the contour is greater then 50px, only then does it count it as text
        [x,y,w,h] = cv2.boundingRect(cnt)   # finds the bounding rectangle of the contour
        if  h>28:                           # if the height of the bouding rectangle is to small then it ignores it
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)   # creates a rectangle that shows the bounding rect
            roi = thresh[y:y+h,x:x+w]                       # it takes the points that are inside the contour and stores them in roi
            roismall = cv2.resize(roi,(10,10))              # resizes the image to (10,10)???????????
            cv2.imshow('norm',im)                           # displays the image 'img' in a window called norm
            key = cv2.waitKey(0)                            # waits for a key

            if key == 27:  # (escape to quit)               # if you press escape it quits the screen
                sys.exit()                                  # stops the program if you press the escape button
            elif key in keys:                               # if you press any other key thats not the escape key
                responses.append(int(chr(key)))             # inserts the key into the responses
                sample = roismall.reshape((1,100))          # reshapes the sample to be (1,100)??????????
                samples = np.append(samples,sample,0)       # add the sample into the samples

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)