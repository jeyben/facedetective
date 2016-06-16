import numpy as np
import cv2
import pdb

cap = cv2.VideoCapture('mtv-short.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pdb.set_trace()
    cv2.imshow('frame',gray)
    cv2.waitKey(25)
       
    for x in xrange(0,24):
    	cap.grab()

cap.release()
cv2.destroyAllWindows()