import cv2
import sys
import os
import pdb
import shutil


# Get user supplied values
videoPath = sys.argv[1]
cascPath = sys.argv[2]
resultDirPath = "faces_result"

scaleFactorValue=1.1
minNeighborsValue=5
sizeValue=(30, 30)
##############
# face detect function
##############
def getFaces(faceCascade, grayScale):
    faces = faceCascade.detectMultiScale(
        grayScale,
        scaleFactor=scaleFactorValue,
        minNeighbors=minNeighborsValue,
        minSize=sizeValue,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


#prepare result dir
resultPath = "{0}/{1}_{2}_{3}_{4}".format(resultDirPath, videoPath, scaleFactorValue, minNeighborsValue, sizeValue)
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

for the_file in os.listdir(resultPath):
    file_path = os.path.join(resultPath, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

cap = cv2.VideoCapture(videoPath)
frameCounter = 0
matches = 0
targetMatches = 11
imageCounter=0
burstingMode = None
framesInCurrentBurst = 0
defaultSkipFrameCount=24 * 2 

faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while (cap.isOpened() and matches < targetMatches):
    imageCounter += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = getFaces(faceCascade, gray)
    goodFaces = 0
    if len(faces) > 0:
        print "Found {0} faces in frame {1}".format(len(faces), frameCounter)
        if burstingMode:
            framesInCurrentBurst+=1
        else:
            burstingMode=True
            framesInCurrentBurst=0
            matches += 1

        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceCropGray = gray[y:y+h,x:x+w]
            faceCropColor = frame[y:y+h,x:x+w]
            eyes = eyesCascade.detectMultiScale(faceCropGray)
            if(len(eyes) == 2):
                #TODO figure out if the eyes are are next to each other...
                goodFaces += 1
            
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(faceCropColor,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        if(goodFaces > 0):
            cv2.imwrite("{0}/f_{1}.png".format(resultPath, frameCounter), frame)

    else:
        print "No faces in frame {0}".format(frameCounter)
        #if no faces break any burst
        burstingMode=None
        framesInCurrentBurst=0

    skipFramesCount= defaultSkipFrameCount
    #bursting?
    if burstingMode and framesInCurrentBurst<3:
        skipFramesCount=1
    else:
        burstingMode=None
        framesInCurrentBurst = 0

    frameCounter+=skipFramesCount
    for x in xrange(0, skipFramesCount):
        cap.grab()

cap.release()
cv2.destroyAllWindows()