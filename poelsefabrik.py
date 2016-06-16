import cv2
import sys
import os
import pdb
import shutil


# Get user supplied values
videoPath = sys.argv[1]
cascPath = sys.argv[2]
resultDirPath = "faces_result"

scaleFactorValue=1.3
minNeighborsValue=5
sizeValue=(100, 100)
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

faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while (cap.isOpened() and frameCounter < 100):
    frameCounter += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = getFaces(faceCascade, gray)

    if len(faces) > 0:
        print "Found {0} faces in frame".format(len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceCropGray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            eyes = eyesCascade.detectMultiScale(faceCropGray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imwrite("{0}/f_{1}.png".format(resultPath, frameCounter), frame)
    else:
        print "No faces in frame {0}".format(frameCounter)
    for x in xrange(0, 24):
        cap.grab()

cap.release()
cv2.destroyAllWindows()