import cv2
import sys
import os
import shutil
import numpy as np


# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
resultPath = "faces_result"

maxFaceHeightPx = 0.7 * 480
minFaceHeightPx = 0.3 * 480

##############
# face detect function
##############
def getFaces(faceCascade, grayScale):
    faces = faceCascade.detectMultiScale(
        grayScale,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces

def isGoodFaceMatch(faces):
    goodMatches = list() 
    for face in enumerate(faces):
        if(face[3] < maxFaceHeightPx && face[3] > minFaceHeightPx):
            goodMatches.append(face)

    return faces
    

#prepare result dir
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

for the_file in os.listdir(resultPath):
    file_path = os.path.join(resultPath, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


#loop through input files
#for imagePath, dirs, filenames in os.walk(imagePath):
#    print(imagePath)
#    for f in filenames:
#        filePath = imagePath + "/" + f
#        image = cv2.imread(filePath)
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#        # Detect faces in the image
#        faceCascade = cv2.CascadeClassifier(cascPath)
#        faces = getFaces(faceCascade, gray)
#
#        if len(faces) > 0:
#            print "Found {0} faces in {1}".format(len(faces), filePath)
#            for (x, y, w, h) in faces:
#                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#            cv2.imwrite("faces_result/f_" + f, image)

