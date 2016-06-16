import cv2
import sys
import os
import pdb
import shutil


# Get user supplied values
videoPath = sys.argv[1]
cascPath = sys.argv[2]
resultPath = "faces_result"

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

cap = cv2.VideoCapture(videoPath)
frameCounter = 0

while (cap.isOpened() and frameCounter < 100):
    frameCounter += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = getFaces(faceCascade, gray)

    if len(faces) > 0:
        print "Found {0} faces in frame".format(len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("faces_result/f_" + frameCounter + ".png", frame)
    else:
        print "No faces in frame {0}".format(frameCounter)
    for x in xrange(0, 24):
        cap.grab()

cap.release()
cv2.destroyAllWindows()