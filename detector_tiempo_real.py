# -*- coding: utf-8 -*-
#-----------------FUENTE-----------------
#https://github.com/techycs18/age-detection-python-opencv
#-----------------------------------------------
# import the necessary packages

import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
import time
import cv2
import os


def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
    #minConf = confianza minima (estimacion)
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    # initialize our results list
    results = []

    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))

    # passing the blob through faceNet
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence of each detecion
        confidence = detections[0, 0, i, 2]

        # get the detection with valued > min confidence value
        if confidence > minConf:
            # get the bbox for that detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # get the roi
            face = frame[startY:endY, startX:endX]

            # ensure the roi in sufficiently large
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0, size=(227, 227),
                                             mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # now make the age prediction and get the blob with largest probability
            ageNet.setInput(faceBlob)
            pred = ageNet.forward()
            i = pred[0].argmax()  # returns the bucket index with max prob
            age = AGE_BUCKETS[i]
            ageConfidence = pred[0][i]

            # construct a dictionary consisting of both the face bounding box location along with the age prediction,
            # then update our results list
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    return results

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# The model architecture
# download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_PROTO = 'deploy_age.prototxt'
# The model pre-trained weights
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_MODEL = 'age_net.caffemodel'
# download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "deploy.prototxt"
# download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"



# Generate the cnn for face recognition
faceNet = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


# Generate the cnn for age recognition
ageNet = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# initialise  the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Now loop over our age detection function we created
while True:
    # grab the frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #call the age detection function for each frame
    results = detect_and_predict_age(frame, faceNet, ageNet, minConf=args["confidence"])


    for r in results:
        #draw the bbox around the face and show the predicted age
        text = "{}: {:2f}%".format(r["age"][0], r["age"][1]*100)
        (startX, startY, endX, endY) = r["loc"]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanupq
cv2.destroyAllWindows()
vs.stop()

