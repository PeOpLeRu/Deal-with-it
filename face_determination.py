# face_determination

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect(img, classifier, scale=None, min_nbs=None):
    result = img.copy()
    rects = classifier.detectMultiScale(result, scaleFactor=scale, minNeighbors=min_nbs)    

    for (x, y, w, h) in rects:
        cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0))

    return result, rects

def drawGlasses(img, glasses, rects):
    result = img.copy()

    if len(rects) == 2:
        drawGlasses.init = True
        drawGlasses.max_y = np.max([rects[0][1] + rects[0][3], rects[1][1] + rects[1][3]])
        drawGlasses.min_y = np.min([rects[0][1], rects[1][1]])
        drawGlasses.max_x = np.max([rects[0][0] + rects[0][2], rects[1][0] + rects[1][2]])
        drawGlasses.min_x = np.min([rects[0][0], rects[1][0]])

        height = drawGlasses.max_y - drawGlasses.min_y
        width = drawGlasses.max_x - drawGlasses.min_x

        drawGlasses.scaled_glasses = cv.resize(glasses, (width, height))
        
        mask_glasses = drawGlasses.scaled_glasses[:, :, 3]
        drawGlasses.pos_y, drawGlasses.pos_x = np.where(mask_glasses == 255)
        
    if drawGlasses.init:
        result[drawGlasses.min_y:drawGlasses.max_y, drawGlasses.min_x:drawGlasses.max_x][drawGlasses.pos_y, drawGlasses.pos_x] = drawGlasses.scaled_glasses[drawGlasses.pos_y, drawGlasses.pos_x, :3]

    return result

drawGlasses.init = False

# load imgages and classifiers

conf = cv.imread("solvay_conference.jpg")
cooper = cv.imread("cooper.jpg")
pict_glasses = cv.imread("dealwithit.png", cv.IMREAD_UNCHANGED)

casc_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
casc_eyes = cv.CascadeClassifier("haarcascade_eye.xml")
casc_lbp_face = cv.CascadeClassifier("lbpcascade_frontalface.xml")

cv.namedWindow("Camera", cv.WINDOW_GUI_NORMAL)
cam = cv.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()

    frame_with_eyes, rects = detect(frame, casc_eyes, 1.2, 5)

    frame_with_glasses = drawGlasses(frame, pict_glasses, rects)

    cv.imshow("Camera", frame_with_glasses)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv.destroyAllWindows()